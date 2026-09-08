"""One shape, one native implementation, one profiler-enabled logical call.

No benchmark workers or compilation. Correctness and warmup are outside
cudaProfilerStart/Stop; ncu must use --profile-from-start off.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

from .api import ROOT, Shape, Native, inputs, error_metrics
from .check import expected_support
from .cuda_run import precision, identity_gpu, validate_correctness
from .fp64_points import prepare_reference, check_points
from .run import atomic_json, sha, source_identity

KERNEL_IDENTIFIERS = {0: 'naive', 1: 'spatial', 2: 'shared_halo', 3: 'register_outputs'}


def select_shape(shape_id):
    atlas = json.loads((ROOT/'configs/atlas.json').read_text())
    matches = [Shape.from_record(r) for r in atlas['shapes'] if r['shape_id'] == shape_id]
    if len(matches) != 1 or matches[0].id != shape_id:
        raise ValueError('shape ID must identify one frozen Atlas record')
    return matches[0]


def run(args, receipt):
    p = json.loads(args.protocol.read_text())
    if p['warmup'] != 10 or p['atol'] != 1e-4 or p['rtol'] != 1e-4:
        raise ValueError('frozen correctness/warmup protocol mismatch')
    shape = select_shape(args.shape_id)
    source = source_identity()
    receipt.update(shape=shape.record(), implementation=args.impl,
                   expected_kernel_identifier=KERNEL_IDENTIFIERS[args.impl],
                   source=source, protocol=p, protocol_sha256=sha(args.protocol),
                   library_sha256=sha(args.library), native_correctness_sha256=sha(args.correctness))
    if not expected_support('cuda', shape, args.impl):
        receipt.update(status='UNSUPPORTED', reason='native support contract')
        return 2
    validate_correctness(json.loads(args.correctness.read_text()), source, sha(args.library))
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        receipt.update(status='NOT_RUN', reason='real CUDA unavailable')
        return 2
    settings = precision(torch)
    native = Native(args.library)
    stream = torch.cuda.Stream(device=0)
    x, w = inputs(shape, p['seed'])
    points = prepare_reference(shape, x, w, seed=p['seed'], count=p['fp64_points'])
    receipt.update(gpu=identity_gpu(torch), torch=str(torch.__version__), cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), precision=settings, process_id=os.getpid(),
                   native_build=native.lib.gc_build_info().decode(), stream='non-default')
    with torch.inference_mode(), torch.cuda.stream(stream):
        tx, tw = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
        output = torch.empty(shape.output, device=tx.device, dtype=torch.float32)
        reference = F.conv2d(tx, tw, padding=1, groups=shape.groups)
        def call():
            return native.cuda(shape, tx, tw, output, args.impl, stream.cuda_stream)
        def check():
            stream.synchronize()
            actual = output.cpu().numpy()
            full = error_metrics(actual, reference.cpu().numpy(), atol=p['atol'], rtol=p['rtol'])
            oracle = check_points(points, actual, atol=p['atol'], rtol=p['rtol'])
            if not full['passed'] or not oracle['passed']:
                raise RuntimeError('full FP32 or independent FP64 correctness failed')
            return {'torch_cuda_fp32_full': full, 'independent_fp64_points': oracle}
        call()
        receipt['correctness_before_profile'] = check()
        for _ in range(p['warmup']): call()
        stream.synchronize()
        label = f'groupconv_target::{shape.id}::k{args.impl}'
        receipt.update(status='READY', warmup_calls=p['warmup'], nvtx_label=label,
                       profiler_api='torch.cuda.profiler.start/stop -> cudaProfilerStart/Stop',
                       target_logical_calls=0,
                       boundary='one native call and stream completion between profiler start/stop; no graph')
        atomic_json(args.output, receipt)
        torch.cuda.profiler.start()
        try:
            torch.cuda.nvtx.range_push(label)
            try:
                call()
                stream.synchronize()
                receipt['target_logical_calls'] = 1
            finally:
                torch.cuda.nvtx.range_pop()
        finally:
            torch.cuda.profiler.stop()
        receipt['correctness_after_profile'] = check()
    receipt.update(status='PASS', counter_availability='UNKNOWN: inspect ncu report independently')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape-id', required=True)
    parser.add_argument('--impl', required=True, type=int, choices=range(4))
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--correctness', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new correctness JSON file')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/profile-protocol.json')
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error('output already exists')
    receipt = {'kind': 'single_native_profile_target', 'status': 'RUNNING',
               'counter_availability': 'NOT_VERIFIED'}
    started = time.monotonic()
    try:
        code = run(args, receipt)
    except BaseException as exc:
        receipt.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
        code = 1
    receipt['elapsed_seconds_including_profiler'] = time.monotonic() - started
    atomic_json(args.output, receipt)
    print(json.dumps({'status': receipt['status'], 'output': str(args.output)}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
