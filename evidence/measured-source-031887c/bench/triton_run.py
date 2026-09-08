"""Triton check -> tuning-only search/freeze -> immutable holdout evaluation.

GPU imports are lazy. Each GPU job runs in a bounded process group. Output
directories must be new; raw failures are preserved and cannot yield a freeze.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import time

from .api import ROOT, Shape, inputs, error_metrics
from .check import boundary_shapes
from .cuda_run import precision, timing, validate_correctness, identity_gpu, shape_worker
from .fp64_points import prepare_reference, check_points
from .run import source_identity, sha, atomic_json
from backends.triton import BLOCKS, supported

METRICS = ('t_device_op_graph_us', 't_api_us', 't_host_feed_us')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_protocol(path):
    p = json.loads(Path(path).read_text())
    if (p['blocks'] != list(BLOCKS) or p['num_warps'] != 4 or p['num_stages'] != 1
            or p['enable_fp_fusion'] is not False or p['explicit_fma'] is not True
            or p['tf32'] is not False or p['dtype'] != 'float32' or p['layout'] != 'NCHW'
            or p['batches'] != 5 or p['samples_per_batch'] != 30):
        raise ValueError('frozen Triton protocol altered')
    for key in ('repeat_cap', 'repeat_target_us', 'warmup', 'shape_timeout_seconds',
                'total_budget_seconds', 'fp64_points'):
        if not isinstance(p[key], int) or p[key] <= 0:
            raise ValueError('invalid protocol bound: ' + key)
    if p['atol'] != 1e-4 or p['rtol'] != 1e-4 or not p['correctness_seeds']:
        raise ValueError('correctness protocol altered')
    return p


def split_shapes(atlas, split):
    shapes = [Shape.from_record(r) for r in atlas['shapes']]
    ids = {s.id for s in shapes}
    tune, hold = split['tuning'], split['holdout']
    if (len(ids) != len(shapes) or len(set(tune)) != len(tune)
            or len(set(hold)) != len(hold) or set(tune) & set(hold)
            or set(tune) | set(hold) != ids):
        raise ValueError('split must be a disjoint, complete partition')
    return {k: [s for s in shapes if s.id in set(v)]
            for k, v in (('tuning', tune), ('holdout', hold))}


def summarize(samples, p):
    expected = {(b, s) for b in range(p['batches']) for s in range(p['samples_per_batch'])}
    if len(samples) != len(expected) or {(r['batch'], r['sample']) for r in samples} != expected:
        raise ValueError('incomplete or duplicated raw sample grid')
    result = {}
    for metric in METRICS:
        if any(not math.isfinite(r[metric]) or r[metric] <= 0 for r in samples):
            raise ValueError('nonpositive/nonfinite timing')
        medians = [statistics.median(r[metric] for r in samples if r['batch'] == b)
                   for b in range(p['batches'])]
        result[metric] = {'batch_medians': medians, 'median_batch_medians': statistics.median(medians)}
    return result


def select_blocks(records, shapes, p):
    """Fail closed: every supported tuning shape and candidate must be complete."""
    expected = {(s.id, block) for s in shapes if supported(s) for block in BLOCKS}
    rows = {(r['shape_id'], r['block']): r for r in records}
    if len(rows) != len(records) or set(rows) != expected:
        raise ValueError('candidate evidence set differs from tuning subset')
    scores, selected = {}, {}
    for cpg in (1, 2, 4):
        group = [s for s in shapes if supported(s) and s.cin // s.groups == cpg]
        if not group:
            raise ValueError('missing cpg tuning group')
        scores[str(cpg)] = {}
        for block in BLOCKS:
            values = []
            for s in group:
                row = rows[s.id, block]
                if row['status'] != 'PASS':
                    raise ValueError('failed tuning candidate forbids freeze')
                validate_candidate(row, s, p, [p['seed']], False)
                summary = summarize(row['samples'], p)
                values.append(summary['t_device_op_graph_us']['median_batch_medians'])
            scores[str(cpg)][str(block)] = math.exp(sum(map(math.log, values)) / len(values))
        selected[str(cpg)] = min(BLOCKS, key=lambda b: (scores[str(cpg)][str(b)], b))
    return selected, scores


def device_environment():
    import torch
    import triton
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device unavailable; device correctness/performance NOT_RUN')
    driver = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version',
                                     '--format=csv,noheader'],text=True,timeout=10).strip()
    if not driver: raise RuntimeError('driver identity unavailable')
    cudnn_version = torch.backends.cudnn.version()
    if not isinstance(cudnn_version,int) or cudnn_version <= 0:
        raise RuntimeError('cuDNN runtime identity unavailable')
    return {'gpu': identity_gpu(torch), 'torch': str(torch.__version__),
            'torch_git':torch.version.git_version, 'cudnn':cudnn_version,
            'driver_uuid_versions':driver, 'cuda': torch.version.cuda, 'triton': triton.__version__}


def validate_candidate(row, shape, p, seeds, full_output):
    """Validate evidence contents, not just PASS labels and latency samples."""
    checks, compiled = row.get('correctness', []), row.get('compile', [])
    if (len(checks) != len(seeds) or {c.get('seed') for c in checks} != set(seeds)
            or len(compiled) != len(seeds) or {c.get('seed') for c in compiled} != set(seeds)):
        raise ValueError('candidate correctness/compile evidence missing or duplicated')
    for check in checks:
        fp32 = check.get('torch_cuda_fp32_full', {})
        if (fp32.get('passed') is not True or fp32.get('failed_elements') != 0
                or fp32.get('elements') != math.prod(shape.output)
                or fp32.get('atol') != p['atol'] or fp32.get('rtol') != p['rtol']
                or any(not isinstance(fp32.get(k),(int,float)) or not math.isfinite(fp32[k])
                       or fp32[k] < 0 for k in ('max_abs','max_rel','rms'))):
            raise ValueError('candidate full FP32 evidence invalid')
        x,w = inputs(shape, check['seed'])
        expected = prepare_reference(shape,x,w,seed=check['seed'],
            count=math.prod(shape.output) if full_output else p['fp64_points'])
        fp64 = check.get('fp64', {})
        for key,value in expected.items():
            if key not in ('positions','expected_fp64') and fp64.get(key) != value:
                raise ValueError('candidate FP64 reference metadata mismatch')
        points = fp64.get('points', [])
        if (fp64.get('passed') is not True or fp64.get('failed_points') != 0
                or fp64.get('atol') != p['atol'] or fp64.get('rtol') != p['rtol']
                or len(points) != expected['point_count']):
            raise ValueError('candidate FP64 point coverage missing')
        for point,position,value in zip(points,expected['positions'],expected['expected_fp64']):
            actual = point.get('actual')
            if (point.get('position') != position or point.get('expected_fp64') != value
                    or point.get('finite') is not True or point.get('passed') is not True
                    or not isinstance(actual,(int,float)) or not math.isfinite(actual)
                    or abs(actual-value) > p['atol']+p['rtol']*abs(value)):
                raise ValueError('candidate FP64 point value invalid')
    for item in compiled:
        if (item.get('block') != row['block'] or item.get('num_warps') != p['num_warps']
                or item.get('num_stages') != p['num_stages'] or item.get('enable_fp_fusion') is not False
                or item.get('explicit_fma') is not True or not item.get('metadata')
                or not isinstance(item.get('first_call_jit_and_launch_us'),(int,float))
                or not math.isfinite(item['first_call_jit_and_launch_us'])
                or item['first_call_jit_and_launch_us'] <= 0):
            raise ValueError('candidate compile metadata invalid')
        path = Path(item.get('ptx_path',''))
        if not path.is_file() or not path.read_text().strip() or sha(path) != item.get('ptx_sha256'):
            raise ValueError('candidate PTX artifact missing or changed')


def gpu_worker(conn, record, p, blocks, artifact_dir, check_only):
    if hasattr(os, 'setsid'):
        os.setsid()
    result = {'shape': record, 'rows': [], 'device_correctness': 'NOT_RUN', 'performance': 'NOT_RUN'}
    try:
        import torch
        import torch.nn.functional as F
        from backends.triton import make_call
        shape = Shape.from_record(record)
        env = device_environment()
        if env['triton'] != p['triton_version']:
            raise RuntimeError('Triton version mismatch: ' + env['triton'])
        result.update(environment=env, precision=precision(torch))
        if not supported(shape):
            result['rows'] = [{'shape_id': shape.id, 'block': b, 'status': 'UNSUPPORTED',
                               'samples': []} for b in blocks]
            return
        stream = torch.cuda.Stream()
        calls, rowmap, repeats = {}, {}, {}
        with torch.inference_mode(), torch.cuda.stream(stream):
            # Boundary mode checks all seeds and all output points independently.
            seeds = p['correctness_seeds'] if check_only else [p['seed']]
            for seed in seeds:
                x, w = inputs(shape, seed)
                tx, tw = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
                reference = F.conv2d(tx, tw, padding=1, groups=shape.groups)
                stream.synchronize()
                reference_cpu = reference.cpu().numpy()
                oracle = prepare_reference(shape, x, w, seed=seed,
                    count=math.prod(shape.output) if check_only else p['fp64_points'])
                order = list(blocks)
                random.Random(f"{p['seed']}:{shape.id}:compile:{seed}").shuffle(order)
                for block in order:
                    row = rowmap.setdefault(block, {'shape_id': shape.id, 'block': block,
                        'status': 'PASS', 'correctness': [], 'samples': [], 'compile': []})
                    try:
                        output = torch.empty(shape.output, device=tx.device, dtype=torch.float32)
                        call = make_call(shape, tx, tw, output, block)
                        stream.synchronize(); start = time.perf_counter_ns()
                        call(); stream.synchronize()
                        first_us = (time.perf_counter_ns() - start) / 1000
                        actual = output.cpu().numpy()
                        full = error_metrics(actual, reference_cpu, atol=p['atol'], rtol=p['rtol'])
                        points = check_points(oracle, actual, atol=p['atol'], rtol=p['rtol'])
                        row['correctness'].append({'seed': seed, 'torch_cuda_fp32_full': full,
                                                   'fp64': points})
                        if not full['passed'] or not points['passed']:
                            raise RuntimeError('full FP32 or independent FP64 correctness failed')
                        ptx = call.compiled.asm.get('ptx')
                        if not isinstance(ptx, str) or not ptx.strip():
                            raise RuntimeError('compiled PTX missing')
                        path = Path(artifact_dir) / f'{shape.id}-b{block}-seed{seed}.ptx'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(ptx)
                        row['compile'].append({'seed': seed, 'first_call_jit_and_launch_us': first_us,
                            'ptx_path': str(path), 'ptx_sha256': sha(path),
                            'metadata': str(call.compiled.metadata), 'block': block,
                            'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False,
                            'explicit_fma': True})
                        if not check_only:
                            calls[block] = call
                            for _ in range(p['warmup']): call()
                            stream.synchronize()
                            pilot = timing(call, torch, stream, 1)
                            repeats[block] = max(1, min(p['repeat_cap'],
                                int(p['repeat_target_us'] / max(pilot['t_api_us'], 1))))
                            timing(call, torch, stream, repeats[block])
                    except Exception as exc:
                        row.update(status='OOM' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'FAIL',
                                   reason=str(exc))
            if not check_only:
                for batch in range(p['batches']):
                    order = list(blocks)
                    random.Random(f"{p['seed']}:{shape.id}:{batch}").shuffle(order)
                    for position, block in enumerate(order):
                        row = rowmap[block]
                        if row['status'] != 'PASS': continue
                        try:
                            for sample in range(p['samples_per_batch']):
                                row['samples'].append({'shape_id': shape.id, 'block': block,
                                    'implementation': 'triton', 'batch': batch, 'sample': sample,
                                    'implementation_order': position, 'repeats': repeats[block],
                                    **timing(calls[block], torch, stream, repeats[block])})
                            row['summary'] = summarize(row['samples'], p) if batch == p['batches'] - 1 else None
                        except Exception as exc:
                            row.update(status='OOM' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'FAIL',
                                       reason=str(exc))
        result['rows'] = list(rowmap.values())
        result['device_correctness'] = 'PASS' if all(r['status'] == 'PASS' for r in result['rows']) else 'FAIL'
        result['performance'] = 'NOT_RUN' if check_only else result['device_correctness']
    except BaseException as exc:
        result.update(status='FAIL', reason=str(exc))
        result['rows'] = [{'shape_id': record['shape_id'], 'block': b, 'status': 'FAIL',
                          'samples': [], 'reason': str(exc)} for b in blocks]
    finally:
        conn.send(result)
        conn.close()


def isolated(target, args, seconds):
    context = mp.get_context('spawn')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(child, *args))
    try:
        process.start(); child.close()
        if not parent.poll(seconds):
            return {'status': 'TIMEOUT', 'reason': 'bounded worker timeout', 'rows': []}
        try:
            return parent.recv()
        except EOFError:
            return {'status': 'FAIL', 'reason': 'worker exited without receipt', 'rows': []}
    finally:
        try:
            # start() can be interrupted before a PID exists; never join then.
            if process.pid is not None:
                # Also kill nested tuned-worker descendants in the worker's session.
                if hasattr(os, 'killpg'):
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                elif process.is_alive():
                    process.kill()
                process.join(5)
                if process.is_alive(): process.kill(); process.join(5)
        finally:
            child.close()
            parent.close()


def require_binding(receipt, binding, environment):
    if receipt.get('status') != 'PASS' or receipt.get('binding') != binding:
        raise ValueError('receipt status/source/protocol/library binding mismatch')
    if receipt.get('environment') != environment:
        raise ValueError('receipt device/software environment mismatch')


def check_shapes():
    return boundary_shapes() + [Shape(n=3, cin=6, cout=6, groups=g, h=5, w=9) for g in (3, 6)]


def validate_check_receipt(receipt, binding, env, p):
    require_binding(receipt, binding, env)
    if receipt.get('kind') != 'triton_check' or receipt.get('device_correctness') != 'PASS':
        raise ValueError('expected completed Triton boundary check')
    shapes = {s.id: s for s in check_shapes()}
    jobs = receipt.get('jobs', [])
    if len(jobs) != len(shapes) or {j['shape_id'] for j in jobs} != set(shapes):
        raise ValueError('boundary check job coverage missing')
    for job in jobs:
        if sha(job['path']) != job['sha256']:
            raise ValueError('boundary evidence hash changed')
        data = json.loads(Path(job['path']).read_text())
        shape = shapes[job['shape_id']]
        if (job.get('suite') != 'triton' or data.get('suite') != 'triton'
                or data.get('binding') != binding or Shape.from_record(data['shape']) != shape):
            raise ValueError('boundary job identity mismatch')
        if supported(shape) and data.get('environment') != env:
            raise ValueError('boundary job environment mismatch')
        rows = data['rows']
        if len(rows) != len(BLOCKS) or {r['block'] for r in rows} != set(BLOCKS):
            raise ValueError('boundary block coverage missing')
        for row in rows:
            expected = 'PASS' if supported(shape) else 'UNSUPPORTED'
            if row['status'] != expected or row['shape_id'] != shape.id:
                raise ValueError('boundary support/correctness mismatch')
            if expected == 'UNSUPPORTED': continue
            validate_candidate(row,shape,p,p['correctness_seeds'],True)
            checks = row['correctness']
            if (len(checks) != len(p['correctness_seeds'])
                    or {c['seed'] for c in checks} != set(p['correctness_seeds'])):
                raise ValueError('boundary seed coverage missing')
            for check in checks:
                if (not check['torch_cuda_fp32_full']['passed'] or not check['fp64']['passed']
                        or check['fp64']['full_output'] is not True):
                    raise ValueError('full independent boundary correctness missing')
            if len(row['compile']) != len(p['correctness_seeds']):
                raise ValueError('boundary compiled artifact coverage missing')
            for item in row['compile']:
                if sha(item['ptx_path']) != item['ptx_sha256']:
                    raise ValueError('boundary PTX hash changed')


def validate_frozen(frozen, path, binding, env, p, tuning_shapes):
    require_binding(frozen, binding, env)
    if frozen.get('kind') != 'triton_frozen':
        raise ValueError('expected frozen configuration')
    receipt = json.loads((Path(path).parent/'receipt.json').read_text())
    require_binding(receipt, binding, env)
    if receipt.get('kind') != 'triton_tune' or receipt.get('frozen_sha256') != sha(path):
        raise ValueError('frozen file does not match completed tuning receipt')
    if (frozen.get('holdout_performance_accessed') is not False or
            frozen.get('tuning_ids') != [s.id for s in tuning_shapes if supported(s)]):
        raise ValueError('frozen tuning IDs/holdout exclusion mismatch')
    for entry in frozen['evidence']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('tuning/check evidence changed')
    check_entry = receipt['check_report']
    if sha(check_entry['path']) != check_entry['sha256']:
        raise ValueError('boundary receipt changed')
    validate_check_receipt(json.loads(Path(check_entry['path']).read_text()), binding, env, p)
    rows = []
    shapes_by_id = {s.id:s for s in tuning_shapes}
    if (len(receipt['jobs']) != len(tuning_shapes) or
            {j['shape_id'] for j in receipt['jobs']} != set(shapes_by_id)):
        raise ValueError('tuning job coverage mismatch')
    expected_evidence = list(receipt['jobs']) + [check_entry]
    for job in receipt['jobs']:
        if sha(job['path']) != job['sha256']:
            raise ValueError('tuning raw samples changed')
        data = json.loads(Path(job['path']).read_text())
        shape = shapes_by_id[job['shape_id']]
        if (job.get('suite') != 'triton' or data.get('suite') != 'triton'
                or data.get('binding') != binding or Shape.from_record(data['shape']) != shape):
            raise ValueError('tuning job identity mismatch')
        if supported(shape):
            if (data.get('environment') != env or data.get('device_correctness') != 'PASS'
                    or data.get('performance') != 'PASS'):
                raise ValueError('tuning job environment/correctness/performance mismatch')
            rows.extend(data['rows'])
            expected_evidence.extend({'path':item['ptx_path'],'sha256':item['ptx_sha256']}
                                    for row in data['rows'] for item in row.get('compile',[]))
        elif (len(data['rows']) != len(BLOCKS) or {r['block'] for r in data['rows']} != set(BLOCKS)
              or any(r['shape_id'] != shape.id or r['status'] != 'UNSUPPORTED' for r in data['rows'])):
            raise ValueError('unsupported tuning denominator mismatch')
    def evidence_set(items):
        values = [(str(Path(e['path']).resolve()),e['sha256']) for e in items]
        if len(set(v[0] for v in values)) != len(values): raise ValueError('duplicate evidence path')
        return set(values)
    if evidence_set(frozen['evidence']) != evidence_set(expected_evidence):
        raise ValueError('frozen evidence set does not close over jobs/check/PTX')
    selected, scores = select_blocks(rows, tuning_shapes, p)
    if frozen['selected'] != selected or frozen['scores_graph_us'] != scores:
        raise ValueError('frozen selection differs from tuning raw samples')


def main(argv=None):
    # SIGTERM must unwind the active isolated() finally block before returning.
    def terminate(signum, frame):
        raise SystemExit('terminated by signal ' + str(signum))
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('check', 'tune', 'evaluate'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/triton-protocol.json')
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--correctness', type=Path, required=True)
    parser.add_argument('--check-report', type=Path)
    parser.add_argument('--frozen', type=Path)
    args = parser.parse_args(argv)
    p = load_protocol(args.protocol)
    source = source_identity()
    library = args.library.resolve()
    validate_correctness(json.loads(args.correctness.read_text()), source, sha(library))
    split = json.loads((ROOT/'configs/split.json').read_text())
    subsets = split_shapes(json.loads((ROOT/'configs/atlas.json').read_text()), split)
    binding = {'source_tree_sha256': source['source_tree_sha256'], 'protocol_sha256': sha(args.protocol),
               'split_sha256': sha(ROOT/'configs/split.json'), 'library_sha256': sha(library),
               'native_correctness_sha256': sha(args.correctness)}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # A run-owned, initially nonexistent cache: first-call cost includes JIT or
    # within-run cache reuse, never a claimed pure compiler time.
    os.environ['TRITON_CACHE_DIR'] = str(out/'jit-cache')
    started = time.monotonic()
    manifest = {'kind': 'triton_' + args.phase, 'status': 'RUNNING', 'source': source,
        'binding': binding, 'protocol': p, 'source_status': 'IMPLEMENTED',
        'device_correctness': 'NOT_RUN', 'performance': 'NOT_RUN', 'jobs': []}
    try:
        env = device_environment()
        manifest['environment'] = env
        if env['triton'] != p['triton_version']:
            raise ValueError('requires Triton ' + p['triton_version'])
        if args.phase == 'check':
            shapes = check_shapes()
        elif args.phase == 'tune':
            if not args.check_report: raise ValueError('--check-report is required')
            check = json.loads(args.check_report.read_text())
            validate_check_receipt(check, binding, env, p)
            shapes = subsets['tuning']
            manifest['check_report'] = {'path': str(args.check_report.resolve()), 'sha256': sha(args.check_report)}
        else:
            if not args.frozen: raise ValueError('--frozen is required')
            frozen = json.loads(args.frozen.read_text())
            validate_frozen(frozen, args.frozen, binding, env, p, subsets['tuning'])
            manifest['frozen'] = {'path': str(args.frozen.resolve()), 'sha256': sha(args.frozen)}
            shapes = subsets['holdout']
        atomic_json(out/'receipt.json', manifest)
        allrows = []
        for shape in shapes:
            if args.phase == 'evaluate' and supported(shape):
                blocks = [frozen['selected'][str(shape.cin // shape.groups)]]
            else: blocks = list(BLOCKS)
            tasks = ['triton', 'cuda'] if args.phase == 'evaluate' and supported(shape) else ['triton']
            random.Random(f"{p['seed']}:{shape.id}:suites").shuffle(tasks)
            for suite in tasks:
                remaining = p['total_budget_seconds'] - (time.monotonic() - started)
                if remaining <= 0: raise TimeoutError('total phase budget exhausted')
                if suite == 'triton':
                    result = isolated(gpu_worker, (shape.record(), p, blocks, str(out/'ptx'),
                        args.phase == 'check'), min(p['shape_timeout_seconds'], remaining))
                else:
                    result = isolated(shape_worker, (shape.record(), p, str(library)),
                                      min(p['shape_timeout_seconds'], remaining))
                result.update(binding=binding, suite=suite)
                path = out/'shapes'/f'{shape.id}-{suite}.json'
                atomic_json(path, result)
                manifest['jobs'].append({'path': str(path), 'sha256': sha(path), 'suite': suite,
                                         'shape_id': shape.id})
                atomic_json(out/'receipt.json', manifest)
                rows = result.get('rows', [])
                if not rows or any(r['status'] not in ('PASS', 'UNSUPPORTED') for r in rows):
                    raise RuntimeError('shape job failed: ' + shape.id + ': ' + suite)
                if supported(shape) and any(r['status'] != 'PASS' for r in rows):
                    raise RuntimeError('supported shape rejected')
                if suite == 'cuda':
                    for row in rows:
                        if row['status'] == 'PASS':
                            row_samples = [r for r in result['samples']
                                           if r['implementation'] == row['implementation']]
                            summarize(row_samples, p)
                if suite == 'triton' and supported(shape): allrows.extend(rows)
        manifest.update(status='PASS', device_correctness='PASS',
                        performance='NOT_RUN' if args.phase == 'check' else 'PASS',
                        elapsed_seconds=time.monotonic() - started,
                        denominator={'all_shapes': len(shapes), 'supported': sum(map(supported, shapes)),
                                     'unsupported': sum(not supported(s) for s in shapes)})
        if args.phase == 'tune':
            selected, scores = select_blocks(allrows, shapes, p)
            frozen = {'kind': 'triton_frozen', 'status': 'PASS', 'binding': binding,
                'environment': env, 'selected': selected, 'scores_graph_us': scores,
                'tuning_ids': [s.id for s in shapes if supported(s)],
                'holdout_performance_accessed': False,
                'search_elapsed_seconds': time.monotonic() - started,
                'evidence': manifest['jobs'] + [manifest['check_report']] + [
                    {'path': item['ptx_path'], 'sha256': item['ptx_sha256']}
                    for row in allrows for item in row['compile']]}
            atomic_json(out/'frozen.json', frozen)
            manifest['frozen_sha256'] = sha(out/'frozen.json')
    except BaseException as exc:
        manifest.update(status='FAIL', reason=str(exc), elapsed_seconds=time.monotonic() - started)
        atomic_json(out/'receipt.json', manifest)
        raise
    atomic_json(out/'receipt.json', manifest)
    print(json.dumps({'status': manifest['status'], 'receipt': str(out/'receipt.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
