"""Authoritative full-output correctness coverage; unavailable is never PASS."""
from __future__ import annotations
import argparse
import ctypes as ct
import dataclasses
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import time
from collections import Counter

import numpy as np
from .api import (ROOT, Shape, Native, NativeError, OpenCL, cshape, tensor, native_int,
                  inputs, torch_reference, oracle_points, error_metrics)

SEEDS = (20260908, 17, 101)
ERROR = {'atol': 1e-4, 'rtol': 1e-4, 'eps': 1e-12}


def _reference_worker(connection):
    # Separate address space avoids loading PyTorch's bundled OpenMP and the
    # native library's OpenMP together. No duplicate-runtime override is used.
    import torch
    torch.set_num_threads(1)
    connection.send(str(torch.__version__))
    while True:
        task = connection.recv()
        if task is None:
            break
        try:
            connection.send(('OK', torch_reference(*task)))
        except Exception as exc:
            connection.send(('ERROR', str(exc)))
    connection.close()


class ReferenceWorker:
    def __init__(self):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_reference_worker, args=(child,))
        self.process.start()
        child.close()
        self.version = self.connection.recv()

    def evaluate(self, shape, x, w):
        self.connection.send((shape, x, w))
        status, value = self.connection.recv()
        if status != 'OK':
            raise RuntimeError(value)
        return value

    def close(self):
        if self.process.is_alive():
            self.connection.send(None)
        self.connection.close()
        self.process.join(10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


def boundary_shapes():
    base = Shape(cin=4, cout=4, groups=2, h=3, w=5)
    values = [dataclasses.replace(base, h=h, w=w, groups=g, cin=c, cout=c)
              for h, w in [(1, 1), (1, 19), (11, 1), (7, 13), (9, 17)]
              for c, g in [(4, 4), (4, 2), (4, 1)]]
    values += [dataclasses.replace(base, cin=6, cout=6, groups=2, h=5, w=9),
               dataclasses.replace(base, n=4, h=3, w=7),
               dataclasses.replace(base, cin=4, cout=6, groups=2),
               dataclasses.replace(base, cin=3, cout=6, groups=3),
               dataclasses.replace(base, h=9, w=11, sh=2, sw=2, ph=0, pw=0),
               dataclasses.replace(base, h=9, w=11, dh=2, dw=2),
               dataclasses.replace(base, h=9, w=11, r=5, s=5, ph=2, pw=2),
               dataclasses.replace(base, h=9, w=11, r=7, s=7, ph=3, pw=3)]
    return values


def course_shapes():
    path = ROOT / 'configs/course.json'
    if path.exists():
        value = json.loads(path.read_text())
        rows = value if isinstance(value, list) else value['shapes']
        return [Shape.from_record(r) for r in rows]
    # Exact PLAN course set; useful while a source-only checkout is being built.
    return [Shape(cin=c, cout=c, h=h, w=h, groups=g)
            for c in (16, 32, 64) for h in (16, 32, 64) for g in (1, 2, 4, c)]


def positions(shape, full=False):
    n, c, h, w = shape.output
    if full:
        return list(itertools.product(range(n), range(c), range(h), range(w)))
    return sorted(set(itertools.product({0, n - 1}, {0, c // 2, c - 1},
                                       {0, h // 2, h - 1}, {0, w // 2, w - 1})))


def variants(backend):
    if backend == 'cpu':
        return [(0, 1), (1, 1), (1, 2), (2, 1), (2, 2)]
    return [(i, None) for i in range(4 if backend == 'cuda' else 3)]


def expected_support(backend, shape, impl):
    b = (shape.cin == shape.cout and shape.r == shape.s == 3 and
         shape.sh == shape.sw == shape.dh == shape.dw == 1 and
         shape.ph == shape.pw == 1)
    if backend == 'cpu' or (backend == 'cuda' and impl == 0):
        return True
    return b and (backend != 'cuda' or impl != 2 or shape.cin // shape.groups in (1, 2, 4))


def contract_checks(native, opencl=None):
    """Actual ABI calls for metadata and bounds; separate from numerical passes."""
    shape = Shape(cin=2, cout=2, groups=2, h=2, w=3)
    x, w = inputs(shape)
    y = np.empty(shape.output, np.float32)
    checks = []
    cases = [('dtype', 1), ('rank', 1), ('strides', 2), ('misaligned', 1),
             ('device', 1), ('buffer_bytes', 1), ('alias', 1), ('groups', 1),
             ('geometry', 1), ('implementation', 1)]
    for name, expected in cases:
        p, xd, wd, yd = cshape(shape), tensor(x), tensor(w), tensor(y)
        impl = 0
        if name == 'dtype': xd.dtype = 99
        elif name == 'rank': xd.rank = 3
        elif name == 'strides': xd.strides[2] += 1
        elif name == 'misaligned': xd.data += 1
        elif name == 'device': xd.device = 1
        elif name == 'buffer_bytes': xd.bytes -= 4
        elif name == 'alias': yd.data = xd.data
        elif name == 'groups': p.groups = 3
        elif name == 'geometry': p.sh = 0
        elif name == 'implementation': impl = 99
        actual = native.lib.gc_cpu_conv(ct.byref(p), ct.byref(xd), ct.byref(wd),
                                        ct.byref(yd), impl, 1)
        checks.append({'case': name, 'status': 'PASS' if actual == expected else 'FAIL',
                       'expected_status': expected, 'actual_status': actual})
    for nonfinite in (np.nan, np.inf, -np.inf):
        m = error_metrics(np.array([nonfinite]), np.array([0.]))
        checks.append({'case': 'reject_nonfinite_' + str(nonfinite),
                       'status': 'PASS' if not m['passed'] and m['failed_elements'] == 1 else 'FAIL'})
    for name, function in [('python_shape_overflow', lambda: cshape(dataclasses.replace(shape, n=2**64 + 1))),
                           ('python_impl_overflow', lambda: native_int(2**32)),
                           ('python_nonintegral', lambda: native_int(1.5))]:
        try:
            function()
            status = 'FAIL'
        except ValueError:
            status = 'PASS'
        checks.append({'case': name, 'status': status})
    if opencl is not None:
        opencl.upload(shape, x, w, y)
        yd = tensor(y)
        actual = native.lib.gc_opencl_download(opencl.context, ct.byref(yd))
        checks.append({'case': 'opencl_download_before_run', 'expected_status': 1,
                       'actual_status': actual, 'status': 'PASS' if actual == 1 else 'FAIL'})
        opencl.run(0)
        yd.bytes = 2**64 - 1
        actual = native.lib.gc_opencl_download(opencl.context, ct.byref(yd))
        checks.append({'case': 'opencl_download_pointer_range_overflow', 'expected_status': 1,
                       'actual_status': actual, 'status': 'PASS' if actual == 1 else 'FAIL'})
    return checks


def run(backend='all', limit=None, shape_set='course'):
    started = time.monotonic()
    native = Native()
    worker = ReferenceWorker()
    selected = ['cpu', 'opencl', 'cuda'] if backend == 'all' else [backend]
    availability, contexts = {}, {}
    for name in selected:
        try:
            if name == 'opencl':
                contexts[name] = OpenCL(native)
                availability[name] = {'available': True, 'identity': contexts[name].info}
            elif name == 'cuda':
                available = bool(native.lib.gc_cuda_available())
                if available:
                    import torch
                    available = torch.cuda.is_available()
                availability[name] = {'available': available,
                                      'reason': None if available else 'Native CUDA or torch CUDA unavailable'}
                if available:
                    contexts[name] = torch.cuda.Stream()
                    availability[name]['identity'] = torch.cuda.get_device_name()
            else:
                availability[name] = {'available': True}
        except (NativeError, RuntimeError) as exc:
            availability[name] = {'available': False, 'reason': str(exc)}
    main_shapes = course_shapes() if shape_set == 'course' else boundary_shapes()
    if limit is not None:
        main_shapes = main_shapes[:limit]
    # --limit is a smoke limit for course rows, not permission to omit edge tests.
    shapes = [(s, shape_set) for s in main_shapes]
    if shape_set == 'course':
        shapes += [(s, 'boundary') for s in boundary_shapes()]
        batch4_path=ROOT/'configs/course-batch4.json'
        if batch4_path.exists():
            shapes += [(Shape.from_record(s),'batch4') for s in json.loads(batch4_path.read_text())['shapes']]
    records = []
    try:
        checks = contract_checks(native, contexts.get('opencl'))
        for shape, suite in shapes:
            full = suite == 'boundary'
            pos = positions(shape, full)
            cases = list(SEEDS)
            if suite == 'boundary' and shape == boundary_shapes()[4]:
                cases += ['zeros', 'impulse', 'group_isolation', 'cancellation']
            for seed in cases:
                x, w = inputs(shape, seed if isinstance(seed, int) else SEEDS[0])
                if seed == 'zeros':
                    x.fill(0)
                elif seed == 'impulse':
                    x.fill(0)
                    x[0, 0, 0, shape.w // 2] = 1
                    w.fill(1)
                elif seed == 'group_isolation':
                    x.fill(0)
                    x[:, :shape.cin // shape.groups] = 1
                    w.fill(1)
                elif seed == 'cancellation':
                    x.fill(1)
                    w.reshape(-1)[::2] = 1
                    w.reshape(-1)[1::2] = -1
                ref = worker.evaluate(shape, x, w)
                oracle = oracle_points(shape, x, w, pos)
                references = [('torch_cpu_fp32_full', ref),
                              ('python_fp64_full' if full else 'python_fp64_fixed_points', oracle)]
                for name in selected:
                    for impl, threads in variants(name):
                        common = {**shape.record(), 'suite': suite, 'backend': name,
                                  'implementation': impl, 'threads': threads,
                                  'expected_supported': expected_support(name, shape, impl),
                                  'seed': seed if isinstance(seed, int) else None,
                                  'input_pattern': 'random' if isinstance(seed, int) else seed}
                        status, reason, actual = 'PASS', None, None
                        if not availability[name]['available']:
                            status, reason = 'NOT_RUN', availability[name].get('reason')
                        else:
                            try:
                                y = np.full(shape.output, np.nan, np.float32)
                                if name == 'cpu':
                                    actual = native.cpu(shape, x, w, y, impl, threads)
                                elif name == 'opencl':
                                    cl = contexts[name]
                                    cl.upload(shape, x, w, y)
                                    cl.run(impl)
                                    actual = cl.download(y)
                                else:
                                    stream = contexts[name]
                                    with torch.cuda.device(stream.device), torch.cuda.stream(stream):
                                        dx = torch.from_numpy(x).to(stream.device)
                                        dw = torch.from_numpy(w).to(stream.device)
                                        dy = torch.full(shape.output, float('nan'), device=stream.device)
                                        native.cuda(shape, dx, dw, dy, impl, stream.cuda_stream)
                                        done = torch.cuda.Event()
                                        done.record(stream)
                                        done.synchronize()
                                        actual = dy.cpu().numpy()
                            except NativeError as exc:
                                status = {2: 'UNSUPPORTED', 4: 'NOT_RUN'}.get(exc.status, 'FAIL')
                                reason = str(exc)
                            except RuntimeError as exc:
                                status, reason = 'FAIL', str(exc)
                        if status == 'UNSUPPORTED' and common['expected_supported']:
                            status, reason = 'FAIL', 'Unexpected support rejection: ' + str(reason)
                        if actual is not None and not common['expected_supported']:
                            status, reason = 'FAIL', 'Implementation accepted geometry outside declared support'
                            actual = None
                        for reference, expected in references:
                            metrics = None
                            result_status = status
                            if actual is not None:
                                values = actual if reference == 'torch_cpu_fp32_full' else np.array([actual[p] for p in pos])
                                metrics = error_metrics(values, expected, **ERROR)
                                result_status = 'PASS' if metrics['passed'] else 'FAIL'
                            records.append({**common, 'reference': reference, 'status': result_status,
                                            'reason': reason, 'metrics': metrics})
    finally:
        worker.close()
        if 'opencl' in contexts:
            contexts['opencl'].close()
    counts = dict(Counter(r['status'] for r in records))
    for status in ('PASS', 'FAIL', 'UNSUPPORTED', 'NOT_RUN'):
        counts.setdefault(status, 0)
    from .run import source_identity
    return {'schema_version': 1, 'kind': 'correctness', 'requested_backend': backend,
            'source': source_identity(),
            'shape_set': shape_set, 'limit': limit, 'seeds': SEEDS, 'error_config': ERROR,
            'oracle_position_policy': {'boundary': 'all outputs',
              'course': 'Cartesian unique n={0,last}, co={0,center,last}, oh/ow={0,center,last}'},
            'availability': availability, 'library': {'path': native.path,
               'sha256': hashlib.sha256(Path(native.path).read_bytes()).hexdigest(),
               'build_info': native.lib.gc_build_info().decode(), 'abi': native.lib.gc_abi_version()},
            'torch_version': worker.version, 'reference_process': 'isolated spawn worker, torch CPU threads=1',
            'counts': counts, 'contract_checks': checks,
            'contract_counts': dict(Counter(c['status'] for c in checks)),
            'record_unit': 'one implementation/thread/shape/seed/reference; two references per invocation',
            'records': records, 'elapsed_seconds': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'opencl', 'cuda', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/correctness.json')
    parser.add_argument('--shape-set', choices=['course', 'boundary'], default='course')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    report = run(args.backend, args.limit, args.shape_set)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'counts': report['counts'],
                      'contract_counts': report['contract_counts']}))
    unavailable = args.backend != 'all' and not report['availability'][args.backend]['available']
    required_not_run = any(r['status'] == 'NOT_RUN' and
                           (r['backend'] != 'cuda' or args.backend == 'cuda')
                           for r in report['records'])
    return int(bool(report['counts']['FAIL'] or report['contract_counts'].get('FAIL') or
                    unavailable or required_not_run))


if __name__ == '__main__':
    raise SystemExit(main())
