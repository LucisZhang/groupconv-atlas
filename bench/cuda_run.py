"""Bounded, resumable CUDA eager benchmark; keep frozen CPU/OpenCL protocol separate."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import time
import numpy as np
from .api import ROOT, Shape, Native, NativeError, inputs, error_metrics
from .check import boundary_shapes, expected_support, SEEDS
from .run import atomic_json, source_identity, sha
from .stats import describe
from .fp64_points import prepare_reference, check_points

VARIANTS = ('k0', 'k1', 'k2', 'k3', 'torch_cuda_default', 'torch_cuda_tuned')


def precision(torch):
    """Use exactly one precision-control generation; read back every chosen flag."""
    if hasattr(torch.backends, 'fp32_precision'):
        targets = {'global': torch.backends, 'matmul': torch.backends.cuda.matmul,
                   'cudnn': torch.backends.cudnn, 'conv': torch.backends.cudnn.conv,
                   'rnn': torch.backends.cudnn.rnn}
        for target in targets.values(): target.fp32_precision = 'ieee'
        settings = {k: v.fp32_precision for k, v in targets.items()}
        if any(v != 'ieee' for v in settings.values()): raise RuntimeError('FP32 precision readback failed')
        settings['api'] = 'fp32_precision'
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        settings = {'api': 'allow_tf32', 'matmul': torch.backends.cuda.matmul.allow_tf32,
                    'cudnn': torch.backends.cudnn.allow_tf32}
        if settings['matmul'] or settings['cudnn']: raise RuntimeError('TF32 disable failed')
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    return settings


def validate_correctness(report, identity, library_hash, smoke=False):
    if report.get('kind') != 'correctness': raise ValueError('unknown correctness report kind')
    if report.get('library', {}).get('sha256') != library_hash: raise ValueError('correctness library mismatch')
    if report.get('source', {}).get('source_tree_sha256') != identity['source_tree_sha256']:
        raise ValueError('correctness source mismatch or missing source identity')
    if report.get('availability', {}).get('cuda', {}).get('available') is not True:
        raise ValueError('correctness CUDA unavailable')
    if report.get('counts', {}).get('FAIL', 0) or report.get('contract_counts', {}).get('FAIL', 0):
        raise ValueError('correctness report has FAIL counts')
    records = report.get('records', [])
    checks = report.get('contract_checks', [])
    if not records or not checks or any(r.get('status') == 'FAIL' for r in records + checks):
        raise ValueError('missing or failed correctness evidence')
    cuda = [r for r in records if r.get('backend') == 'cuda']
    if not cuda or any(r.get('status') not in ('PASS', 'UNSUPPORTED') for r in cuda):
        raise ValueError('CUDA correctness incomplete')
    if any(r.get('status') != 'PASS' for r in checks): raise ValueError('contract checks incomplete')
    if not smoke:
        if report.get('limit') is not None: raise ValueError('limited correctness report is smoke only')
        for shape in boundary_shapes():
            for impl in range(4):
                for seed in SEEDS:
                    matched = [r for r in cuda if r.get('shape_id') == shape.id and
                               r.get('implementation') == impl and r.get('seed') == seed]
                    expected = 'PASS' if expected_support('cuda', shape, impl) else 'UNSUPPORTED'
                    if not matched or any(r['status'] != expected for r in matched):
                        raise ValueError('boundary implementation/seed coverage missing or inconsistent')
                    if expected == 'PASS' and not {'torch_cpu_fp32_full', 'python_fp64_full'} <= {r.get('reference') for r in matched}:
                        raise ValueError('independent full-output reference coverage missing')


def gpu_snapshot():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,power.limit,temperature.gpu,clocks.sm', '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc: return 'UNKNOWN: ' + str(exc)


def identity_gpu(torch):
    prop = torch.cuda.get_device_properties(0)
    return {'name': prop.name, 'uuid': str(getattr(prop, 'uuid', 'UNKNOWN')),
            'capability': list(torch.cuda.get_device_capability(0)), 'memory': prop.total_memory,
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET')}


def timing(call, torch, stream, repeats):
    # Capture event nodes and logical ops together so Python cannot starve the timed interval.
    cache = getattr(call, '_graph_cache', {})
    if repeats not in cache:
        pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                  torch.cuda.Event(enable_timing=True, external=True)) for _ in range(repeats)]
        graph = torch.cuda.CUDAGraph()
        outputs = []
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for start, end in pairs:
                start.record(stream); outputs.append(call()); end.record(stream)
        graph.replay(); stream.synchronize()
        captured = outputs[-1].clone(); stream.synchronize()
        eager = call(); stream.synchronize()
        for output in (captured,):
            if not torch.isfinite(output).all().item() or not torch.allclose(output, eager, atol=1e-4, rtol=1e-4):
                raise RuntimeError('captured output failed full eager comparison')
        cache[repeats] = (graph, pairs, outputs)
        call._graph_cache = cache
    graph, pairs, _ = cache[repeats]
    graph.replay(); stream.synchronize()
    device = sum(start.elapsed_time(end) * 1000 for start, end in pairs) / repeats
    # Separate pass avoids charging event-record instrumentation to API wall time.
    stream.synchronize(); begin = time.perf_counter_ns()
    for _ in range(repeats): call()
    feed = (time.perf_counter_ns() - begin) / 1000 / repeats
    stream.synchronize()
    api = (time.perf_counter_ns() - begin) / 1000 / repeats
    return {'t_device_op_graph_us': device, 't_api_us': api, 't_host_feed_us': feed}


def tuned_worker(connection, shape_record, protocol):
    """A fresh process prevents the default baseline's cuDNN cache hiding search."""
    try:
        import torch
        import torch.nn.functional as F
        shape = Shape.from_record(shape_record)
        precision(torch)
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, 'benchmark_limit'):
            torch.backends.cudnn.benchmark_limit = protocol['cudnn_benchmark_limit']
        stream = torch.cuda.Stream(device=0)
        x, w = inputs(shape, protocol['seed'])
        with torch.inference_mode(), torch.cuda.stream(stream):
            tx, tw = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
            def call():
                return F.conv2d(tx, tw, stride=(shape.sh,shape.sw), padding=(shape.ph,shape.pw),
                                dilation=(shape.dh,shape.dw), groups=shape.groups)
            stream.synchronize(); begin = time.perf_counter_ns()
            result = call(); stream.synchronize()
            first = (time.perf_counter_ns()-begin)/1000
            actual = result.cpu().numpy()
            for _ in range(protocol['warmup']): call()
            stream.synchronize()
            pilot = timing(call, torch, stream, 1)
            repeats = max(1,min(protocol['repeat_cap'],int(protocol['repeat_target_us']/max(pilot['t_api_us'],1))))
            timing(call, torch, stream, repeats)
            connection.send({'status':'PASS','actual':actual,'first':first,'repeats':repeats})
            while connection.recv() == 'measure':
                connection.send({'status':'PASS','samples':[timing(call,torch,stream,repeats)
                                      for _ in range(protocol['samples_per_batch'])]})
    except BaseException as exc:
        connection.send({'status':'OOM' if 'OutOfMemory' in type(exc).__name__ else 'FAIL','reason':str(exc)})
    finally: connection.close()


def shape_worker(connection, shape_record, protocol, library):
    """Spawn isolation gives the parent a hard per-shape timeout, including CUDA hangs."""
    if hasattr(os, 'setsid'): os.setsid()
    shape = Shape.from_record(shape_record)
    rows = {name: {'shape_id': shape.id, 'implementation': name, 'status': 'NOT_RUN'} for name in VARIANTS}
    samples = []
    tuned_process = None
    tuned_connection = None
    try:
        import torch
        import torch.nn.functional as F
        settings = precision(torch)
        if hasattr(torch.backends.cudnn, 'benchmark_limit'):
            torch.backends.cudnn.benchmark_limit = protocol['cudnn_benchmark_limit']
        native = Native(library)
        stream = torch.cuda.Stream(device=0)
        x, w = inputs(shape, protocol['seed'])
        point_reference = prepare_reference(shape, x, w, seed=protocol['seed'], count=protocol.get('fp64_points',64))
        with torch.inference_mode(), torch.cuda.stream(stream):
            tx, tw = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
            y = torch.empty(shape.output, device='cuda', dtype=torch.float32)
            def conv():
                return F.conv2d(tx, tw, stride=(shape.sh, shape.sw), padding=(shape.ph, shape.pw),
                                dilation=(shape.dh, shape.dw), groups=shape.groups)
            stream.synchronize()
            begin = time.perf_counter_ns(); reference_tensor = conv(); stream.synchronize()
            default_first = (time.perf_counter_ns() - begin) / 1000
            reference = reference_tensor.cpu().numpy()
            calls = {f'k{i}': (lambda i=i: native.cuda(shape, tx, tw, y, i, stream.cuda_stream)) for i in range(4)}
            calls.update(torch_cuda_default=conv, torch_cuda_tuned=conv)
            repeats = {}
            for name in VARIANTS:
                row = rows[name]
                try:
                    if name == 'torch_cuda_tuned':
                        context = mp.get_context('spawn')
                        tuned_connection, child = context.Pipe()
                        tuned_process = context.Process(target=tuned_worker, args=(child,shape_record,protocol))
                        tuned_process.start(); child.close()
                        response = tuned_connection.recv()
                        if response['status'] != 'PASS':
                            row.update(response); continue
                        metrics = error_metrics(response['actual'], reference, atol=protocol['atol'],rtol=protocol['rtol'])
                        point_metrics = check_points(point_reference,response['actual'],protocol['atol'],protocol['rtol'])
                        row.update(status='PASS' if metrics['passed'] and point_metrics['passed'] else 'FAIL',error=metrics,
                                   fp64_points=point_metrics,
                                   first_call_initialization_search_us=response['first'],benchmark=True,
                                   cache_isolation='separate fresh process')
                        repeats[name] = response['repeats']
                        continue
                    torch.backends.cudnn.benchmark = False
                    begin = time.perf_counter_ns(); result = calls[name](); stream.synchronize()
                    row['first_call_initialization_search_us'] = default_first if name == 'torch_cuda_default' else (time.perf_counter_ns()-begin)/1000
                    row['benchmark'] = bool(torch.backends.cudnn.benchmark) if name.startswith('torch') else None
                    actual = result.cpu().numpy()
                    metrics = error_metrics(actual, reference, atol=protocol['atol'], rtol=protocol['rtol'])
                    point_metrics = check_points(point_reference,actual,protocol['atol'],protocol['rtol'])
                    row.update(status='PASS' if metrics['passed'] and point_metrics['passed'] else 'FAIL', error=metrics,fp64_points=point_metrics)
                    if name.startswith('k') and not expected_support('cuda', shape, int(name[1])):
                        row.update(status='FAIL', reason='implementation unexpectedly accepted unsupported shape')
                    if row['status'] != 'PASS': continue
                    for _ in range(protocol['warmup']): calls[name]()
                    stream.synchronize()
                    pilot = timing(calls[name], torch, stream, 1)
                    repeats[name] = max(1, min(protocol['repeat_cap'], int(protocol['repeat_target_us']/max(pilot['t_api_us'], 1))))
                    timing(calls[name], torch, stream, repeats[name])
                except NativeError as exc:
                    supported = expected_support('cuda', shape, int(name[1])) if name.startswith('k') else True
                    row.update(status='UNSUPPORTED' if exc.status == 2 and not supported else 'FAIL', reason=str(exc))
                except torch.cuda.OutOfMemoryError as exc:
                    row.update(status='OOM', reason=str(exc)); torch.cuda.empty_cache()
                except Exception as exc: row.update(status='FAIL', reason=str(exc))
            for batch in range(protocol['batches']):
                order = list(VARIANTS); random.Random(f"{protocol['seed']}:{shape.id}:{batch}").shuffle(order)
                for order_index, name in enumerate(order):
                    if rows[name]['status'] != 'PASS': continue
                    try:
                        if name == 'torch_cuda_tuned':
                            tuned_connection.send('measure'); response = tuned_connection.recv()
                            if response['status'] != 'PASS':
                                rows[name].update(response); continue
                            for sample, value in enumerate(response['samples']):
                                samples.append({'shape_id':shape.id,'implementation':name,'batch':batch,'sample':sample,
                                                'implementation_order':order_index,'repeats':repeats[name],**value})
                            continue
                        torch.backends.cudnn.benchmark = False
                        for sample in range(protocol['samples_per_batch']):
                            samples.append({'shape_id': shape.id, 'implementation': name, 'batch': batch,
                                            'sample': sample, 'implementation_order': order_index, 'repeats': repeats[name],
                                            **timing(calls[name], torch, stream, repeats[name])})
                    except torch.cuda.OutOfMemoryError as exc:
                        rows[name].update(status='OOM', reason=str(exc)); torch.cuda.empty_cache()
                    except Exception as exc: rows[name].update(status='FAIL', reason=str(exc))
        result = {'shape': shape_record, 'rows': list(rows.values()), 'samples': samples, 'precision': settings,
                  'stream': 'non-default', 'strides': {'input': list(tx.stride()), 'weight': list(tw.stride())}}
    except BaseException as exc:
        for row in rows.values(): row.update(status='OOM' if 'OutOfMemory' in type(exc).__name__ else 'FAIL', reason=str(exc))
        result = {'shape': shape_record, 'rows': list(rows.values()), 'samples': samples}
    if tuned_process is not None:
        if tuned_process.is_alive():
            try: tuned_connection.send('stop')
            except (BrokenPipeError, EOFError): pass
        tuned_process.join(5)
        if tuned_process.is_alive(): tuned_process.terminate(); tuned_process.join(5)
        if tuned_process.is_alive(): tuned_process.kill(); tuned_process.join(5)
        tuned_connection.close()
    connection.send(result); connection.close()


def finalize(out, shapes, protocol, kind, source_hash, started):
    rows, samples = [], []
    for shape in shapes:
        path = out/'shapes'/f'{shape.id}.json'
        if not path.exists(): continue
        data = json.loads(path.read_text())
        samples.extend(data['samples'])
        for row in data['rows']:
            row.update(layout='NCHW', dtype='float32', measurement_kind=kind, shape=data['shape'],
                       host_to_host_status='NOT_RUN', module_status='NOT_RUN', graph_status='PASS' if row['status']=='PASS' else 'NOT_RUN')
            values = [s for s in data['samples'] if s['implementation'] == row['implementation']]
            row['sample_count'] = len(values)
            for field in ('t_device_op_graph_us', 't_api_us', 't_host_feed_us'):
                v = [s[field] for s in values]
                if v:
                    row[field] = describe(v)
                    meds = [float(np.median([s[field] for s in values if s['batch'] == b]))
                            for b in sorted({s['batch'] for s in values})]
                    row[field]['batch_medians'] = meds
                    row[field]['batch_cv'] = float(np.std(meds, ddof=1)/np.mean(meds)) if len(meds)>1 else None
            rows.append(row)
    atomic_json(out/'bench.json', rows)
    with (out/'samples.jsonl').open('w') as f:
        for sample in samples: f.write(json.dumps(sample, allow_nan=False)+'\n')
    fields = sorted({k for row in rows for k in row})
    with (out/'bench.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k,v in row.items()} for row in rows)
    counts = {s: sum(r['status']==s for r in rows) for s in ('PASS','FAIL','UNSUPPORTED','OOM','NOT_RUN','TIMEOUT')}
    atomic_json(out/'summary.json', {'measurement_kind': kind, 'shape_count': len(shapes), 'completed_shape_count': len(rows)//len(VARIANTS),
                'row_count':len(rows), 'expected_row_count':len(shapes)*len(VARIANTS), 'sample_count':len(samples),
                'status_counts':counts, 'source_tree_sha256':source_hash, 'protocol':protocol,
                'elapsed_seconds':time.monotonic()-started, 'interpretation':'descriptive only; independent confirmation and dispatch audit remain separate'})
    return counts


def kill_shape_group(process):
    """Reap a shape worker and its tuned child, including interrupted startup."""
    if process is None or process.pid is None: return
    if hasattr(os, 'killpg'):
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    if process.is_alive(): process.kill()
    process.join(5)


def terminate_run(signum, frame):
    raise SystemExit(128 + signum)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shape-set', choices=['course','course-batch4','atlas'], default='course')
    p.add_argument('--shape-file', type=Path, help='additional frozen no-bias Shape records')
    p.add_argument('--shape-ids', nargs='+'); p.add_argument('--batches', type=int); p.add_argument('--samples', type=int)
    p.add_argument('--resume', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--correctness', type=Path, required=True)
    p.add_argument('--shape-timeout', type=float); p.add_argument('--budget-seconds', type=float)
    args = p.parse_args(argv)
    protocol = json.loads((ROOT/'configs/cuda-protocol.json').read_text())
    for key, value in [('batches',args.batches), ('samples_per_batch',args.samples),
                       ('shape_timeout_seconds',args.shape_timeout), ('total_budget_seconds',args.budget_seconds)]:
        if value is not None:
            if value <= 0: p.error('counts, timeouts and budgets must be positive')
            protocol[key] = value
    if not args.smoke and (protocol['batches']<5 or protocol['samples_per_batch']<30):
        p.error('formal sampling requires at least 5 batches x 30 samples; use --smoke for reduced runs')
    config = json.loads((args.shape_file or ROOT/f'configs/{args.shape_set}.json').read_text())
    if any(r.get('bias',False) or r.get('dtype','float32')!='float32' or r.get('layout','NCHW')!='NCHW' for r in config['shapes']):
        raise ValueError('shape file must preserve the no-bias FP32 NCHW contract')
    shapes = [Shape.from_record(r) for r in config['shapes']]
    if args.shape_ids:
        unknown = set(args.shape_ids)-{s.id for s in shapes}
        if unknown: p.error('unknown shape IDs: '+','.join(sorted(unknown)))
        shapes = [s for s in shapes if s.id in args.shape_ids]
    identity = source_identity(); native = Native()
    validation = json.loads(args.correctness.read_text())
    validate_correctness(validation, identity, sha(native.path), args.smoke)
    import torch
    if not torch.cuda.is_available() or not native.lib.gc_cuda_available(): raise RuntimeError('real torch and native CUDA required')
    settings = precision(torch)
    key = {'source':identity, 'library_sha256':sha(native.path), 'protocol':protocol, 'shapes':[s.id for s in shapes],
           'correctness_sha256':sha(args.correctness), 'gpu':identity_gpu(torch), 'torch':str(torch.__version__),
           'torch_git':torch.version.git_version, 'cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
           'precision':settings, 'measurement_kind':'SMOKE' if args.smoke else 'FORMAL'}
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    if (out/'run-key.json').exists():
        if not args.resume or json.loads((out/'run-key.json').read_text()) != key: raise RuntimeError('existing run; resume requires identical source/library/validation/protocol/GPU')
    else: atomic_json(out/'run-key.json', key)
    atomic_json(out/'provenance.json', {**key, 'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'os':platform.platform(), 'python':platform.python_version(), 'torch_build':torch.__config__.show(),
                'native_build':native.lib.gc_build_info().decode(), 'nvidia_smi':gpu_snapshot(),
                'exclusive_device':'NOT_VERIFIED', 'frequency_control':'NOT_CONTROLLED', 'backend_dispatch':'UNKNOWN'})
    started = time.monotonic(); context = mp.get_context('spawn')
    active_process = None
    active_connections = ()
    previous_sigterm = signal.signal(signal.SIGTERM, terminate_run)
    try:
        for index, shape in enumerate(shapes):
            path = out/'shapes'/f'{shape.id}.json'
            if args.resume and path.exists(): continue
            remaining = protocol['total_budget_seconds']-(time.monotonic()-started)
            result = None
            if remaining > 0:
                parent, child = context.Pipe(duplex=False)
                process = context.Process(target=shape_worker, args=(child, shape.record(), protocol, native.path))
                active_process = process
                active_connections = (parent, child)
                process.start(); child.close()
                if parent.poll(min(remaining, protocol['shape_timeout_seconds'])):
                    try: result = parent.recv()
                    except EOFError: pass
                if result is None: kill_shape_group(process)
                process.join(5)
                if process.is_alive(): kill_shape_group(process)
                parent.close()
                active_process = None
                active_connections = ()
            if result is None:
                status = 'TIMEOUT' if remaining > 0 else 'NOT_RUN'
                result = {'shape':shape.record(), 'samples':[], 'rows':[{'shape_id':shape.id,'implementation':name,
                           'status':status,'reason':'shape worker timed out/crashed' if remaining>0 else 'total run budget exhausted'} for name in VARIANTS]}
            atomic_json(path, result)
            counts = finalize(out, shapes, protocol, key['measurement_kind'], identity['source_tree_sha256'], started)
            print(json.dumps({'shape':shape.id,'progress':f'{index+1}/{len(shapes)}','counts':counts}), flush=True)
    finally:
        kill_shape_group(active_process)
        for connection in active_connections: connection.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
    counts = finalize(out, shapes, protocol, key['measurement_kind'], identity['source_tree_sha256'], started)
    return int(any(counts[s] for s in ('FAIL','OOM','NOT_RUN','TIMEOUT')))


if __name__ == '__main__': raise SystemExit(main())
