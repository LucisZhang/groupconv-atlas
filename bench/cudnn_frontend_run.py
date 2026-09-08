"""Isolated direct cuDNN Frontend 1.18.0 benchmark; no framework fallback."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/cudnn-frontend-protocol.json'
METRICS = ('t_device_op_graph_us', 't_api_us', 't_host_feed_us')


class Unsupported(RuntimeError):
    pass


def strides(dims, layout):
    """Physical strides for logical N,C,H,W (or K,Cg,R,S), including C=1."""
    if len(dims) != 4 or any(d <= 0 for d in dims):
        raise ValueError('four positive logical dimensions required')
    _, c, h, w = dims
    if layout == 'nchw': return (c*h*w, h*w, w, 1)
    if layout == 'channels_last': return (h*w*c, 1, w*c, c)
    raise ValueError('unknown layout')


def validate_protocol(p, smoke=False):
    if p['frontend_version'] != '1.18.0': raise ValueError('unverified Frontend API version')
    if (p['torch_version'],p['cuda_version'],p['cudnn_version']) != ('2.8.0+cu128','12.8',91002):
        raise ValueError('audited torch/CUDA/cuDNN versions required')
    if p['source_identity_version'] != 2 or p['fp64_points'] != 64:
        raise ValueError('source identity v2 and 64-point oracle required')
    if p['dtype'] != 'float32' or p['compute_dtype'] != 'float32': raise ValueError('FP32 required')
    if set(p['excluded_numerical_notes']) != {'TENSOR_CORE', 'DOWN_CONVERT_INPUTS', 'REDUCED_PRECISION_REDUCTION'}:
        raise ValueError('strict FP32 numerical exclusions required')
    for key in ('candidate_cap', 'tuning_samples', 'warmup', 'batches', 'samples_per_batch', 'repeat_cap'):
        if type(p[key]) is not int or p[key] <= 0: raise ValueError(key + ' must be a positive integer')
    for key in ('tuning_seconds', 'shape_timeout_seconds', 'total_budget_seconds', 'repeat_target_us'):
        if not math.isfinite(p[key]) or p[key] <= 0: raise ValueError(key + ' must be finite and positive')
    if type(p['workspace_bytes']) is not int or p['workspace_bytes'] < 0: raise ValueError('invalid workspace cap')
    if not p['heuristic_modes'] or any(m not in ('A', 'B', 'FALLBACK') for m in p['heuristic_modes']):
        raise ValueError('invalid heuristics')
    if p['heuristic_modes'] != ['A','FALLBACK']: raise ValueError('fixed A/FALLBACK heuristic contract required')
    if not smoke and (p['batches'] < 5 or p['samples_per_batch'] < 30):
        raise ValueError('formal requires 5x30 or more; reduced runs require --smoke')


def dependencies(p):
    """Return actual vendor package only; missing/incompatible runtimes are explicit."""
    try:
        torch = importlib.import_module('torch')
        cudnn = importlib.import_module('cudnn')
    except (ImportError, OSError) as exc:
        raise Unsupported('required torch/CUDA or nvidia-cudnn-frontend unavailable: ' + str(exc)) from exc
    if getattr(cudnn, '__version__', None) != p['frontend_version']:
        raise Unsupported('requires audited nvidia-cudnn-frontend==' + p['frontend_version'])
    if str(torch.__version__) != p['torch_version'] or torch.version.cuda != p['cuda_version']:
        raise Unsupported('requires audited torch '+p['torch_version']+' / CUDA '+p['cuda_version'])
    if not torch.cuda.is_available(): raise Unsupported('CUDA unavailable; no CPU or torch timing fallback')
    for name in ('pygraph', 'create_handle', 'set_stream', 'get_stream', 'backend_version', 'numerical_note','heur_mode'):
        if not hasattr(cudnn, name): raise Unsupported('Frontend API missing: ' + name)
    if cudnn.backend_version() != p['cudnn_version'] or torch.backends.cudnn.version() != p['cudnn_version']:
        raise Unsupported('Frontend and torch both require cuDNN '+str(p['cudnn_version']))
    for name in p['excluded_numerical_notes']:
        if not hasattr(cudnn.numerical_note,name): raise Unsupported('strict FP32 exclusion unavailable: '+name)
    for name in p['heuristic_modes']:
        if not hasattr(cudnn.heur_mode,name): raise Unsupported('heuristic unavailable: '+name)
    return torch, cudnn


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def best_candidate(candidates):
    valid = [c for c in candidates if c['status'] == 'PASS' and
             math.isfinite(c.get('tuning_median_graph_us', float('nan'))) and c['tuning_median_graph_us'] > 0]
    return min(valid, key=lambda c: (c['tuning_median_graph_us'], c['index'])) if valid else None


def failure(exc):
    if isinstance(exc, Unsupported) or 'NotSupported' in type(exc).__name__: return 'UNSUPPORTED'
    if 'OutOfMemory' in type(exc).__name__ or 'out of memory' in str(exc).lower(): return 'OOM'
    return 'FAIL'


def build_failure(exc):
    # Exact v1.18.0 vendor rejection diagnostics; other failures stay failures.
    rejected = ('Deselecting execution plan', 'Skipping plan since workspace violation',
                'Workspace size is too large', 'Skipping plan since shared memory violation',
                'CUDNN_STATUS_NOT_SUPPORTED')
    return 'UNSUPPORTED' if any(message in str(exc) for message in rejected) else failure(exc)


def runtime_identity(torch, cudnn):
    from .cuda_run import identity_gpu, gpu_snapshot
    paths = {str(Path(cudnn.__file__).resolve())}
    paths.update(str(p.resolve()) for p in Path(cudnn.__file__).parent.rglob('*')
                 if p.is_file() and p.suffix in ('.py','.so'))
    maps = Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            path = line.split()[-1]
            if path.startswith('/') and ('libcudnn' in path or '_compiled_module' in path) and Path(path).is_file():
                paths.add(path)
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'torch_cuda': torch.version.cuda, 'torch_git': torch.version.git_version,
            'frontend': cudnn.__version__, 'backend_version': cudnn.backend_version(),
            'torch_cudnn': torch.backends.cudnn.version(), 'gpu': identity_gpu(torch),
            'gpu_snapshot': gpu_snapshot(), 'runtime_files_sha256': {p: digest(p) for p in sorted(paths)}}


def oracle_gate(actual, reference, prepared, p):
    """Full FP32 gate first, then independent points; no timing after either fails."""
    from .api import error_metrics
    from .fp64_points import check_points
    full = error_metrics(actual,reference,atol=p['atol'],rtol=p['rtol'])
    points = check_points(prepared,actual,p['atol'],p['rtol']) if full['passed'] else {'status':'NOT_RUN','passed':False}
    return {'error':full,'fp64_points':points,'passed':bool(full['passed'] and points['passed'])}


def worker(path, shape_record, layout, p):
    """One process/stream per shape-layout; parent enforces the hard wall limit."""
    if hasattr(os, 'setsid'): os.setsid()
    row = {'shape_id': shape_record['shape_id'], 'layout': layout, 'status': 'RUNNING',
           'implementation': 'direct_cudnn_frontend', 'candidates': [], 'samples': [],
           'host_device_roundtrip': 'NOT_RUN', 'graph_status': 'NOT_RUN'}
    save(path, row)
    handle = None
    begin = time.perf_counter()
    try:
        torch, cudnn = dependencies(p)
        import torch.nn.functional as F
        from .api import Shape, inputs
        from .fp64_points import prepare_reference, check_points
        from .cuda_run import precision, timing
        from .stats import describe
        s = Shape.from_record(shape_record)
        row['runtime'] = runtime_identity(torch, cudnn)
        row['precision'] = precision(torch)
        row['numerical_filter'] = {'excluded': p['excluded_numerical_notes'],
            'per_plan_numeric_readback': 'UNAVAILABLE in audited Python API; vendor filters bar matching plans before build'}
        stream = torch.cuda.Stream(device=0)
        handle = cudnn.create_handle()
        cudnn.set_stream(handle, stream.cuda_stream)
        row['stream'] = {'requested': stream.cuda_stream, 'actual': cudnn.get_stream(handle), 'non_default': True}
        if int(cudnn.get_stream(handle)) != stream.cuda_stream: raise RuntimeError('cuDNN stream readback mismatch')
        with torch.inference_mode(), torch.cuda.stream(stream):
            x, w = inputs(s, p['seed'])
            row['inputs'] = {'seed':p['seed'],'serialization':'contiguous NumPy FP32 C-order bytes',
                             'input_sha256':hashlib.sha256(x.tobytes(order='C')).hexdigest(),
                             'weight_sha256':hashlib.sha256(w.tobytes(order='C')).hexdigest()}
            prepared = prepare_reference(s,x,w,seed=p['seed'],count=p['fp64_points'])
            row['oracle_gate'] = p['correctness_gate']
            row['prepared_fp64_reference'] = prepared
            nx, nw = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
            reference = F.conv2d(nx, nw, stride=(s.sh,s.sw), padding=(s.ph,s.pw),
                                 dilation=(s.dh,s.dw), groups=s.groups)
            stream.synchronize()
            reference_cpu = reference.cpu().numpy()
            row['reference_fp64_points'] = check_points(prepared,reference_cpu,p['atol'],p['rtol'])
            if not row['reference_fp64_points']['passed']: raise RuntimeError('torch reference failed independent FP64 points')
            def allocate(dims):
                return torch.empty_strided(dims, strides(dims, layout), device='cuda', dtype=torch.float32)
            tx, tw, ty = allocate(s.input), allocate(s.weight), allocate(s.output)
            row['tensors'] = {name: {'dims': list(t.shape), 'strides': list(t.stride()), 'dtype': str(t.dtype)}
                              for name, t in [('x',tx),('w',tw),('y',ty)]}
            stream.synchronize(); start = time.perf_counter_ns()
            tx.copy_(nx); tw.copy_(nw); stream.synchronize()
            row['setup_layout_materialization_us'] = (time.perf_counter_ns()-start)/1000
            row['layout_roundtrip_equal'] = bool(torch.equal(tx, nx) and torch.equal(tw, nw))
            if not row['layout_roundtrip_equal']: raise RuntimeError('layout materialization changed values')
            start = time.perf_counter_ns()
            graph = cudnn.pygraph(name='groupconv_atlas', io_data_type=cudnn.data_type.FLOAT,
                                 intermediate_data_type=cudnn.data_type.FLOAT,
                                 compute_data_type=cudnn.data_type.FLOAT, handle=handle)
            X = graph.tensor(dim=list(tx.shape), stride=list(tx.stride()), data_type=cudnn.data_type.FLOAT, name='X', uid=1)
            W = graph.tensor(dim=list(tw.shape), stride=list(tw.stride()), data_type=cudnn.data_type.FLOAT, name='W', uid=2)
            Y = graph.conv_fprop(image=X, weight=W, padding=[s.ph,s.pw], stride=[s.sh,s.sw],
                                 dilation=[s.dh,s.dw], compute_data_type=cudnn.data_type.FLOAT, name='groupconv')
            Y.set_output(True).set_dim(list(ty.shape)).set_stride(list(ty.stride())).set_data_type(cudnn.data_type.FLOAT)
            graph.validate(); graph.build_operation_graph()
            graph.create_execution_plans([getattr(cudnn.heur_mode, m) for m in p['heuristic_modes']])
            graph.deselect_numeric_notes([getattr(cudnn.numerical_note, n) for n in p['excluded_numerical_notes']])
            graph.deselect_workspace_greater_than(p['workspace_bytes'])
            row['numerical_filter']['application_status'] = 'APPLIED'
            row['numerical_filter']['actual_enum_values'] = {n:str(getattr(cudnn.numerical_note,n)) for n in p['excluded_numerical_notes']}
            row['heuristic_enum_values'] = {n:str(getattr(cudnn.heur_mode,n)) for n in p['heuristic_modes']}
            row['setup_graph_heuristics_filter_us'] = (time.perf_counter_ns()-start)/1000
            row['graph_description'] = repr(graph)
            row['graph_description_sha256'] = hashlib.sha256(repr(graph).encode()).hexdigest()
            count = graph.get_execution_plan_count()
            row['candidate_count'] = count
            row['heuristic_modes'] = p['heuristic_modes']
            row['workspace_cap_bytes'] = p['workspace_bytes']
            pointers = {X.get_uid(): tx.data_ptr(), W.get_uid(): tw.data_ptr(), Y.get_uid(): ty.data_ptr()}
            def make_call(index, workspace):
                def call():
                    graph.execute_plan_at_index(pointers, workspace.data_ptr(), index, handle=handle)
                    return ty
                return call
            tuning_start = time.monotonic()
            row['candidates'] = [{'index': i, 'status': 'NOT_RUN', 'reason': 'not reached'} for i in range(count)]
            for index, c in enumerate(row['candidates']):
                if index >= p['candidate_cap'] or time.monotonic()-tuning_start >= p['tuning_seconds']:
                    c['reason'] = 'candidate cap' if index >= p['candidate_cap'] else 'soft tuning budget exhausted'
                    continue
                c['status'] = 'RUNNING'; c.pop('reason', None); save(path, row)
                candidate_start = time.perf_counter_ns()
                call = workspace = None
                try:
                    c['phase'] = 'support_build'
                    c['plan_name'] = graph.get_plan_name_at_index(index)
                    c['behavior_notes'] = [str(n) for n in graph.get_behavior_notes_for_plan_at_index(index)]
                    start = time.perf_counter_ns()
                    graph.build_plan_at_index(index)
                    c['support_and_build_us'] = (time.perf_counter_ns()-start)/1000
                    size = graph.get_workspace_size_plan_at_index(index)
                    c['workspace_required_bytes'] = size
                    if size > p['workspace_bytes']: raise RuntimeError('vendor workspace filter violated')
                    c['phase'] = 'validate_warmup_capture_tune'
                    workspace = torch.empty(max(1,size), device='cuda', dtype=torch.uint8)
                    call = make_call(index, workspace)
                    call(); stream.synchronize()
                    gate = oracle_gate(ty.cpu().numpy(), reference_cpu, prepared, p)
                    c.update(gate)
                    if not gate['passed']: raise RuntimeError('full-output FP32 or independent FP64 point comparison failed')
                    for _ in range(p['warmup']): call()
                    stream.synchronize()
                    timing(call, torch, stream, 1)  # Capture and eager/graph comparison excluded from tuning samples.
                    c['tuning_samples'] = [timing(call, torch, stream, 1) for _ in range(p['tuning_samples'])]
                    c['tuning_median_graph_us'] = describe([v['t_device_op_graph_us'] for v in c['tuning_samples']])['median']
                    c['status'] = 'PASS'
                except Exception as exc:
                    c.update(status=build_failure(exc) if c['phase'] == 'support_build' else failure(exc),
                             reason=str(exc), exception=type(exc).__name__)
                finally:
                    c['build_validate_warmup_capture_tune_us'] = (time.perf_counter_ns()-candidate_start)/1000
                    if call is not None: call._graph_cache = {}
                    call = workspace = None
                    save(path, row)
                # CUDA failures can poison capture/the context; never hide them by selecting another plan.
                if c['status'] in ('FAIL', 'OOM') or (c['status'] != 'PASS' and c['phase'] != 'support_build'):
                    row.update(status=c['status'], reason='candidate failed; inspect retained candidate receipt')
                    return
            row['tuning_elapsed_seconds'] = time.monotonic()-tuning_start
            chosen = best_candidate(row['candidates'])
            if chosen is None: raise Unsupported('no validated executable plan within budget and strict FP32 filters')
            row['selected_plan'] = {k: v for k,v in chosen.items() if k != 'tuning_samples'}
            row['selection_scope'] = p['selection']
            workspace = torch.empty(max(1, chosen['workspace_required_bytes']), device='cuda', dtype=torch.uint8)
            resident = make_call(chosen['index'], workspace)
            restored = torch.empty(s.output, device='cuda', dtype=torch.float32)
            def caller_nchw():
                if layout == 'channels_last':
                    tx.copy_(nx); tw.copy_(nw)
                resident()
                if layout == 'channels_last':
                    restored.copy_(ty)
                    return restored
                return ty
            row['allocation_policy'] = 'preallocated output/workspace and conversion buffers; unlike torch allocating Conv output'
            row['conversion_policy'] = 'caller_nchw: input + weights every invocation, output restored; identity for nchw'
            calls = {'resident': resident, 'caller_nchw': caller_nchw}
            repeats = {}
            row['path_correctness'] = {}
            for name, call in calls.items():
                actual = call(); stream.synchronize()
                metrics = oracle_gate(actual.cpu().numpy(), reference_cpu, prepared, p)
                row['path_correctness'][name] = metrics
                if not metrics['passed']: raise RuntimeError(name + ' full-output comparison failed')
                for _ in range(p['warmup']): call()
                pilot = timing(call, torch, stream, 1)
                repeats[name] = max(1, min(p['repeat_cap'], int(p['repeat_target_us']/max(pilot['t_api_us'],1))))
                timing(call, torch, stream, repeats[name])
            row['graph_status'] = 'PASS'
            row['graph_output_check'] = 'full eager-versus-captured comparison performed by cuda_run.timing for each path/repeat count'
            row['repeats'] = repeats
            row['setup_before_formal_seconds'] = time.perf_counter()-begin
            rng = random.Random(p['seed'])
            for batch in range(p['batches']):
                for sample in range(p['samples_per_batch']):
                    order = list(calls); rng.shuffle(order)
                    for name in order:
                        values = timing(calls[name], torch, stream, repeats[name])
                        for value in values.values():
                            if not math.isfinite(value) or value <= 0: raise RuntimeError('invalid latency')
                        row['samples'].append(dict(path=name, batch=batch, sample=sample, **values))
                save(path, row)
            row['summary'] = {}
            for name in calls:
                medians = {metric: [statistics.median(r[metric] for r in row['samples']
                    if r['path'] == name and r['batch'] == b) for b in range(p['batches'])] for metric in METRICS}
                row['summary'][name] = {'batch_medians': medians,
                    'descriptive': {m: describe(v) for m,v in medians.items()}}
            row['status'] = 'PASS'
            row['runtime_after'] = runtime_identity(torch,cudnn)
    except Exception as exc:
        row.update(status=failure(exc), reason=str(exc), exception=type(exc).__name__)
    finally:
        # Process teardown releases graph, handle and device allocations together.
        row['elapsed_seconds'] = time.perf_counter()-begin
        save(path, row)


def stop(process):
    if process is None or process.pid is None: return
    if hasattr(os, 'killpg'):
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    if process.is_alive(): process.kill()
    process.join(5)


def terminate(signum, frame):
    raise SystemExit(128+signum)


def report(out, key, rows):
    counts = {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}
    save(out/'bench.json', {'kind': 'direct_cudnn_frontend', 'run_key': key, 'counts': counts, 'records': rows})
    with (out/'raw.jsonl').open('w') as f:
        for row in rows:
            for sample in row.get('samples', []):
                f.write(json.dumps(dict(shape_id=row['shape_id'], layout=row['layout'], **sample), allow_nan=False)+'\n')
    with (out/'bench.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['shape_id','layout','status','path',*METRICS,'reason'])
        for row in rows:
            summaries = row.get('summary', {})
            for path in summaries or {'NOT_RUN': None}:
                writer.writerow([row['shape_id'],row['layout'],row['status'],path,
                    *[summaries[path]['descriptive'][m]['median'] if summaries else '' for m in METRICS], row.get('reason','')])
    save(out/'summary.json', {'counts':counts, 'measurement_kind':key['measurement_kind'],
        'interpretation':'descriptive batch medians, separate resident/caller_nchw and graph/eager; no independent-session inference'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shape-set', choices=['course','course-batch4','atlas'], default='course')
    parser.add_argument('--shape-ids', nargs='+')
    parser.add_argument('--layouts', nargs='+', choices=['nchw','channels_last'])
    parser.add_argument('--batches', type=int); parser.add_argument('--samples', type=int)
    parser.add_argument('--workspace-bytes', type=int); parser.add_argument('--candidate-cap', type=int)
    parser.add_argument('--tuning-seconds', type=float); parser.add_argument('--shape-timeout', type=float)
    parser.add_argument('--budget-seconds', type=float)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    p = json.loads(PROTOCOL.read_text())
    for field, value in [('batches',args.batches),('samples_per_batch',args.samples),
        ('workspace_bytes',args.workspace_bytes),('candidate_cap',args.candidate_cap),
        ('tuning_seconds',args.tuning_seconds),('shape_timeout_seconds',args.shape_timeout),
        ('total_budget_seconds',args.budget_seconds)]:
        if value is not None: p[field] = value
    try: validate_protocol(p,args.smoke)
    except ValueError as exc: parser.error(str(exc))
    shapes = json.loads((ROOT/f'configs/{args.shape_set}.json').read_text())['shapes']
    if args.shape_ids:
        unknown = set(args.shape_ids)-{s['shape_id'] for s in shapes}
        if unknown: parser.error('unknown shape IDs: '+','.join(sorted(unknown)))
        shapes = [s for s in shapes if s['shape_id'] in args.shape_ids]
    layouts = list(dict.fromkeys(args.layouts or p['layouts']))
    from .run import source_identity
    identity = source_identity()
    if identity.get('source_identity_version') != p['source_identity_version']:
        parser.error('source identity version mismatch')
    key = {'source':identity, 'protocol':p, 'protocol_sha256':digest(PROTOCOL),
           'shape_set':args.shape_set, 'shape_config_sha256':digest(ROOT/f'configs/{args.shape_set}.json'),
           'shape_ids':[s['shape_id'] for s in shapes], 'layouts':layouts,
           'measurement_kind':'SMOKE' if args.smoke else 'FORMAL'}
    unavailable = None
    try:
        torch,cudnn = dependencies(p)
        key['runtime'] = runtime_identity(torch,cudnn)
    except Unsupported as exc:
        unavailable = str(exc); key['runtime'] = {'status':'UNSUPPORTED','reason':unavailable}
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'run-key.json').exists():
        old = json.loads((out/'run-key.json').read_text())
        # Volatile temperature/clock snapshots are evidence, not resume identity.
        def stable(k):
            value = json.loads(json.dumps(k)); value.get('runtime',{}).pop('gpu_snapshot',None)
            return value
        if not args.resume or stable(old) != stable(key): parser.error('output exists or resume identity differs')
        key = old
    elif any(out.iterdir()): parser.error('output directory must be empty or a matching --resume run')
    save(out/'run-key.json',key)
    rows = [{'shape_id':s['shape_id'], 'layout':layout, 'implementation':'direct_cudnn_frontend',
             'status':'NOT_RUN', 'reason':'not reached'} for s in shapes for layout in layouts]
    row_indices = {(r['shape_id'],r['layout']):i for i,r in enumerate(rows)}
    active = None; active_path = None; active_index = None; started = time.monotonic()
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        for s in shapes:
            for layout in layouts:
                path = out/(s['shape_id']+'-'+layout+'.json')
                row = {'shape_id':s['shape_id'], 'layout':layout, 'implementation':'direct_cudnn_frontend','status':'NOT_RUN'}
                row_index = row_indices[(s['shape_id'],layout)]
                if args.resume and path.exists() and json.loads(path.read_text())['status'] == 'PASS':
                    rows[row_index] = json.loads(path.read_text()); continue
                remaining = p['total_budget_seconds']-(time.monotonic()-started)
                if unavailable: row.update(status='UNSUPPORTED',reason=unavailable)
                elif remaining <= 0: row['reason'] = 'total budget exhausted'
                else:
                    active = mp.get_context('spawn').Process(target=worker,args=(path,s,layout,p))
                    active_path = path; active_index = row_index
                    active.start()
                    active.join(min(p['shape_timeout_seconds'],remaining))
                    timed_out = active.is_alive()
                    if timed_out: stop(active)
                    else:
                        active.join(5)
                        if active.is_alive(): stop(active)
                    if path.exists(): row = json.loads(path.read_text())
                    if timed_out: row.update(status='TIMEOUT',reason='hard shape/layout or total wall timeout')
                    elif row['status'] == 'RUNNING' or active.exitcode != 0:
                        row.update(status='FAIL',reason='worker exited without completed receipt',exitcode=active.exitcode)
                    active = None
                    active_path = active_index = None
                save(path,row); rows[row_index] = row; report(out,key,rows)
    finally:
        stop(active)
        if active_path is not None:
            row = json.loads(active_path.read_text()) if active_path.exists() else rows[active_index]
            row.update(status='INTERRUPTED',reason='parent interrupted; process group reaped')
            save(active_path,row); rows[active_index] = row
        signal.signal(signal.SIGTERM,previous)
        report(out,key,rows)
    return 0 if rows and all(r['status']=='PASS' for r in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
