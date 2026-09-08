"""Offline, source-bound supplemental audits. No GPU or snapshot code is executed."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import platform
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import statistics as st

NATIVE = ('k0', 'k1', 'k2', 'k3')
MODEL = (*NATIVE, 'torch_cuda_default', 'torch_cuda_tuned')
BOUNDARY = (*NATIVE, 'torch_tuned_nchw', 'torch_tuned_channels_last')
MODULE = ('torch_block_tuned', 'block_one_k3_substitution')
FIELDS = ('t_device_op_graph_us', 't_api_us')
STATES = {'PASS', 'FAIL', 'UNSUPPORTED', 'OOM', 'TIMEOUT', 'NOT_RUN', 'RUNNING'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def compare(base, candidate, *, paired=False, draws=10000, seed=20260908):
    require(len(base) == len(candidate) and base, 'equal nonempty batch arrays required')
    require(all(positive(v) for v in [*base, *candidate]), 'invalid batch medians')
    result = {'ratio_of_median_batch_medians': st.median(base)/st.median(candidate),
              'batches': len(base), 'classification': 'UNCERTAIN', 'ci95': None,
              'inference_scope': 'descriptive; no independent-session claim', 'paired': paired}
    if not paired or len(base) < 10:
        return result
    require(type(draws) is int and draws >= 1000, 'at least 1000 resamples required')
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        ix = rng.choices(range(len(base)), k=len(base))
        values.append(st.median([base[i] for i in ix])/st.median([candidate[i] for i in ix]))
    values.sort()
    def quantile(q):
        pos = (len(values)-1)*q
        lo = int(pos)
        return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(pos-lo)
    lo, hi = quantile(.025), quantile(.975)
    result.update(ci95=[lo,hi], inference_scope='paired block bootstrap within this run only',
                  classification='WIN' if lo > 1.05 else 'LOSS' if hi < 1/1.05 else 'TIE' if lo >= 1/1.05 and hi <= 1.05 else 'UNCERTAIN')
    return result


def medians(samples, batches, count, fields):
    require(len(samples) == batches*count, 'incomplete PASS sample denominator')
    indexed = {}
    for row in samples:
        key = row.get('batch'), row.get('sample')
        require(all(type(v) is int for v in key), 'batch/sample must be integers')
        require(key not in indexed, 'duplicate sample')
        require(0 <= key[0] < batches and 0 <= key[1] < count, 'foreign sample index')
        require(all(positive(row.get(f)) for f in fields), 'invalid timing sample')
        indexed[key] = row
    return {f: [st.median(indexed[b,s][f] for s in range(count)) for b in range(batches)] for f in fields}


def strict_precision(value):
    require(value == {'api':'allow_tf32','matmul':False,'cudnn':False} or
            value == dict(api='fp32_precision', **{k:'ieee' for k in ('global','matmul','cudnn','conv','rnn')}), 'missing strict FP32 settings')
    require(not any(type(value.get(k)) is int for k in ('matmul','cudnn')), 'boolean precision readback required')


def fp32(value, elements=None):
    require(value.get('passed') is True and value.get('failed_elements') == 0 and
            type(value.get('elements')) is int and value['elements'] > 0 and
            value.get('atol') == value.get('rtol') == 1e-4, 'full FP32 oracle missing')
    require(all(type(value.get(k)) in (int,float) and math.isfinite(value[k]) and value[k] >= 0 for k in ('max_abs','max_rel','rms')), 'FP32 nonfinite or missing error values')
    require(value.get('epsilon') == 1e-12 and value['rms'] <= value['max_abs'] + 1e-12, 'FP32 error summary inconsistent')
    if elements is not None:
        require(value['elements'] == elements, 'FP32 output denominator mismatch')


def shape_context(record):
    from bench.api import Shape, FIELDS
    require(all(type(record.get(k)) is int for k in FIELDS), 'oracle geometry must use integers')
    shape = Shape.from_record(record)
    require(record.get('shape_id') == shape.id and record.get('dtype') == 'float32' and
            record.get('layout') == 'NCHW' and record.get('bias') is False, 'noncanonical oracle shape')
    for name, expected in (('input_dims',shape.input),('weight_dims',shape.weight),('output_dims',shape.output)):
        values = record.get(name, [])
        require(isinstance(values,(list,tuple)) and all(type(v) is int for v in values) and
                tuple(values) == expected, 'noncanonical oracle dimensions')
    return shape.id, tuple(shape.output)


def bind_shape_validator(source):
    import bench.api as shape_api
    require(source.get('source_files',{}).get('bench/api.py') == sha(shape_api.__file__),
            'shape validator source binding missing')


def fp64(value, shape):
    shape_id, dims = shape_context(shape)
    require(value.get('shape_id') == shape_id and value.get('output_count') == math.prod(dims), 'FP64 shape binding missing')
    points = value.get('points', [])
    require(value.get('passed') is True and value.get('point_count') == 64 and
            value.get('failed_points') == 0 and len(points) == 64 and
            value.get('reference') == 'independent_python_fp64_points' and
            value.get('atol') == value.get('rtol') == 1e-4, 'FP64 oracle missing')
    require(len({tuple(p.get('position', [])) for p in points}) == 64, 'duplicate FP64 points')
    for p in points:
        position = p.get('position',[])
        require(len(position) == 4 and all(type(v) is int and 0 <= v < d for v,d in zip(position,dims)), 'FP64 position outside shape')
        require(p.get('passed') is True and p.get('finite') is True and
                all(type(p.get(k)) in (float,int) and math.isfinite(p[k]) for k in ('actual','expected_fp64')) and
                abs(p['actual']-p['expected_fp64']) <= 1e-4+1e-4*abs(p['expected_fp64']), 'FP64 point mismatch')
        absolute = abs(p['actual']-p['expected_fp64'])
        for field, expected in [('absolute_error',absolute),('relative_error',absolute/max(abs(p['expected_fp64']),1e-12)),('threshold',1e-4+1e-4*abs(p['expected_fp64']))]:
            require(type(p.get(field)) in (int,float) and math.isfinite(p[field]) and math.isclose(p[field],expected,rel_tol=1e-12,abs_tol=1e-15), 'FP64 error detail inconsistent')


def gate(value, shape):
    require(value.get('passed', True) is True, 'oracle gate failed')
    _,dims = shape_context(shape)
    fp32(value.get('error', value.get('fp32_full', {})), math.prod(dims))
    fp64(value.get('fp64_points', {}), shape)


def bind_source(source, plan, snapshot):
    files = source.get('source_files', {})
    require(source.get('source_identity_version') == 2 and files, 'source identity v2 required')
    tree = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    require(tree == source.get('source_tree_sha256') == plan['source_tree_sha256'], 'source tree binding mismatch')
    for relative, digest in files.items():
        path = snapshot/relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(snapshot.resolve()), 'unsafe source path')
        require(path.is_file() and sha(path) == digest, 'snapshot file differs: '+relative)


def contract_receipt(checks):
    expected = dict(dtype=1,rank=1,strides=2,misaligned=1,device=1,buffer_bytes=1,alias=1,groups=1,geometry=1,implementation=1)
    cases = {r.get('case'):r for r in checks}
    mandatory = set(expected) | {'reject_nonfinite_nan','reject_nonfinite_inf','reject_nonfinite_-inf','python_shape_overflow','python_impl_overflow','python_nonintegral'}
    require(len(cases) == len(checks) and mandatory <= set(cases), 'complete ABI contract denominator missing')
    for name, value in expected.items():
        require(cases[name].get('expected_status') == cases[name].get('actual_status') == value and cases[name].get('status') == 'PASS', 'ABI status evidence mismatch')
    require(all(r.get('status') == 'PASS' for r in checks), 'contract check failed')


def native_receipt(key, correctness):
    require(correctness is not None, 'native correctness receipt required')
    report = read(correctness)
    from bench.cuda_run import validate_correctness
    root = Path(__file__).resolve().parents[1]
    for name, module in tuple(sys.modules.items()):
        if name.startswith('bench.') and getattr(module,'__file__',None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(root), 'validator dependency outside repository')
            relative = str(path.relative_to(root))
            require(key.get('source',{}).get('source_files',{}).get(relative) == sha(path), 'validator source binding missing: '+relative)
    validate_correctness(report,key['source'],key.get('library_sha256'),smoke=False)
    if 'correctness_sha256' in key:
        require(key['correctness_sha256'] == sha(correctness), 'correctness file SHA mismatch')
    require(report.get('kind') == 'correctness' and report.get('source',{}).get('source_tree_sha256') == key['source']['source_tree_sha256'], 'correctness source missing')
    require(report.get('library',{}).get('sha256') == key.get('library_sha256') and key.get('library_sha256'), 'library binding missing')
    require(report.get('availability',{}).get('cuda',{}).get('available') is True, 'CUDA not available')
    require(report.get('limit') is None and report.get('counts',{}).get('FAIL') == 0, 'correctness failed or limited')
    rows = report.get('records', []); contracts = report.get('contract_checks', [])
    contract_receipt(contracts)
    counts = Counter(r.get('status') for r in rows)
    require(all(report.get('counts',{}).get(k,0) == counts.get(k,0) for k in set(counts)|set(report.get('counts',{}))), 'correctness counts do not match records')
    identities = [tuple(r.get(k) for k in ('suite','backend','shape_id','implementation','threads','seed','input_pattern','reference')) for r in rows]
    require(len(identities) == len(set(identities)), 'duplicate correctness record')
    for row in rows:
        if row.get('status') == 'PASS': fp32(row.get('metrics',{}))
    require(rows and contracts and any(r.get('backend') == 'cuda' and r.get('status') == 'PASS' for r in rows), 'missing executed CUDA correctness')
    require(all(r.get('status') in ('PASS','UNSUPPORTED') for r in rows) and all(r.get('status') == 'PASS' for r in contracts), 'incomplete correctness')


def unit(name, row, samples, plan, fields=FIELDS):
    status = row.get('status', 'NOT_RUN') if row else 'NOT_RUN'
    require(status in STATES, 'unknown execution status')
    result = {'unit':name, 'execution_status':status, 'assessment':'NOT_ASSESSED',
              'reason':row.get('reason', 'missing unit') if row else 'missing unit'}
    if status == 'PASS':
        values = medians(samples, plan['batches'], plan['samples_per_batch'], fields)
        result.update(assessment='DESCRIPTIVE', batch_medians=values,
                      median_batch_medians={k:st.median(v) for k,v in values.items()})
        result.pop('reason', None)
    return result


def indexed_rows(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, 'duplicate row identity')
        result[identity] = row
    return result


def analyze(kind, run, plan, snapshot, correctness=None, draws=10000):
    run, snapshot = Path(run), Path(snapshot)
    require(plan.get('kind') == kind and plan.get('schema_version') == 1, 'explicit analysis plan required')
    require(type(plan.get('batches')) is int and plan['batches'] >= 5 and plan.get('samples_per_batch',0) >= 30, 'formal protocol counts required')
    key = read(run/('summary.json' if kind == 'module' else 'provenance.json' if kind == 'boundaries' else 'run-key.json'))
    bind_source(key.get('source',{}), plan, snapshot)
    require(key.get('measurement_kind') in ('FORMAL','FORMAL_MODEL_CONV_SHAPES','DESCRIPTIVE'), 'SMOKE/unexecuted run cannot be assessed')
    protocol = key if kind == 'module' else key['protocol']
    require(all(protocol.get(k) == plan[k] for k in ('batches','samples_per_batch')), 'plan protocol counts differ')
    if kind != 'frontend': native_receipt(key, correctness)
    units = []; separate = {}; pairs = []; context = []
    def append(name, row, samples, fields=FIELDS):
        result = unit(name,row,samples,plan,fields); units.append(result); return result
    if kind == 'module':
        require(key.get('model') == 'torchvision.models.mobilenet_v2' and key.get('block_path') == 'features.3' and
                key.get('fixed_implementation') == 'k3' and key.get('batch_size') == plan['batch_size'], 'fixed module identity mismatch')
        rows = key.get('measurements',{}); raw = [json.loads(s) for s in (run/'samples.jsonl').read_text().splitlines() if s.strip()]
        require(set(rows) <= set(MODULE) and all(s.get('implementation') in MODULE for s in raw), 'foreign module implementation')
        if any(r.get('status') == 'PASS' for r in rows.values()):
            strict_precision(key.get('precision',{}))
            require(key.get('gpu') and key.get('native_stream',{}).get('non_default') is True and key.get('input_sha256'), 'module execution evidence missing')
            require(key.get('state_file_sha256') == sha(run/'random-state.pt') and key.get('script_sha256') == sha(snapshot/'bench/module_run.py'), 'module state/script binding mismatch')
            selected_shape = key.get('selected_shape',{})
            _, leaf_dims = shape_context(selected_shape)
            require(key.get('observed',{}).get('conv',{}).get('output',{}).get('shape') == list(leaf_dims), 'observed leaf shape mismatch')
            fp32(key.get('checks',{}).get('leaf_conv',{}),math.prod(leaf_dims))
            fp32(key.get('checks',{}).get('complete_block',{}),math.prod(key['observed']['block']['output']['shape']))
            fp32(key.get('checks',{}).get('complete_model',{}),plan['batch_size']*1000)
            fp64(key.get('checks',{}).get('leaf_fp64_points',{}),selected_shape)
        for name in MODULE:
            row = rows.get(name,{})
            if row.get('status') == 'PASS': fp32(row.get('graph_vs_eager',{}),math.prod(key['observed']['block']['output']['shape']))
            samples = [s for s in raw if s['implementation'] == name]
            for sample in samples:
                order = list(MODULE); random.Random(f"{key['random_seed']}:{sample['batch']}").shuffle(order)
                require(sample.get('order') == order.index(name), 'module paired block order mismatch')
            append(name,row,samples,(*FIELDS,'t_host_feed_us'))
        pairs.append((MODULE[0],MODULE[1],True))
        separate = {k:key.get(k) for k in ('model_first_call_with_hooks_us','model_first_call_with_hooks_boundary','profile','amdahl_instrumented')}
        context.append('paired randomized implementation blocks within this run; no independent-session inference')
    elif kind == 'frontend':
        # The frozen runner filters atlas configuration order, not CLI selection order.
        shape_ids=key.get('shape_ids')
        require(isinstance(shape_ids,list) and all(isinstance(s,str) for s in shape_ids) and
                len(shape_ids)==len(set(shape_ids))==len(plan['shape_ids']) and
                set(shape_ids)==set(plan['shape_ids']) and key.get('layouts') == plan['layouts'], 'Frontend planned denominator mismatch')
        frontend_shapes = {r['shape_id']:r for r in read(snapshot/f"configs/{key['shape_set']}.json")['shapes']}
        require(set(plan['shape_ids']) <= set(frontend_shapes), 'Frontend planned shape absent from configuration')
        require(key.get('shape_config_sha256') == sha(snapshot/f"configs/{key['shape_set']}.json"), 'Frontend shape configuration binding missing')
        bench = read(run/'bench.json'); require(bench.get('run_key') == key, 'Frontend run-key mismatch')
        rows = indexed_rows(bench['records'],lambda r:(r['shape_id'],r['layout']))
        require(set(rows) <= {(s,l) for s in plan['shape_ids'] for l in plan['layouts']}, 'foreign Frontend job')
        for shape in plan['shape_ids']:
            for layout in plan['layouts']:
                row = rows.get((shape,layout),{}); samples = row.get('samples',[])
                require(all(s.get('path') in ('resident','caller_nchw') for s in samples), 'foreign Frontend path')
                if row.get('status') == 'PASS':
                    bind_shape_validator(key['source'])
                    strict_precision(row.get('precision',{}))
                    require(row.get('runtime',{}).get('gpu') and row.get('runtime',{}).get('runtime_files_sha256') and row.get('runtime_after'), 'Frontend runtime/library evidence missing')
                    for field in ('torch','torch_cuda','frontend','backend_version','gpu'):
                        require(row['runtime'].get(field) == row['runtime_after'].get(field) == key.get('runtime',{}).get(field), 'Frontend environment drift: '+field)
                    runtime_files = {}
                    for runtime in (key.get('runtime',{}), row['runtime'], row['runtime_after']):
                        files = runtime.get('runtime_files_sha256',{})
                        require(files, 'runtime file evidence missing')
                        for path, digest in files.items():
                            require(runtime_files.setdefault(path,digest) == digest, 'runtime library bytes changed')
                    require(row.get('stream',{}).get('non_default') is True and row['stream']['actual'] == row['stream']['requested'], 'Frontend stream mismatch')
                    require(row.get('graph_status') == 'PASS' and row.get('layout_roundtrip_equal') is True and row.get('inputs',{}).get('input_sha256') and row['inputs'].get('weight_sha256'), 'Frontend graph/layout/input evidence missing')
                    require(row.get('numerical_filter',{}).get('application_status') == 'APPLIED' and row['numerical_filter']['excluded'] == protocol['excluded_numerical_notes'], 'Frontend numeric filter missing')
                    fp64(row.get('reference_fp64_points',{}),frontend_shapes[shape])
                    require(positive(row.get('setup_before_formal_seconds')) and positive(row.get('setup_layout_materialization_us')), 'Frontend setup separation missing')
                    candidates = row.get('candidates',[])
                    require(candidates and [c['index'] for c in candidates] == list(range(row.get('candidate_count', -1))), 'candidate denominator missing')
                    for c in candidates:
                        require(c.get('status') in STATES, 'candidate status missing')
                        if c['status'] == 'PASS':
                            gate(c,frontend_shapes[shape]); require(c.get('workspace_required_bytes', math.inf) <= protocol['workspace_bytes'] and positive(c.get('tuning_median_graph_us')), 'candidate budget/tuning evidence missing')
                    selected = row.get('selected_plan',{})
                    require(any(c.get('index') == selected.get('index') and c.get('status') == 'PASS' for c in candidates), 'selected plan not validated')
                    require(selected.get('index') == min((c for c in candidates if c['status']=='PASS'), key=lambda c:c['tuning_median_graph_us'])['index'], 'selected plan not tuning winner')
                for path in ('resident','caller_nchw'):
                    if row.get('status') == 'PASS': gate(row.get('path_correctness',{}).get(path,{}),frontend_shapes[shape])
                    append(f'{shape}/{layout}/{path}',row,[s for s in samples if s['path']==path],(*FIELDS,'t_host_feed_us'))
                separate[f'{shape}/{layout}'] = {k:row.get(k) for k in ('setup_layout_materialization_us','setup_before_formal_seconds','tuning_elapsed_seconds','selected_plan','candidates','conversion_policy','allocation_policy')}
                pairs.append((f'{shape}/{layout}/resident',f'{shape}/{layout}/caller_nchw',False))
        if set(plan['layouts']) == {'nchw','channels_last'}:
            for shape in plan['shape_ids']:
                for path in ('resident','caller_nchw'):
                    pairs.append((f'{shape}/nchw/{path}',f'{shape}/channels_last/{path}',False))
        context.append('different layouts run in separate processes; no paired layout CI; within-path comparisons descriptive here')
    else:
        shapes = plan['shape_ids']; variants = BOUNDARY if kind == 'boundaries' else MODEL
        if kind == 'models':
            exported = read(run/'model-shapes.json'); require(sha(run/'model-shapes.json') == key.get('shape_report_sha256') == plan['shape_report_sha256'], 'model export binding mismatch')
            require([r['shape_id'] for r in exported['shapes']] == shapes and len(shapes) == 12 and sum(r['bias'] is True for r in exported['shapes']) == 4, '12 actual-layer denominator required')
            require(key.get('shape_protocol_sha256') == sha(snapshot/'configs/model-shapes-protocol.json') == plan['shape_protocol_sha256'], 'model frozen protocol mismatch')
            require(key.get('source_operations') == [r['source_operation_id'] for r in exported['shapes']], 'model source-operation identity mismatch')
        shape_records = {r['shape_id']:r for r in (read(snapshot/'configs/atlas.json')['shapes'] if kind == 'boundaries' else exported['shapes'])}
        require(set(shapes) <= set(shape_records), 'planned shape absent from source configuration')
        found = {p.stem for p in (run/'shapes').glob('*.json')}
        require(found <= set(shapes), 'foreign shape shard')
        for shape in shapes:
            data = read(run/'shapes'/f'{shape}.json') if shape in found else {'rows':[],'samples':[]}
            if shape in found: require(data.get('shape') == shape_records[shape], 'shape geometry/source identity mismatch')
            rows = indexed_rows(data.get('rows',[]), lambda r:r['implementation']); raw = data.get('samples',[])
            require(set(rows) <= set(variants) and all(s.get('implementation') in variants for s in raw), 'foreign implementation')
            if any(r.get('status')=='PASS' for r in rows.values()):
                strict_precision(data.get('precision',{})); require(data.get('gpu') if kind == 'boundaries' else key.get('gpu') and data.get('stream') == 'non-default', 'GPU execution identity missing')
            if kind == 'boundaries':
                require(all(s.get('residency') in ('target_layout_ready','caller_nchw','host_to_host') and s.get('cache_policy') in ('steady','eviction_attempt') for s in raw), 'foreign boundary condition')
                if any(r.get('status') == 'PASS' for r in rows.values()):
                    conversion = data.get('conversions',{})
                    require(set(conversion) == {'input_nchw_to_channels_last','weight_nchw_to_channels_last','output_channels_last_to_nchw'}, 'conversion evidence missing')
                    require(all(len(v)==protocol['conversion_samples'] and all(all(positive(sample.get(f)) for f in FIELDS) for sample in v) for v in conversion.values()), 'conversion sample denominator invalid')
                    require(data.get('eviction',{}).get('bytes') == protocol['eviction_bytes'] and data['eviction'].get('all_caches_cleared') == 'NOT_VERIFIED', 'eviction boundary evidence missing')
                    require(all(positive(data.get('layout_initialization_search_us',{}).get(k)) for k in ('NCHW','channels_last')), 'layout setup evidence missing')
            for name in variants:
                row = rows.get(name,{})
                original = shape_records[shape]
                biased = original.get('bias') is True
                if name in NATIVE and not biased:
                    common = (original['cin'] == original['cout'] and original['r'] == original['s'] == 3 and original['sh'] == original['sw'] == original['dh'] == original['dw'] == 1 and original['ph'] == original['pw'] == 1)
                    supported = name == 'k0' or common and (name != 'k2' or original['cin']//original['groups'] in (1,2,4))
                    require(not (row.get('status') == 'PASS' and not supported), 'unsupported kernel reported PASS')
                    require(not (row.get('status') == 'UNSUPPORTED' and supported), 'supported kernel reported UNSUPPORTED')
                if biased:
                    require(row.get('status', 'NOT_RUN') != 'PASS', 'biased source operation was measured')
                if kind == 'boundaries':
                    for residency in ('target_layout_ready','caller_nchw','host_to_host'):
                        if row.get('status') == 'PASS': gate(row.get('correctness',{}).get(residency,{}),data['shape'])
                        for cache in ('steady','eviction_attempt'):
                            selected = [s for s in raw if s['implementation']==name and s.get('residency')==residency and s.get('cache_policy')==cache]
                            fields = ('t_host_to_host_us',) if residency=='host_to_host' else FIELDS
                            append(f'{shape}/{name}/{residency}/{cache}',row,selected,fields)
                    separate[f'{shape}/{name}/first'] = row.get('first_variant_call_us_after_framework_setup')
                else:
                    if row.get('status') == 'PASS': gate(row,data['shape'])
                    append(f'{shape}/{name}',row,[s for s in raw if s['implementation']==name],(*FIELDS,'t_host_feed_us'))
                    separate[f'{shape}/{name}/first'] = row.get('first_call_initialization_search_us')
            if kind == 'boundaries':
                for name in variants:
                    for residency in ('target_layout_ready','caller_nchw'):
                        pairs.append((f'{shape}/{name}/{residency}/steady',f'{shape}/{name}/{residency}/eviction_attempt',False))
                    pairs.append((f'{shape}/{name}/target_layout_ready/steady',f'{shape}/{name}/caller_nchw/steady',False))
                for residency in ('target_layout_ready','caller_nchw'):
                    pairs.append((f'{shape}/torch_tuned_nchw/{residency}/steady',f'{shape}/torch_tuned_channels_last/{residency}/steady',False))
                separate[shape] = {k:data.get(k) for k in ('conversions','layout_initialization_search_us','eviction','order','search_cache')}
            else:
                pairs.append((f'{shape}/torch_cuda_tuned',f'{shape}/k3',False))
        context.append('boundaries use fixed implementation order; models tuned baseline is subprocess-fed; no statistical pairing inferred from labels')
    expected = plan['expected_units']
    require(len(expected) == len(set(expected)) and {u['unit'] for u in units} == set(expected), 'complete planned unit denominator mismatch')
    by = {u['unit']:u for u in units}; comparisons=[]
    for base,candidate,paired in pairs:
        a,b = by[base],by[candidate]
        if a['execution_status'] != 'PASS' or b['execution_status'] != 'PASS':
            comparisons.append({'baseline':base,'candidate':candidate,'assessment':'NOT_ASSESSED'}); continue
        for field in FIELDS:
            comparisons.append({'baseline':base,'candidate':candidate,'metric':field,**compare(a['batch_medians'][field],b['batch_medians'][field],paired=paired,draws=draws)})
    artifacts = [run/('summary.json' if kind == 'module' else 'provenance.json' if kind == 'boundaries' else 'run-key.json')]
    if kind == 'module': artifacts += [run/'samples.jsonl',run/'random-state.pt']
    elif kind == 'frontend': artifacts += [run/'bench.json']
    else:
        artifacts += sorted((run/'shapes').glob('*.json'))
        if kind == 'models': artifacts += [run/'model-shapes.json']
    if correctness is not None: artifacts.append(Path(correctness))
    return {'kind':kind,'assessment':'AUDITED_DESCRIPTIVE','planned_units':len(expected),
            'execution_counts':dict(Counter(u['execution_status'] for u in units)), 'units':units,
            'comparisons':comparisons,'setup_conversion_profile_separate':separate,'context':context,
            'source_tree_sha256':plan['source_tree_sha256'],'analysis_script_sha256':sha(__file__), 'python_version':platform.python_version(),
            'input_artifacts_sha256':{str(p):sha(p) for p in artifacts},
            'statistics':{'estimator':'median(batch medians); baseline/candidate ratio', 'ci':'paired batch-index bootstrap only for verified randomized module blocks', 'bootstrap_draws':draws, 'seed':20260908, 'practical_margin':1.05, 'minimum_batches_for_ci':10}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['boundaries','models','module','frontend'],required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--entry',required=True);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--correctness',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    plan={}
    try:
        plan=read(args.plan)['runs'][args.entry]
        result=analyze(args.kind,args.run,plan,args.snapshot,args.correctness)
    except (ValueError,KeyError,OSError,TypeError) as exc:
        result={'kind':args.kind,'assessment':'NOT_ASSESSED','reason':str(exc),'performance_failure_claimed':False, 'planned_units':len(plan.get('expected_units',[])),
                'units':[{'unit':u,'assessment':'NOT_ASSESSED','execution_status':'UNKNOWN'} for u in plan.get('expected_units',[])]}
    result['plan_sha256']=sha(args.plan)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    require(not args.output.exists(),'use a new analysis output path')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'assessment':result['assessment'],'output':str(args.output)}))
    return 2 if result['assessment']=='NOT_ASSESSED' else 0


if __name__=='__main__': raise SystemExit(main())
