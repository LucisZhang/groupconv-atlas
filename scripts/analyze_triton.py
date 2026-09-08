"""Offline audit of frozen 031887c Triton check/tune/evaluate artifacts."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from unittest.mock import patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import triton_run as tr
from bench.cuda_run import VARIANTS
from scripts.analyze_supplemental import (require, sha, read, bind_source, native_receipt,
                                          strict_precision, fp32, fp64, shape_context)

FROZEN_TREE = '5b499cd77b1594e501fce8c8d457ec147ffe410ae5764fa6b1df1a4566934cbd'
FROZEN_FILE_COUNT = 71


class Artifacts:
    """Map one reviewed remote results root; never rewrite evidence bytes."""
    def __init__(self, local, remote):
        self.local, self.remote = Path(local).resolve(), Path(remote)
        require(self.remote.is_absolute(), 'remote root must be absolute')
        self.reads = {}

    def path(self, value):
        p = Path(value)
        if p.is_absolute() and p.is_relative_to(self.local):
            result = p.resolve()
        else:
            require(p.is_absolute() and p.is_relative_to(self.remote), 'artifact outside declared remote root')
            result = (self.local / p.relative_to(self.remote)).resolve()
        require(result.is_relative_to(self.local), 'artifact traversal or external symlink')
        return result

    def hash(self, value):
        p = self.path(value)
        result = sha(p)
        self.reads[str(p.relative_to(self.local))] = result
        return result

    def entry(self, entry):
        require(self.hash(entry['path']) == entry['sha256'], 'artifact hash mismatch')
        return read(self.path(entry['path']))

    @contextmanager
    def validators(self):
        with patch.object(tr, 'Path', self.path), patch.object(tr, 'sha', self.hash):
            yield


def runtime(env, p):
    require(env.get('torch') == '2.8.0+cu128' and env.get('cuda') == '12.8'
            and env.get('cudnn') == 91002 and env.get('triton') == p['triton_version'],
            'frozen runtime version mismatch')
    require(isinstance(env.get('torch_git'), str) and env['torch_git'], 'torch build identity missing')
    gpu = env.get('gpu', {})
    require(isinstance(gpu.get('name'), str) and gpu['name'] and
            isinstance(gpu.get('uuid'), str) and gpu['uuid'] not in ('', 'UNKNOWN'), 'GPU identity missing')
    require(type(gpu.get('memory')) is int and gpu['memory'] > 0 and
            isinstance(gpu.get('capability'), list) and len(gpu['capability']) == 2 and
            all(type(v) is int and v >= 0 for v in gpu['capability']), 'GPU properties missing')
    drivers = env.get('driver_uuid_versions', '')
    def uuid(value):
        require(isinstance(value,str) and re.fullmatch(r'(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',value) is not None,'malformed GPU UUID')
        return value.removeprefix('GPU-').lower()
    require(isinstance(drivers,str),'GPU driver readback missing')
    rows=[line.split(',') for line in drivers.splitlines() if line.strip()]
    require(len(rows)==1 and len(rows[0])==2 and uuid(rows[0][0].strip())==uuid(gpu['uuid'])
            and re.fullmatch(r'\d+(?:\.\d+)+',rows[0][1].strip()) is not None,'GPU UUID/driver identity mismatch')
    require(isinstance(gpu.get('visible_devices'), str), 'visibility identity missing')


def samples(rows, shape, p, implementation, blocks=None, expected_block=None):
    require(all(type(row.get(metric)) in (int,float) for row in rows for metric in tr.METRICS), 'nonnumeric timing')
    summary = tr.summarize(rows, p)
    repeats = set()
    for row in rows:
        require(all(type(row.get(k)) is int for k in ('batch','sample','repeats','implementation_order')), 'noninteger sample coordinates')
        require(row.get('shape_id') == shape.id and row.get('implementation') == implementation, 'sample identity mismatch')
        require(1 <= row['repeats'] <= p['repeat_cap'], 'repeat bound mismatch')
        repeats.add(row['repeats'])
        if blocks is not None:
            require(type(expected_block) is int and type(row.get('block')) is int and row['block'] == expected_block, 'sample block differs from enclosing candidate')
        order = list(blocks if blocks is not None else VARIANTS)
        random.Random(f"{p['seed']}:{shape.id}:{row['batch']}").shuffle(order)
        chosen = row.get('block') if blocks is not None else implementation
        require(chosen in order and row['implementation_order'] == order.index(chosen), 'sample order mismatch')
    require(len(repeats) == 1, 'repeat count drift')
    return summary


def denominator(receipt, shapes):
    expected = {'all_shapes':len(shapes), 'supported':sum(tr.supported(s) for s in shapes),
                'unsupported':sum(not tr.supported(s) for s in shapes)}
    require(receipt.get('denominator') == expected, 'phase denominator mismatch')


def validate_source(source, snapshot):
    require(len(source.get('source_files', {})) == FROZEN_FILE_COUNT, 'complete frozen source denominator required')
    bind_source(source, {'source_tree_sha256':FROZEN_TREE}, Path(snapshot))
    root = Path(__file__).resolve().parents[1]
    for name, module in tuple(sys.modules.items()):
        if (name == 'bench' or name.startswith('bench.') or name == 'backends.triton'
                or name.startswith('backends.triton.')) and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(root), 'validator imported outside local checkout')
            relative = str(path.relative_to(root))
            require(source['source_files'].get(relative) == sha(path), 'validator source differs: '+relative)


def analyze(root, remote_root, snapshot, library, correctness):
    root = Path(root).resolve()
    artifacts = Artifacts(root, remote_root)
    receipts = {phase:read(Path(root)/f'triton-{phase}'/'receipt.json') for phase in ('check','tune','evaluate')}
    for phase in receipts: artifacts.hash(Path(root)/f'triton-{phase}'/'receipt.json')
    source = receipts['check']['source']
    validate_source(source, snapshot)
    p = tr.load_protocol(Path(snapshot)/'configs/triton-protocol.json')
    require(p['correctness_seeds'] == [0,7,20260908], 'three frozen correctness seeds required')
    env = receipts['check'].get('environment', {})
    runtime(env, p)
    binding = {'source_tree_sha256':FROZEN_TREE, 'protocol_sha256':sha(Path(snapshot)/'configs/triton-protocol.json'),
               'split_sha256':sha(Path(snapshot)/'configs/split.json'), 'library_sha256':sha(library),
               'native_correctness_sha256':sha(correctness)}
    native_receipt({'source':source,'library_sha256':binding['library_sha256'],
                    'correctness_sha256':binding['native_correctness_sha256']}, correctness)
    subsets = tr.split_shapes(read(Path(snapshot)/'configs/atlas.json'),read(Path(snapshot)/'configs/split.json'))
    require(all(len(v) == 40 and sum(tr.supported(s) for s in v) == 15 for v in subsets.values()), 'frozen 15/25 split required')
    for phase, receipt in receipts.items():
        tr.require_binding(receipt,binding,env)
        require(receipt.get('source') == source and receipt.get('protocol') == p and
                receipt.get('kind') == 'triton_'+phase and receipt.get('device_correctness') == 'PASS', 'phase provenance mismatch')
        require(receipt.get('performance') == ('NOT_RUN' if phase == 'check' else 'PASS'), 'phase execution incomplete')
        denominator(receipt,tr.check_shapes() if phase == 'check' else subsets['tuning' if phase == 'tune' else 'holdout'])
    frozen_path = Path(root)/'triton-tune/frozen.json'
    frozen = read(frozen_path)
    require(artifacts.path(receipts['tune']['check_report']['path']) == (Path(root)/'triton-check/receipt.json').resolve(), 'foreign check receipt')
    require(artifacts.path(receipts['evaluate']['frozen']['path']) == frozen_path.resolve() and
            artifacts.hash(frozen_path) == receipts['evaluate']['frozen']['sha256'], 'foreign frozen selection')
    with artifacts.validators():
        tr.validate_frozen(frozen,frozen_path,binding,env,p,subsets['tuning'])
    # Existing frozen validation reconstructs selection, all candidates and all three-seed boundary oracles.
    # Add strict precision and timing checks absent from its selection-only validator.
    for phase in ('check','tune'):
        for job in receipts[phase]['jobs']:
            data = artifacts.entry(job); shape = tr.Shape.from_record(data['shape'])
            require(shape_context(data['shape'])[0] == shape.id, 'noncanonical shape')
            if not tr.supported(shape): continue
            strict_precision(data.get('precision'))
            for row in data['rows']:
                for check in row['correctness']: fp32(check['torch_cuda_fp32_full'], math.prod(shape.output))
                if phase == 'tune':
                    actual = samples(row['samples'],shape,p,'triton',list(tr.BLOCKS),row['block'])
                    require(row.get('summary') == actual, 'stored tuning summary mismatch')
    shapes = subsets['holdout']; expected = []
    for shape in shapes:
        suites = ['triton','cuda'] if tr.supported(shape) else ['triton']
        random.Random(f"{p['seed']}:{shape.id}:suites").shuffle(suites)
        expected.extend((shape.id,suite) for suite in suites)
    jobs = receipts['evaluate']['jobs']
    require([(j['shape_id'],j['suite']) for j in jobs] == expected, 'holdout job denominator/order mismatch')
    by_id = {s.id:s for s in shapes}; units=[]; results={}
    for job in jobs:
        data=artifacts.entry(job); shape=by_id[job['shape_id']]; suite=job['suite']
        require(data.get('binding') == binding and data.get('suite') == suite and
                shape_context(data['shape'])[0] == shape.id and tr.Shape.from_record(data['shape']) == shape,
                'holdout job identity mismatch')
        if not tr.supported(shape):
            require(len(data['rows']) == 3 and {r['block'] for r in data['rows']} == set(tr.BLOCKS) and
                    all(r['status'] == 'UNSUPPORTED' and r['shape_id'] == shape.id and not r.get('samples') for r in data['rows']), 'unsupported holdout denominator mismatch')
            units.append({'shape_id':shape.id,'implementation':'triton','status':'UNSUPPORTED','candidate_rows':3}); continue
        strict_precision(data.get('precision'))
        if suite == 'triton':
            block=frozen['selected'][str(shape.cin//shape.groups)]
            require(data.get('environment') == env and data.get('device_correctness') == data.get('performance') == 'PASS', 'holdout environment/execution mismatch')
            require(len(data['rows']) == 1 and data['rows'][0].get('block') == block, 'holdout changed frozen candidate')
            row=data['rows'][0]
            require(row.get('status') == 'PASS' and row.get('shape_id') == shape.id,'holdout Triton failed')
            with artifacts.validators(): tr.validate_candidate(row,shape,p,[p['seed']],False)
            fp32(row['correctness'][0]['torch_cuda_fp32_full'],math.prod(shape.output))
            summary=samples(row['samples'],shape,p,'triton',[block],block)
            require(row.get('summary') == summary, 'stored holdout summary mismatch')
            values=[('triton',summary,{'compile':row['compile']})]
        else:
            require(len(data['rows']) == len(VARIANTS) and {r['implementation'] for r in data['rows']} == set(VARIANTS), 'CUDA baseline denominator mismatch')
            require(data.get('stream') == 'non-default', 'CUDA stream boundary missing')
            require(len(data.get('samples',[])) == len(VARIANTS)*p['batches']*p['samples_per_batch'], 'CUDA raw denominator mismatch')
            values=[]
            x,w=tr.inputs(shape,p['seed']); oracle=tr.prepare_reference(shape,x,w,seed=p['seed'],count=p['fp64_points'])
            for row in data['rows']:
                name=row['implementation']
                require(row['status'] == 'PASS' and row['shape_id'] == shape.id, 'supported baseline failed')
                fp32(row.get('error',{}),math.prod(shape.output)); fp64(row.get('fp64_points',{}),shape.record())
                for key,value in oracle.items():
                    if key not in ('positions','expected_fp64'):
                        require(row['fp64_points'].get(key) == value, 'baseline FP64 input binding mismatch')
                points=row['fp64_points']['points']
                require([v['position'] for v in points] == oracle['positions'] and [v['expected_fp64'] for v in points] == oracle['expected_fp64'], 'baseline FP64 reference differs')
                require(type(row.get('first_call_initialization_search_us')) in (int,float) and math.isfinite(row['first_call_initialization_search_us']) and row['first_call_initialization_search_us'] > 0, 'baseline first-call evidence missing')
                if name.startswith('torch_'): require(row.get('benchmark') is (name == 'torch_cuda_tuned'), 'baseline naming/settings mismatch')
                if name == 'torch_cuda_tuned': require(row.get('cache_isolation') == 'separate fresh process', 'tuned cache isolation missing')
                summary=samples([s for s in data['samples'] if s['implementation'] == name],shape,p,name)
                values.append((name,summary,{'first_call_initialization_search_us':row['first_call_initialization_search_us']}))
        for name,summary,setup in values:
            units.append({'shape_id':shape.id,'implementation':name,'status':'PASS','summary':summary,'setup':setup})
            results[shape.id,name]=summary
    comparisons=[]
    for shape in shapes:
        for baseline in VARIANTS:
            item={'shape_id':shape.id,'baseline':baseline,'candidate':'triton','classification':'UNCERTAIN' if tr.supported(shape) else 'UNSUPPORTED','paired':False,'ci95':None}
            if tr.supported(shape): item['ratios']={m:results[shape.id,baseline][m]['median_batch_medians']/results[shape.id,'triton'][m]['median_batch_medians'] for m in tr.METRICS}
            comparisons.append(item)
    require(type(frozen.get('search_elapsed_seconds')) in (int,float) and math.isfinite(frozen['search_elapsed_seconds']) and frozen['search_elapsed_seconds'] > 0,'search elapsed missing')
    return {'analysis_status':'PASS','measurement_status':'PASS','statistical_scope':'DESCRIPTIVE_INDEPENDENT_SUITES_NO_PAIRED_CI',
            'metadata_semantic_review':'NOT_ASSESSED',
            'denominator':{'tuning':{'supported':15,'unsupported':25,'supported_candidate_rows':45,'unsupported_candidate_rows':75},'holdout':{'supported':15,'unsupported':25,'jobs':55,'triton_supported_rows':15,'triton_unsupported_candidate_rows':75,'cuda_rows':90}},
            'source':source,'binding':binding,'environment':env,'selected':frozen['selected'],
            'search_elapsed_seconds':frozen['search_elapsed_seconds'],'units':units,'counts':dict(Counter(u['status'] for u in units)),
            'comparisons':comparisons,'artifact_hashes':artifacts.reads,
            'limitations':['Receipts cannot independently prove execution or absence of unrecorded holdout access.',
                          'CUDA child jobs inherit receipt environment; they do not independently attest GPU identity.',
                          'PTX and native library bytes are hash-checked, not executed or independently recompiled.']}


def inventory(root, remote_root):
    """Unaudited states preserve failures and missing jobs on rejection."""
    base=Path(__file__).resolve().parents[1]
    subsets=tr.split_shapes(read(base/'configs/atlas.json'),read(base/'configs/split.json'))
    mapper=Artifacts(root,remote_root); phases={}
    for phase, shapes in [('check',tr.check_shapes()),('tune',subsets['tuning']),('evaluate',subsets['holdout'])]:
        path=Path(root)/f'triton-{phase}'/'receipt.json'
        try: receipt=read(path)
        except (OSError,ValueError): receipt={}
        jobs=receipt.get('jobs',[]); units=[]
        for shape in shapes:
            suites=['triton','cuda'] if phase=='evaluate' and tr.supported(shape) else ['triton']
            for suite in suites:
                matches=[j for j in jobs if j.get('shape_id')==shape.id and j.get('suite')==suite]
                value={'shape_id':shape.id,'suite':suite,'evidence_status':'NOT_ASSESSED',
                       'expected_support':tr.supported(shape),'observed_rows':[]}
                if len(matches)==1:
                    try:
                        data=mapper.entry(matches[0]);value['observed_rows']=[
                            {k:r[k] for k in ('block','implementation','status','reason') if k in r}
                            for r in data.get('rows',[])]
                    except (OSError,ValueError,KeyError,TypeError): pass
                value['matching_jobs']=len(matches);units.append(value)
        phases[phase]={'observed_receipt_status':receipt.get('status','MISSING'),
                       'expected_jobs':len(units),'observed_jobs':len(jobs),'units':units}
    return phases


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('results','remote-root','snapshot','library','correctness','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args(argv)
    require(not args.output.exists(),'output must be new')
    try:
        result=analyze(args.results,args.remote_root,args.snapshot,args.library,args.correctness)
    except (ValueError,KeyError,TypeError,OSError,AssertionError) as exc:
        result={'analysis_status':'NOT_ASSESSED','measurement_status':'NOT_ASSESSED','reason':str(exc),'comparisons':[],
                'unaudited_inventory':inventory(args.results,args.remote_root)}
    result['analyzer_sha256']=sha(__file__)
    result['analysis_helper_sha256']=sha(Path(__file__).with_name('analyze_supplemental.py'))
    result['python']=sys.version
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output: json.dump(result,output,indent=2,allow_nan=False)
    print(json.dumps({'analysis_status':result['analysis_status'],'output':str(args.output)}))
    return 0 if result['analysis_status'] == 'PASS' else 2


if __name__ == '__main__': raise SystemExit(main())
