"""Offline audit of one frozen CUDA Atlas; never runs a device or merges sessions."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import random
import platform
from pathlib import Path
import sys
import numpy as np
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.api import Shape
from bench.check import expected_support
from bench.cuda_run import validate_correctness

VARIANTS = ('k0','k1','k2','k3','torch_cuda_default','torch_cuda_tuned')
FIELDS = ('t_device_op_graph_us','t_api_us')
PAIRS = (('k0','k1'),('k0','k2'),('k0','k3'),('k1','k2'),('torch_cuda_tuned','k3'))
STATES = {'PASS','FAIL','UNSUPPORTED','OOM','NOT_RUN','TIMEOUT','UNSTABLE'}

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())

def compare(base, candidate, draws=10000, seed=20260908):
    a,b = np.asarray(base,float),np.asarray(candidate,float)
    require(a.ndim == b.ndim == 1 and len(a)==len(b) and len(a)>0, 'paired batches required')
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(a>0) and np.all(b>0), 'positive finite medians required')
    ratio=float(np.median(a)/np.median(b))
    result={'ratio':ratio,'batch_count':len(a),'ci95':None,'classification':'UNCERTAIN'}
    if len(a)<10: return result
    require(draws>=1000,'at least 1000 bootstrap draws')
    indexes=np.random.default_rng(seed).integers(0,len(a),(draws,len(a)))
    values=np.median(a[indexes],axis=1)/np.median(b[indexes],axis=1)
    lo,hi=map(float,np.quantile(values,[.025,.975]))
    result.update(ci95=[lo,hi],classification='WIN' if lo>1.05 else 'LOSS' if hi<1/1.05 else 'TIE' if lo>=1/1.05 and hi<=1.05 else 'UNCERTAIN')
    return result

def canonical_record(record, shape):
    canonical=json.loads(json.dumps(shape.record()))
    actual=json.loads(json.dumps(record))
    require(actual==canonical and all(type(actual[k]) is int for k in ('n','cin','cout','h','w','r','s','groups','sh','sw','ph','pw','dh','dw')) and actual['bias'] is False and all(type(v) is int for field in ('input_dims','weight_dims','output_dims') for v in actual[field]), 'canonical shape record mismatch')


def strict_precision(settings):
    legacy={'api':'allow_tf32','matmul':False,'cudnn':False}
    modern={k:'ieee' for k in ('global','matmul','cudnn','conv','rnn')} | {'api':'fp32_precision'}
    require(settings==legacy or settings==modern, 'strict FP32 precision readback mismatch')
    if settings.get('api')=='allow_tf32':
        require(settings['matmul'] is False and settings['cudnn'] is False,'strict FP32 boolean readback required')


def analysis_dependencies(snapshot, source_files):
    root=Path(__file__).resolve().parents[1]
    dependencies={}
    for name,module in sorted(sys.modules.items()):
        raw=getattr(module,'__file__',None)
        if not raw: continue
        path=Path(raw).resolve()
        if name=='bench' or name.startswith('bench.') or name=='scripts':
            require(path.is_relative_to(root),'analysis repository import outside workspace')
            relative=str(path.relative_to(root))
            require(relative in source_files and sha(path)==source_files[relative], 'analysis dependency differs from frozen snapshot: '+relative)
            require((snapshot/relative).is_file() and sha(snapshot/relative)==sha(path),'analysis snapshot dependency mismatch')
            dependencies[name]={'path':relative,'sha256':sha(path)}
    # Python and NumPy determine quantiles/RNG; record loaded binary/source files,
    # without executing any Python from the supplied snapshot.
    runtime={name:{'file':Path(module.__file__).name,'sha256':sha(module.__file__)}
             for name,module in sorted(sys.modules.items())
             if getattr(module,'__file__',None) and Path(module.__file__).is_file()
             and (name=='numpy' or name.startswith('numpy.') or name.split('.')[0] in sys.stdlib_module_names)}
    return {'repository':dependencies,'python':platform.python_version(),
            'python_executable_sha256':sha(sys.executable),'numpy_version':np.__version__,'loaded_numpy_and_stdlib_files':runtime}


def audit_shape(data, shape, batches, seed=20260908, repeat_cap=20, precision=None):
    canonical_record(data['shape'],shape)
    if any(r.get('status')=='PASS' for r in data['rows']):
        strict_precision(data.get('precision',{}))
        require(data['precision']==precision,'shape precision differs from run readback')
    rows=data['rows']; samples=data['samples']
    require(len(rows)==6 and {r['implementation'] for r in rows}==set(VARIANTS),'six unique implementations required')
    require(all(s['shape_id']==shape.id and s['implementation'] in VARIANTS for s in samples),'foreign sample')
    indexed={}; medians={}; statuses={}; repeats={}
    orders={}
    for batch in range(batches):
        order=list(VARIANTS);random.Random(f'{seed}:{shape.id}:{batch}').shuffle(order)
        orders[batch]=order
    for sample in samples:
        k=(sample['implementation'],sample['batch'],sample['sample'])
        require(k not in indexed,'duplicate sample')
        require(type(k[1]) is int and 0<=k[1]<batches and type(k[2]) is int and 0<=k[2]<30,'sample index out of range')
        require(all(isinstance(sample[f],(int,float)) and not isinstance(sample[f],bool) and math.isfinite(sample[f]) and sample[f]>0 for f in (*FIELDS,'t_host_feed_us')),'invalid timing')
        require(type(sample.get('implementation_order')) is int and sample['implementation_order']==orders[k[1]].index(k[0]),'implementation order mismatch')
        repeat=sample.get('repeats')
        require(type(repeat) is int and 1<=repeat<=repeat_cap,'invalid repeats')
        require(repeats.setdefault(k[0],repeat)==repeat,'repeats changed within implementation')
        indexed[k]=sample
    for row in rows:
        name=row['implementation']; status=row['status']; statuses[name]=status
        require(row['shape_id']==shape.id and status in STATES,'invalid row identity/status')
        supported=not name.startswith('k') or expected_support('cuda',shape,int(name[1]))
        require(not (status=='UNSUPPORTED' and supported),'supported shape reported unsupported')
        require(not (status=='PASS' and not supported),'unsupported shape reported PASS')
        selected=[s for s in samples if s['implementation']==name]
        if status=='UNSUPPORTED': require(not selected,'unsupported samples')
        if status!='PASS': continue
        require(len(selected)==batches*30,'PASS sample count mismatch')
        err=row.get('error',{}); fp=row.get('fp64_points',{})
        require(err.get('passed') is True and type(err.get('failed_elements')) is int and err['failed_elements']==0 and type(err.get('elements')) is int and err['elements']==math.prod(shape.output) and err.get('atol')==err.get('rtol')==1e-4,'missing full FP32 pass')
        require(err.get('epsilon')==1e-12 and all(type(err.get(k)) in (int,float) and math.isfinite(err[k]) and err[k]>=0 for k in ('max_abs','max_rel','rms')) and err['rms']<=err['max_abs'],'invalid full FP32 numeric summary')
        require(fp.get('passed') is True and fp.get('failed_points')==0 and fp.get('point_count')==64 and len(fp.get('points',[]))==64 and fp.get('shape_id')==shape.id,'missing FP64 point pass')
        require(fp.get('atol')==fp.get('rtol')==1e-4 and len({tuple(p['position']) for p in fp['points']})==64,'FP64 point identity/tolerance mismatch')
        require(all(len(p['position'])==4 and all(type(v) is int and 0<=v<dim for v,dim in zip(p['position'],shape.output)) for p in fp['points']),'FP64 positions outside shape')
        require(all(p.get('passed') is True and p.get('finite') is True and math.isfinite(p['actual']) and math.isfinite(p['expected_fp64']) and abs(p['actual']-p['expected_fp64'])<=fp['atol']+fp['rtol']*abs(p['expected_fp64']) for p in fp['points']),'FP64 numeric failure')
        medians[name]={f:[float(np.median([indexed[name,b,s][f] for s in range(30)])) for b in range(batches)] for f in FIELDS}
    return medians,statuses

def analyze(atlas, correctness, snapshot, draws=10000):
    atlas,correctness,snapshot=map(Path,(atlas,correctness,snapshot))
    paths={}
    def load(path):
        paths[str(path)]=sha(path); return read(path)
    key=load(atlas/'run-key.json'); prov=load(atlas/'provenance.json'); summary=load(atlas/'summary.json')
    config=load(snapshot/'configs/atlas.json'); protocol=load(snapshot/'configs/cuda-protocol.json')
    dependencies=analysis_dependencies(snapshot,key['source']['source_files'])
    shapes=[Shape.from_record(r) for r in config['shapes']]
    for record,shape in zip(config['shapes'],shapes): canonical_record(record,shape)
    strict_precision(key.get('precision',{}))
    require(len(shapes)==80 and len({s.id for s in shapes})==80,'80 frozen Atlas shapes required')
    require(key['shapes']==[s.id for s in shapes],'frozen shape order/IDs mismatch')
    require(key['measurement_kind']=='FORMAL','SMOKE excluded')
    expected=dict(protocol,batches=10)
    require(key['protocol']==expected,'not frozen Atlas 10 x 30 protocol')
    require(expected['samples_per_batch']==30 and expected['fp64_points']==64 and expected['tf32'] is False,'unsupported precision/sampling protocol')
    require(all(prov.get(k)==v for k,v in key.items()),'mixed provenance/run key')
    require(key['gpu'].get('uuid') not in (None,'UNKNOWN',''),'GPU UUID missing')
    source=key['source']; files=source['source_files']
    require({'configs/atlas.json','configs/cuda-protocol.json'}<=set(files),'frozen configuration absent from source identity')
    require(hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()==source['source_tree_sha256'],'source tree digest mismatch')
    for name,digest in files.items():
        path=(snapshot/name).resolve()
        require(path.is_relative_to(snapshot.resolve()) and path.is_file() and sha(path)==digest,'snapshot file mismatch: '+name)
    report=load(correctness)
    require(sha(correctness)==key['correctness_sha256'],'correctness digest mismatch')
    validate_correctness(report,source,key['library_sha256'])
    require(report['source']['source_files']==files,'correctness source files mismatch')
    require(summary['source_tree_sha256']==source['source_tree_sha256'] and summary['protocol']==expected and summary['measurement_kind']=='FORMAL','summary binding mismatch')
    require({p.stem for p in (atlas/'shapes').glob('*.json')}=={s.id for s in shapes},'shape file set mismatch')
    rows=[]; allsamples=[]; comparisons=[]; allstatuses=Counter()
    for shape in shapes:
        data=load(atlas/'shapes'/f'{shape.id}.json'); meds,statuses=audit_shape(data,shape,10,expected['seed'],expected['repeat_cap'],key['precision'])
        rows.extend(data['rows']); allsamples.extend(data['samples']); allstatuses.update(statuses.values())
        for base,candidate in PAIRS:
            for field in FIELDS:
                result={'shape_id':shape.id,'baseline':base,'candidate':candidate,'boundary':field,'baseline_status':statuses[base],'candidate_status':statuses[candidate]}
                if base in meds and candidate in meds: result.update(compare(meds[base][field],meds[candidate][field],draws),baseline_batch_medians=meds[base][field],candidate_batch_medians=meds[candidate][field])
                else: result.update(ratio=None,ci95=None,classification='NOT_COMPARABLE')
                comparisons.append(result)
    bench=load(atlas/'bench.json')
    require(len(bench)==480 and len({(r['shape_id'],r['implementation']) for r in bench})==480,'bench denominator mismatch')
    by={(r['shape_id'],r['implementation']):r for r in bench}
    for row in rows:
        saved=by[row['shape_id'],row['implementation']]
        require(all(saved.get(k)==v for k,v in row.items()),'bench row differs from shape record')
        selected=[x for x in allsamples if x['shape_id']==row['shape_id'] and x['implementation']==row['implementation']]
        require(saved['sample_count']==len(selected),'bench sample count mismatch')
        for field in (*FIELDS,'t_host_feed_us'):
            if not selected: continue
            values=[x[field] for x in selected]
            actual=saved[field]
            for metric,q in [('median',.5),('p10',.1),('p90',.9)]:
                require(math.isclose(actual[metric],float(np.quantile(values,q)),rel_tol=1e-12),'saved descriptive summary mismatch')
            bm=[float(np.median([x[field] for x in selected if x['batch']==b])) for b in sorted({x['batch'] for x in selected})]
            require(actual['batch_medians']==bm,'saved batch medians mismatch')
    samplepath=atlas/'samples.jsonl'; paths[str(samplepath)]=sha(samplepath)
    require([json.loads(line) for line in samplepath.read_text().splitlines()]==allsamples,'aggregate raw samples differ')
    require(summary['row_count']==summary['expected_row_count']==480 and summary['shape_count']==summary['completed_shape_count']==80 and summary['sample_count']==len(allsamples),'summary denominator mismatch')
    require(all(summary['status_counts'].get(s,0)==allstatuses[s] for s in STATES),'summary statuses mismatch')
    aggregates=aggregate_comparisons(comparisons)
    return {'audit_status':'PASS','measurement_status':'COMPLETE' if not any(allstatuses[s] for s in STATES-{'PASS','UNSUPPORTED'}) else 'INCOMPLETE','status_counts':dict(allstatuses),'binding':key,'input_sha256':paths,'script_sha256':sha(__file__),'analysis_dependencies':dependencies,'statistics':{'point':'median(batch medians baseline) / median(batch medians candidate)','bootstrap':'paired batch resampling; within this run only, not cross-run stability','draws':draws,'seed':20260908,'practical_threshold':1.05},'comparisons':comparisons,'aggregates':aggregates,'other_types':{k:{'analysis_status':'NOT_IMPLEMENTED','measurement_status':'NOT_ASSESSED'} for k in ('models','boundaries','frontend','triton','module','microbench')},'limitations':['Receipt consistency is not an independent proof of execution or process isolation.','Run-level UUID is consistent; identity-free shape shards cannot independently exclude mixed GPU/source data. Download manifest and execution receipts must supplement this check.','FP64 receipts are checked for distinct valid points and numerical inequalities; expected values are not recomputed from inputs here.']}

def aggregate_comparisons(comparisons):
    aggregates=[]
    for base,candidate in PAIRS:
        for field in FIELDS:
            part=[r for r in comparisons if r['baseline']==base and r['candidate']==candidate and r['boundary']==field]
            ratios=[r['ratio'] for r in part if r['ratio'] is not None]
            worst=min((r for r in part if r['ratio'] is not None),key=lambda r:(r['ratio'],r['shape_id']),default=None)
            aggregates.append({'worst_ratio':worst['ratio'] if worst else None,'worst_shape_id':worst['shape_id'] if worst else None,'worst_classification':worst['classification'] if worst else None,'baseline':base,'candidate':candidate,'boundary':field,'total_shapes':80,'comparable':len(ratios),'classification_counts':dict(Counter(r['classification'] for r in part)),'baseline_status_counts':dict(Counter(r['baseline_status'] for r in part)),'candidate_status_counts':dict(Counter(r['candidate_status'] for r in part)),'geomean':math.exp(sum(map(math.log,ratios))/len(ratios)) if ratios else None})
    return aggregates


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--atlas',type=Path,required=True);p.add_argument('--correctness',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True,help='extracted frozen measured source, not current workspace')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    require(not args.output.exists(),'output must be new')
    result=analyze(args.atlas,args.correctness,args.snapshot)
    args.output.mkdir(parents=True)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['shape_id','baseline','candidate','boundary','baseline_status','candidate_status','ratio','ci95','classification']
    with (args.output/'comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(r[k]) if isinstance(r.get(k),list) else r.get(k) for k in fields} for r in result['comparisons'])
    print(json.dumps({'audit_status':result['audit_status'],'measurement_status':result['measurement_status'],'aggregates':result['aggregates']},ensure_ascii=False))
    return 0
if __name__=='__main__':raise SystemExit(main())
