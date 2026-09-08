"""Frozen-shape benchmark, incremental per-shape files, explicit timing boundaries."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
from .api import ROOT,Shape,Native,NativeError,OpenCL,inputs,error_metrics
from .stats import describe

def _torch_process(connection):
    # macOS torch and Homebrew ship different OpenMP runtimes. A process boundary
    # keeps both valid; IPC and input setup stay outside all reported timings.
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    connection.send({'version':str(torch.__version__),'build':torch.__config__.show(),
                     'backend_identity':'UNKNOWN','reason':'backend dispatch not profiled'})
    while True:
        task=connection.recv()
        if task is None: break
        try:
            if task[0]=='prepare':
                _,shape,x,w=task
                tx,tw=torch.from_numpy(x),torch.from_numpy(w)
                def operation():
                    return F.conv2d(tx,tw,stride=(shape.sh,shape.sw),padding=(shape.ph,shape.pw),
                                    dilation=(shape.dh,shape.dw),groups=shape.groups)
                with torch.inference_mode(): reference=operation().numpy()
                connection.send(('OK',reference))
            elif task[0]=='measure':
                _,threads,warmup,samples,target,cap=task
                torch.set_num_threads(threads)
                with torch.inference_mode():
                    for _ in range(warmup): operation()
                    start=time.perf_counter_ns(); operation(); single=(time.perf_counter_ns()-start)/1000
                    repeats=max(1,min(cap,int(target/max(single,1))))
                    values=[]
                    for _ in range(samples):
                        start=time.perf_counter_ns()
                        for _ in range(repeats): operation()
                        values.append((time.perf_counter_ns()-start)/1000/repeats)
                connection.send(('OK',{'api':values,'repeats':repeats}))
            elif task[0]=='check':
                torch.set_num_threads(task[1])
                with torch.inference_mode(): result=operation().numpy()
                connection.send(('OK',result))
            else: raise ValueError('unknown worker operation')
        except Exception as exc: connection.send(('ERROR',str(exc)))
    connection.close()

class TorchWorker:
    def __init__(self):
        ctx=mp.get_context('spawn'); self.connection,child=ctx.Pipe()
        self.process=ctx.Process(target=_torch_process,args=(child,)); self.process.start(); child.close()
        if not self.connection.poll(60): raise RuntimeError('PyTorch worker did not initialize')
        self.identity=self.connection.recv()
    def ask(self,*task):
        self.connection.send(task)
        if not self.connection.poll(120): raise RuntimeError('PyTorch worker timed out')
        status,data=self.connection.recv()
        if status!='OK': raise RuntimeError(data)
        return data
    def close(self):
        if self.process.is_alive(): self.connection.send(None)
        self.process.join(10)
        if self.process.is_alive(): self.process.terminate(); self.process.join()
        self.connection.close()

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(path)

def source_identity():
    paths=[]
    for name in ('csrc','bench','configs','tests','scripts','backends'):
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file()
                     and not {'__pycache__','.pytest_cache','.DS_Store'} & set(p.parts)
                     and p.suffix not in ('.pyc','.pyo'))
    paths.extend(ROOT/name for name in ('CMakeLists.txt','run.sh','requirements-lock.txt','requirements-cuda.txt') if (ROOT/name).exists())
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    except (subprocess.CalledProcessError,FileNotFoundError): commit=None; diff=b''
    return {'source_identity_version':2,'source_commit':commit,'workspace_diff_sha256':hashlib.sha256(diff).hexdigest(),
            'source_files':files,'source_tree_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}

def native_measure(call,samples,warmup,target,cap,device=False):
    for _ in range(warmup): call()
    start=time.perf_counter_ns(); event=call(); single=(time.perf_counter_ns()-start)/1000
    # OpenCL calls are synchronized once per event. Aggregating host calls is
    # declared as repeated logical calls, not one long asynchronous launch batch.
    repeats=max(1,min(cap,int(target/max(single,1))))
    api=[]; events=[]
    for _ in range(samples):
        timings=[]; start=time.perf_counter_ns()
        for _ in range(repeats):
            result=call()
            if device: timings.append(result)
        api.append((time.perf_counter_ns()-start)/1000/repeats)
        if device: events.append(float(np.mean(timings)))
    return {'api':api,'device':events,'repeats':repeats}

def measure(args):
    protocol=json.loads((ROOT/'configs/protocol.json').read_text())
    config=json.loads((ROOT/f'configs/{args.shape_set}.json').read_text())
    shapes=[Shape.from_record(s) for s in config['shapes']]
    if args.shape_ids: shapes=[s for s in shapes if s.id in args.shape_ids]
    if args.limit: shapes=shapes[:args.limit]
    if not shapes: raise ValueError('empty shape selection')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    protocol={**protocol,'batches':args.batches or protocol['batches'],
              'samples_per_batch':args.samples or protocol['samples_per_batch'],'threads':args.threads}
    identity=source_identity()
    if not identity['source_commit'] and not args.allow_uncommitted:
        raise RuntimeError('Commit a code snapshot before recording measurements')
    if not args.correctness and not args.allow_uncommitted:
        raise RuntimeError('A full correctness report is required before formal measurements')
    native=Native(); worker=TorchWorker(); gpu=None
    availability={}
    if 'opencl' in args.backends:
        try: gpu=OpenCL(native); availability['opencl']={'status':'PASS','identity':gpu.info}
        except NativeError as e: availability['opencl']={'status':'NOT_RUN','reason':str(e)}
    variants=[]
    if 'cpu' in args.backends:
        variants=[('c0',1)]+[(name,t) for name in ('c1','c2','torch_cpu') for t in args.threads]
    if 'opencl' in args.backends: variants += [('o0',1),('o1',1),('o2',1)]
    try:
        correctness_sha=None
        if args.correctness:
            validation=json.loads(Path(args.correctness).read_text())
            if validation['library']['sha256']!=sha(native.path): raise RuntimeError('correctness report library mismatch')
            required=[r for r in validation['records'] if r['backend'] in args.backends]
            if any(r['status'] in ('FAIL','NOT_RUN') for r in required) or validation['contract_counts'].get('FAIL',0):
                raise RuntimeError('required correctness checks failed or were not run')
            for backend in args.backends:
                covered={r['shape_id'] for r in required if r['backend']==backend and r['status']=='PASS'}
                if set(s.id for s in shapes)-covered: raise RuntimeError(f'{backend} correctness shape coverage incomplete')
            correctness_sha=sha(args.correctness)
        run_key={'source_tree_sha256':identity['source_tree_sha256'],'source_commit':identity['source_commit'],
                 'library_sha256':sha(native.path),'os':platform.platform(),'device':args.device,
                 'torch_identity':worker.identity,'availability':availability,'protocol':protocol,
                 'correctness_sha256':correctness_sha,'shapes':[s.id for s in shapes],'backends':args.backends}
        resume_path=out/'run-key.json'
        if resume_path.exists():
            if json.loads(resume_path.read_text())!=run_key: raise RuntimeError('resume source/build/environment/validation mismatch')
            if not args.resume: raise RuntimeError('run exists; use --resume with unchanged source/configuration')
        else: atomic_json(resume_path,run_key)
    except Exception:
        if gpu: gpu.close()
        worker.close()
        raise
    provenance={'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'device':args.device,
                'os':platform.platform(),'logical_cores':os.cpu_count(),'native_build':native.lib.gc_build_info().decode(),
                'library_sha256':sha(native.path),'torch_cpu':worker.identity,'protocol':protocol,
                'availability':availability,'source':identity,'exclusive_device':False,
                'frequency_temperature_power':'NOT_CONTROLLED','cpu_affinity':'OS default; not pinned',
                'environment':{k:os.environ.get(k,'UNSET') for k in ('OMP_DYNAMIC','OMP_PROC_BIND','OMP_PLACES','OMP_WAIT_POLICY','KMP_BLOCKTIME')},
                'cuda':'NOT_RUN','hardware_counters':'NOT_RUN','profile_times_in_main_table':False,
                'correctness_sha256':correctness_sha,'measurement_kind':'FORMAL' if args.correctness else 'SMOKE'}
    if not (out/'provenance.json').exists(): atomic_json(out/'provenance.json',provenance)
    else:
        with (out/'resume-sessions.jsonl').open('a') as f: f.write(json.dumps(provenance)+'\n')
    started=time.monotonic()
    try:
        for index,shape in enumerate(shapes):
            shard=out/'shapes'/f'{shape.id}.json'
            if shard.exists() and args.resume: continue
            x,w=inputs(shape,protocol['seed']); y=np.empty(shape.output,np.float32)
            reference=worker.ask('prepare',shape,x,w)
            if gpu: gpu.upload(shape,x,w,y)
            status={}; errors={}; all_samples=[]
            for name,threads in variants:
                key=f'{name}:{threads}'
                try:
                    if name=='torch_cpu': y[:]=worker.ask('check',threads)
                    elif name.startswith('c'): native.cpu(shape,x,w,y,int(name[1]),threads)
                    elif gpu: gpu.run(int(name[1])); gpu.download(y)
                    else: raise NativeError(4,availability['opencl']['reason'])
                    metrics=error_metrics(y,reference)
                    status[key]={'status':'PASS' if metrics['passed'] else 'FAIL','error':metrics}
                except NativeError as e:
                    status[key]={'status':{2:'UNSUPPORTED',4:'NOT_RUN'}.get(e.status,'FAIL'),'reason':str(e)}
            for batch in range(protocol['batches']):
                rng=random.Random(f"{protocol['seed']}:{shape.id}:{batch}")
                order=variants.copy(); rng.shuffle(order)
                for order_index,(name,threads) in enumerate(order):
                    key=f'{name}:{threads}'
                    if status[key]['status']!='PASS': continue
                    if name=='torch_cpu':
                        data=worker.ask('measure',threads,protocol['warmup'],protocol['samples_per_batch'],
                                        protocol['repeat_target_us'],protocol['repeat_cap'])
                    else:
                        call=(lambda: native.cpu(shape,x,w,y,int(name[1]),threads)) if name.startswith('c') else (lambda: gpu.run(int(name[1])))
                        data=native_measure(call,protocol['samples_per_batch'],protocol['warmup'],
                                            protocol['repeat_target_us'],protocol['repeat_cap'],name.startswith('o'))
                    host=[]
                    if name.startswith('o'):
                        for _ in range(protocol['samples_per_batch']):
                            t=time.perf_counter_ns(); gpu.upload(shape,x,w,y); gpu.run(int(name[1])); gpu.download(y)
                            host.append((time.perf_counter_ns()-t)/1000)
                    for sample,value in enumerate(data['api']):
                        all_samples.append({'shape_id':shape.id,'implementation':name,'threads':threads,
                            'batch':batch,'sample':sample,'implementation_order':order_index,'repeats':data['repeats'],
                            't_api_us':value,'t_device_op_us':(data.get('device') or [None]*len(data['api']))[sample],
                            't_host_to_host_us':host[sample] if host else None,'seed':protocol['seed'],
                            'layout':'NCHW','dtype':'float32','cache_policy':protocol['cache_policy']})
            rows=[]
            for name,threads in variants:
                item={'shape_id':shape.id,'implementation':name,'threads':threads,**status[f'{name}:{threads}'],
                      'n':shape.n,'cin':shape.cin,'cout':shape.cout,'h':shape.h,'w':shape.w,'groups':shape.groups,
                      'cpg':shape.cin//shape.groups,'layout':'NCHW','dtype':'float32','bias':False,
                      'cache_policy':protocol['cache_policy'],'batches':protocol['batches'],'samples_per_batch':protocol['samples_per_batch']}
                values=[s for s in all_samples if s['implementation']==name and s['threads']==threads]
                for boundary in ('api','device_op','host_to_host'):
                    field=f't_{boundary}_us'; v=[s[field] for s in values if s[field] is not None]
                    if v:
                        desc=describe(v); item[field]=desc['median']
                        item[f'p10_{boundary}_us']=desc['p10']; item[f'p90_{boundary}_us']=desc['p90']
                        meds=[float(np.median([s[field] for s in values if s['batch']==b])) for b in range(protocol['batches'])]
                        item[f'batch_medians_{boundary}_us']=meds
                        item[f'batch_cv_{boundary}']=float(np.std(meds,ddof=1)/np.mean(meds)) if len(meds)>1 else None
                    else: item[field]=None
                rows.append(item)
            atomic_json(shard,{'shape':shape.record(),'rows':rows,'samples':all_samples})
            failures=sum(v['status']=='FAIL' for v in status.values())
            print(f'{index+1}/{len(shapes)} C={shape.cin} H={shape.h} G={shape.groups}: {len(all_samples)} samples, failures={failures}',flush=True)
        records=[]; samples=[]; shape_records=[]
        for shape in shapes:
            data=json.loads((out/'shapes'/f'{shape.id}.json').read_text()); records.extend(data['rows']); samples.extend(data['samples']); shape_records.append(data['shape'])
        atomic_json(out/'bench.json',records); atomic_json(out/'shapes.json',shape_records)
        fields=sorted(set(k for r in records for k in r))
        with (out/'bench.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fields); writer.writeheader()
            for row in records: writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
        with (out/'samples.jsonl').open('w') as f:
            for sample in samples: f.write(json.dumps(sample,allow_nan=False)+'\n')
        atomic_json(out/'coverage.json',[{k:r[k] for k in ('shape_id','implementation','threads','status')} for r in records])
        counts={s:sum(r['status']==s for r in records) for s in ('PASS','FAIL','UNSUPPORTED','NOT_RUN')}
        atomic_json(out/'summary.json',{'shape_count':len(shapes),'row_count':len(records),'sample_count':len(samples),
                    'status_counts':counts,'elapsed_seconds':time.monotonic()-started,'protocol':protocol,
                    'hypotheses':{'cpu_parallel_speedup':'INCONCLUSIVE','shared_memory_speedup':'INCONCLUSIVE',
                                  'register_reuse_speedup':'INCONCLUSIVE'},
                    'interpretation':'descriptive medians; confidence classifications require independent confirmation batches',
                    'source_tree_sha256':identity['source_tree_sha256']})
        return 1 if counts['FAIL'] or counts['NOT_RUN'] else 0
    finally:
        if gpu: gpu.close()
        worker.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--device',default='apple-m4')
    p.add_argument('--shape-set',choices=['course','course-batch4','atlas'],default='course'); p.add_argument('--shape-ids',nargs='*')
    p.add_argument('--backends',nargs='+',choices=['cpu','opencl'],default=['cpu','opencl'])
    p.add_argument('--threads',nargs='+',type=int,default=[1,2,4,8]); p.add_argument('--batches',type=int); p.add_argument('--samples',type=int)
    p.add_argument('--limit',type=int); p.add_argument('--resume',action='store_true'); p.add_argument('--allow-uncommitted',action='store_true')
    p.add_argument('--correctness',type=Path)
    a=p.parse_args()
    if a.batches is not None and a.batches<1 or a.samples is not None and a.samples<1: p.error('positive batches and samples required')
    if any(t<1 or t>(os.cpu_count() or 1) for t in a.threads): p.error('thread counts must fit available logical cores')
    raise SystemExit(measure(a))
if __name__=='__main__': main()
