"""Run reviewed extension stages serially with durable receipts and a hard process budget."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .cloud_cuda_session import stop_tree, write_json
from bench.api import ROOT
from bench.run import sha, source_identity

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--correctness',type=Path,required=True)
    parser.add_argument('--model-shapes',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=10800)
    parser.add_argument('--stages',nargs='+',choices=['atlas','boundaries','models','module','frontend','triton','microbench'],
                        default=['atlas','boundaries','models','module','frontend','triton','microbench'])
    args=parser.parse_args(argv)
    if args.seconds<=0: parser.error('positive seconds required')
    out=args.output.resolve();cor=args.correctness.resolve();models=args.model_shapes.resolve()
    if out.exists(): raise ValueError('output must be new')
    out.mkdir(parents=True);os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('PYTHONUNBUFFERED','1')
    protocol=json.loads((ROOT/'configs/cuda-boundaries-protocol.json').read_text())
    jobs=[];py=sys.executable
    if 'atlas' in args.stages: jobs.append(('atlas',[py,'-m','bench.cuda_run','--shape-set','atlas','--batches','10','--output',str(out/'atlas'),'--correctness',str(cor)],7200))
    if 'boundaries' in args.stages: jobs.append(('boundaries',[py,'-m','bench.cuda_boundaries','--output',str(out/'boundaries'),'--correctness',str(cor)],5400))
    if 'models' in args.stages: jobs.append(('models',[py,'-m','bench.model_run','--shape-report',str(models),'--output',str(out/'models'),'--correctness',str(cor)],1800))
    if 'module' in args.stages:
        for n in (1,4): jobs.append((f'module-n{n}',[py,'-m','bench.module_run','--output',str(out/f'module-n{n}'),'--batch-size',str(n),'--batches','10','--timeout','600'],660))
    if 'frontend' in args.stages: jobs.append(('frontend',[py,'-m','bench.cudnn_frontend_run','--shape-set','atlas','--shape-ids',*protocol['shape_ids'],'--output',str(out/'frontend')],3600))
    if 'triton' in args.stages:
        common=[py,'-m','bench.triton_run']
        for phase in ('check','tune','evaluate'):
            extra=[]
            if phase=='tune':extra=['--check-report',str(out/'triton-check'/'receipt.json')]
            if phase=='evaluate':extra=['--frozen',str(out/'triton-tune'/'frozen.json')]
            jobs.append(('triton-'+phase,[*common,phase,'--library','build/libgroupconv.so','--correctness',str(cor),'--output',str(out/('triton-'+phase)),*extra],3600))
    if 'microbench' in args.stages: jobs.append(('microbench',['./build/cuda_microbench'],600))
    started=time.monotonic();receipts=[];active=None;status='RUNNING';error=None
    metadata={'source':source_identity(),'correctness_sha256':sha(cor),'model_shape_report_sha256':sha(models),
              'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'seconds':args.seconds,'stages':args.stages}
    def save(): write_json(out/'receipt.json',{**metadata,'status':status,'error':error,'commands':receipts,'elapsed_seconds':time.monotonic()-started,
            'provider_shutdown':'NOT_PERFORMED_BY_SCRIPT; caller must verify stop/destroy and storage billing'})
    def stop(signum,frame): raise KeyboardInterrupt('interrupted')
    previous=signal.signal(signal.SIGTERM,stop)
    try:
        save()
        for name,command,limit in jobs:
            remaining=args.seconds-(time.monotonic()-started)
            if remaining<=0: raise TimeoutError('session budget exhausted')
            path=out/'logs'/(name+'.log');path.parent.mkdir(exist_ok=True)
            row={'name':name,'command':command,'status':'RUNNING','started_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
            receipts.append(row);save();begin=time.monotonic();print('START',name,flush=True)
            with path.open('w') as f:
                active=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                try: code=active.wait(timeout=min(limit,remaining))
                except (subprocess.TimeoutExpired,KeyboardInterrupt):
                    stop_tree(active);row.update(status='INTERRUPTED_OR_TIMEOUT');raise
            # Direct Frontend returns 2 when all plans are explicitly unsupported.
            row.update(returncode=code,status='PASS' if code==0 else 'UNSUPPORTED' if name=='frontend' and code==2 else 'FAIL',
                       log=str(path.relative_to(out)),log_sha256=sha(path),elapsed_seconds=time.monotonic()-begin)
            active=None;save();print(row['status'],name,flush=True)
            if row['status']=='FAIL': raise RuntimeError(name+' failed')
        status='PASS'
    except BaseException as exc:
        status='FAIL';error=f'{type(exc).__name__}: {exc}'
    finally:
        if active is not None and active.poll() is None: stop_tree(active)
        signal.signal(signal.SIGTERM,previous);save()
    return 0 if status=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
