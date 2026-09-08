"""Read-only environment probe; no private host IDs or credentials in output."""
import argparse
import ctypes as ct
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from bench.api import Native,OpenCL,Shape,inputs
import numpy as np

def command(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return {'status':'PASS' if p.returncode==0 else 'ERROR','returncode':p.returncode,
                'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'status':'NOT_RUN','reason':str(e)}

def probe():
    import torch
    result={'schema_version':1,'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'os':platform.platform(),'architecture':platform.machine(),'python':platform.python_version(),
            'logical_cores':os.cpu_count(),'torch':torch.__version__,'numpy':np.__version__,
            'torch_build':torch.__config__.show(),'cuda_available':torch.cuda.is_available(),
            'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'counter_probe':{'status':'NOT_RUN','reason':'requires real NVIDIA device and actual metric collection'},
            'exclusive_device':False,'clock_lock':'NOT_RUN','temperature':'UNKNOWN','power':'UNKNOWN',
            'cpu_affinity':'not set; operating system scheduling','background_load':'not controlled',
            'toolchain':{k:command([k,'--version']) for k in ('cmake','clang++','nvcc','ncu','nsys')}}
    if platform.system()=='Darwin':
        result['cpu_details']=command(['sysctl','hw.physicalcpu','hw.logicalcpu','hw.memsize','machdep.cpu.brand_string'])
    cache=Path('build/CMakeCache.txt')
    if cache.exists():
        selected=next((line.split('=',1)[1] for line in cache.read_text().splitlines() if line.startswith('CMAKE_CXX_COMPILER:')),None)
        if selected: result['configured_compiler']=command([selected,'--version'])
    if torch.cuda.is_available():
        result['nvidia_smi']=command(['nvidia-smi','--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,temperature.gpu,power.limit','--format=csv'])
    try:
        lib=Native()
        result['native']={'status':'PASS','abi':lib.lib.gc_abi_version(),'build':lib.lib.gc_build_info().decode(),
                          'library_sha256':hashlib.sha256(Path(lib.path).read_bytes()).hexdigest(),
                          'abi_sizes':{'shape':ct.sizeof(__import__('bench.api',fromlist=['CShape']).CShape),
                                       'tensor':ct.sizeof(__import__('bench.api',fromlist=['Tensor']).Tensor)}}
        with OpenCL(lib) as gpu:
            shape=Shape(cin=2,cout=2,h=3,w=5,groups=2)
            x,w=inputs(shape); y=np.empty(shape.output,np.float32)
            gpu.upload(shape,x,w,y); elapsed=gpu.run(); gpu.download(y)
            from bench.api import torch_reference,error_metrics
            result['opencl']={'status':'PASS','info':gpu.info,'event_us':elapsed,
                              'correctness':error_metrics(y,torch_reference(shape,x,w))}
    except Exception as e:
        result['opencl']={'status':'NOT_RUN','reason':str(e)}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='docs/execution/environment-probe.json'); a=p.parse_args()
    value=probe(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:value[k] for k in ('torch','cuda_available','opencl')},ensure_ascii=False))
