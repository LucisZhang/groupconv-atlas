"""ctypes ABI v1. Array ownership stays with callers for every native call."""
from __future__ import annotations
import ctypes as ct
import dataclasses
import hashlib
import json
import os
import operator
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('n','cin','cout','h','w','r','s','groups','sh','sw','ph','pw','dh','dw')

@dataclasses.dataclass(frozen=True)
class Shape:
    n: int = 1
    cin: int = 16
    cout: int = 16
    h: int = 16
    w: int = 16
    r: int = 3
    s: int = 3
    groups: int = 1
    sh: int = 1
    sw: int = 1
    ph: int = 1
    pw: int = 1
    dh: int = 1
    dw: int = 1

    @property
    def output(self):
        return (self.n, self.cout,
                (self.h+2*self.ph-self.dh*(self.r-1)-1)//self.sh+1,
                (self.w+2*self.pw-self.dw*(self.s-1)-1)//self.sw+1)

    @property
    def input(self): return (self.n,self.cin,self.h,self.w)
    @property
    def weight(self): return (self.cout,self.cin//self.groups,self.r,self.s)
    @property
    def id(self):
        key = {**dataclasses.asdict(self), 'dtype':'float32', 'layout':'NCHW', 'bias':False}
        return 'gc_' + hashlib.sha256(json.dumps(key,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]

    def record(self):
        return {**dataclasses.asdict(self),'shape_id':self.id,'input_dims':self.input,
                'weight_dims':self.weight,'output_dims':self.output,'dtype':'float32','layout':'NCHW','bias':False}

    @classmethod
    def from_record(cls, value): return cls(**{k:value[k] for k in FIELDS})

class CShape(ct.Structure):
    _fields_ = [(field,ct.c_int64) for field in FIELDS]

class Tensor(ct.Structure):
    _fields_ = [('data',ct.c_void_p),('bytes',ct.c_uint64),('dtype',ct.c_int32),
                ('device',ct.c_int32),('device_index',ct.c_int32),('rank',ct.c_int32),
                ('dims',ct.c_int64*4),('strides',ct.c_int64*4)]

def native_int(value,bits=32):
    try: value=operator.index(value)
    except TypeError as e: raise ValueError('native integers must be integral') from e
    if not -(1<<(bits-1)) <= value < (1<<(bits-1)):
        raise ValueError(f'integer outside signed {bits}-bit range')
    return value

def cshape(shape): return CShape(*[native_int(getattr(shape,k),64) for k in FIELDS])

def tensor(array):
    if isinstance(array,np.ndarray):
        dtype={np.dtype('float32'):0,np.dtype('float64'):1}.get(array.dtype,-1)
        ptr, size = array.ctypes.data, array.nbytes
        shape, strides = array.shape, tuple(s//array.itemsize for s in array.strides)
        device,index=0,0
    else:
        dtype={'torch.float32':0,'torch.float64':1}.get(str(array.dtype),-1)
        ptr,size=array.data_ptr(),array.numel()*array.element_size()
        shape,strides=tuple(array.shape),tuple(array.stride())
        device,index=(1,array.device.index or 0) if array.device.type=='cuda' else (0,0)
        if array.device.type not in ('cpu','cuda'): device=-1
    return Tensor(ptr,size,dtype,device,index,len(shape),(ct.c_int64*4)(*shape),(ct.c_int64*4)(*strides))

class NativeError(RuntimeError):
    def __init__(self,status,message):
        self.status=status
        super().__init__(message)

class Native:
    def __init__(self,path=None):
        if path is None:
            candidates=list((ROOT/'build').glob('libgroupconv.*'))
            path=os.environ.get('GROUPCONV_LIBRARY') or (str(candidates[0]) if candidates else '')
        if not path: raise FileNotFoundError('Build the native library with ./run.sh cpu first')
        self.path=str(Path(path).resolve())
        self.lib=ct.CDLL(self.path)
        l=self.lib
        l.gc_last_error.restype=ct.c_char_p
        l.gc_build_info.restype=ct.c_char_p
        l.gc_abi_version.restype=ct.c_int
        if l.gc_abi_version()!=1: raise RuntimeError('native ABI version mismatch')
        l.gc_validate_shape.argtypes=[ct.POINTER(CShape),ct.POINTER(ct.c_int64),ct.POINTER(ct.c_int64)]
        l.gc_cpu_conv.argtypes=[ct.POINTER(CShape)]+[ct.POINTER(Tensor)]*3+[ct.c_int,ct.c_int]
        l.gc_cuda_conv.argtypes=[ct.POINTER(CShape)]+[ct.POINTER(Tensor)]*3+[ct.c_int,ct.c_void_p]
        l.gc_opencl_create.argtypes=[ct.POINTER(ct.c_void_p)]
        l.gc_opencl_info.argtypes=[ct.c_void_p]; l.gc_opencl_info.restype=ct.c_char_p
        l.gc_opencl_upload.argtypes=[ct.c_void_p,ct.POINTER(CShape)]+[ct.POINTER(Tensor)]*3
        l.gc_opencl_run.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.POINTER(ct.c_double)]
        l.gc_opencl_download.argtypes=[ct.c_void_p,ct.POINTER(Tensor)]
        l.gc_opencl_destroy.argtypes=[ct.c_void_p]; l.gc_opencl_destroy.restype=None

    def check(self,status):
        if status: raise NativeError(status,self.lib.gc_last_error().decode())

    def cpu(self,shape,x,w,y,impl=0,threads=1):
        p,xd,wd,yd=cshape(shape),tensor(x),tensor(w),tensor(y)
        self.check(self.lib.gc_cpu_conv(ct.byref(p),ct.byref(xd),ct.byref(wd),ct.byref(yd),native_int(impl),native_int(threads)))
        return y

    def cuda(self,shape,x,w,y,impl=0,stream=0):
        p,xd,wd,yd=cshape(shape),tensor(x),tensor(w),tensor(y)
        stream=operator.index(stream)
        if not 0<=stream<(1<<(ct.sizeof(ct.c_void_p)*8)): raise ValueError('stream pointer out of range')
        self.check(self.lib.gc_cuda_conv(ct.byref(p),ct.byref(xd),ct.byref(wd),ct.byref(yd),native_int(impl),stream))
        return y

class OpenCL:
    def __init__(self,native):
        self.native=native; self.context=ct.c_void_p()
        native.check(native.lib.gc_opencl_create(ct.byref(self.context)))
        self.info=native.lib.gc_opencl_info(self.context).decode()

    def upload(self,shape,x,w,y):
        p,xd,wd,yd=cshape(shape),tensor(x),tensor(w),tensor(y)
        self.native.check(self.native.lib.gc_opencl_upload(self.context,ct.byref(p),ct.byref(xd),ct.byref(wd),ct.byref(yd)))

    def run(self,impl=0,repeats=1):
        elapsed=ct.c_double()
        self.native.check(self.native.lib.gc_opencl_run(self.context,native_int(impl),native_int(repeats),ct.byref(elapsed)))
        return elapsed.value

    def download(self,y):
        yd=tensor(y)
        self.native.check(self.native.lib.gc_opencl_download(self.context,ct.byref(yd)))
        return y

    def close(self):
        if self.context.value:
            self.native.lib.gc_opencl_destroy(self.context)
            self.context=ct.c_void_p()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

def inputs(shape,seed=20260908):
    rng=np.random.default_rng(seed ^ int(shape.id[3:11],16))
    return rng.uniform(-1,1,shape.input).astype(np.float32), rng.uniform(-1,1,shape.weight).astype(np.float32)

def torch_reference(shape,x,w):
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        return F.conv2d(torch.from_numpy(x),torch.from_numpy(w),stride=(shape.sh,shape.sw),
                        padding=(shape.ph,shape.pw),dilation=(shape.dh,shape.dw),groups=shape.groups).numpy()

def error_metrics(actual,reference,atol=1e-4,rtol=1e-4,eps=1e-12):
    a=np.asarray(actual,dtype=np.float64); b=np.asarray(reference,dtype=np.float64)
    if a.shape!=b.shape: raise ValueError('reference shape mismatch')
    finite=np.isfinite(a)&np.isfinite(b)
    diff=np.abs(a-b)
    bad=(diff>atol+rtol*np.abs(b))|~finite
    return {'passed':bool(not bad.any()),'max_abs':float(np.max(diff)) if finite.all() else None,
            'max_rel':float(np.max(diff/(np.abs(b)+eps))) if finite.all() else None,
            'rms':float(np.sqrt(np.mean(diff*diff))) if finite.all() else None,
            'failed_elements':int(bad.sum()),'elements':int(a.size),'epsilon':eps,'atol':atol,'rtol':rtol}

def oracle_points(shape,x,w,positions):
    """Independent scalar FP64 point calculation, no native code dependency."""
    values=[]
    for n,co,oh,ow in positions:
        acc=0.0
        base=(co//(shape.cout//shape.groups))*(shape.cin//shape.groups)
        for ci in range(shape.cin//shape.groups):
            for r in range(shape.r):
                ih=oh*shape.sh-shape.ph+r*shape.dh
                for s in range(shape.s):
                    iw=ow*shape.sw-shape.pw+s*shape.dw
                    if 0<=ih<shape.h and 0<=iw<shape.w:
                        acc+=float(x[n,base+ci,ih,iw])*float(w[co,ci,r,s])
        values.append(acc)
    return np.array(values,dtype=np.float64)
