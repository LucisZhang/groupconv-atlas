#include "groupconv.h"
#define CL_TARGET_OPENCL_VERSION 120
#define CL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "opencl_source.h"
#include <cstring>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct Context {
    cl_device_id device=nullptr;
    cl_context context=nullptr;
    cl_command_queue queue=nullptr;
    cl_program program=nullptr;
    cl_kernel kernels[3]={};
    cl_mem buffers[3]={};
    uint64_t capacity[3]={};
    gc_shape shape={};
    bool ready=false;
    bool output_valid=false;
    std::string info;
    ~Context() {
        if(queue) clFinish(queue);
        for(auto b:buffers) if(b) clReleaseMemObject(b);
        for(auto k:kernels) if(k) clReleaseKernel(k);
        if(program) clReleaseProgram(program);
        if(queue) clReleaseCommandQueue(queue);
        if(context) clReleaseContext(context);
    }
};
int fail(cl_int code, const char *where) {
    std::string message=std::string(where)+" (OpenCL "+std::to_string(code)+")";
    return gc_error(GC_DEVICE_ERROR,message.c_str());
}
std::string info_string(cl_device_id d, cl_device_info key) {
    size_t size=0;
    if(clGetDeviceInfo(d,key,0,nullptr,&size)!=CL_SUCCESS || !size) return "UNKNOWN";
    std::vector<char> value(size);
    if(clGetDeviceInfo(d,key,size,value.data(),nullptr)!=CL_SUCCESS) return "UNKNOWN";
    return value.data();
}
int create_impl(void **out) {
    if(!out) return gc_error(GC_INVALID,"null context output");
    *out=nullptr;
    cl_uint count=0;
    cl_int rc=clGetPlatformIDs(0,nullptr,&count);
    if(rc!=CL_SUCCESS || !count) return gc_error(GC_UNAVAILABLE,"no OpenCL platform");
    std::vector<cl_platform_id> platforms(count);
    rc=clGetPlatformIDs(count,platforms.data(),nullptr);
    if(rc!=CL_SUCCESS) return fail(rc,"platform enumeration");
    auto *ctx=new Context;
    for(auto platform:platforms) {
        rc=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&ctx->device,nullptr);
        if(rc==CL_SUCCESS) break;
        ctx->device=nullptr;
    }
    if(!ctx->device) { delete ctx; return gc_error(GC_UNAVAILABLE,"no OpenCL GPU device"); }
    ctx->context=clCreateContext(nullptr,1,&ctx->device,nullptr,nullptr,&rc);
    if(rc!=CL_SUCCESS) { delete ctx; return fail(rc,"context creation"); }
    ctx->queue=clCreateCommandQueue(ctx->context,ctx->device,CL_QUEUE_PROFILING_ENABLE,&rc);
    if(rc!=CL_SUCCESS) { delete ctx; return fail(rc,"profiling queue creation"); }
    size_t length=std::strlen(gc_opencl_source);
    ctx->program=clCreateProgramWithSource(ctx->context,1,&gc_opencl_source,&length,&rc);
    if(rc!=CL_SUCCESS) { delete ctx; return fail(rc,"program creation"); }
    rc=clBuildProgram(ctx->program,1,&ctx->device,"-cl-std=CL1.2",nullptr,nullptr);
    if(rc!=CL_SUCCESS) {
        size_t length=0;
        clGetProgramBuildInfo(ctx->program,ctx->device,CL_PROGRAM_BUILD_LOG,0,nullptr,&length);
        std::vector<char> log(length+1,0);
        clGetProgramBuildInfo(ctx->program,ctx->device,CL_PROGRAM_BUILD_LOG,length,log.data(),nullptr);
        std::string error="OpenCL compile failed: "+std::string(log.data());
        delete ctx;
        return gc_error(GC_DEVICE_ERROR,error.c_str());
    }
    const char *names[]={"naive","shared_tile","register_tile"};
    for(int i=0;i<3;++i) {
        ctx->kernels[i]=clCreateKernel(ctx->program,names[i],&rc);
        if(rc!=CL_SUCCESS) { delete ctx; return fail(rc,"kernel creation"); }
        size_t max_wg=0;
        rc=clGetKernelWorkGroupInfo(ctx->kernels[i],ctx->device,CL_KERNEL_WORK_GROUP_SIZE,sizeof(max_wg),&max_wg,nullptr);
        if(rc!=CL_SUCCESS || max_wg<128) { delete ctx; return gc_error(GC_UNSUPPORTED,"OpenCL kernel requires work-group size 128"); }
    }
    cl_ulong local_mem=0;
    clGetDeviceInfo(ctx->device,CL_DEVICE_LOCAL_MEM_SIZE,sizeof(local_mem),&local_mem,nullptr);
    std::ostringstream info;
    info << "device=" << info_string(ctx->device,CL_DEVICE_NAME)
         << ";vendor=" << info_string(ctx->device,CL_DEVICE_VENDOR)
         << ";version=" << info_string(ctx->device,CL_DEVICE_VERSION)
         << ";driver=" << info_string(ctx->device,CL_DRIVER_VERSION)
         << ";local_memory_bytes=" << local_mem
         << ";queue=in_order_profiling;build=-cl-std=CL1.2;fp_contract=off;fast_math=off";
    ctx->info=info.str();
    *out=ctx;
    return GC_OK;
}
}

int gc_opencl_create(void **out) {
    try { return create_impl(out); }
    catch(const std::bad_alloc &) { return gc_error(GC_DEVICE_ERROR,"host allocation failed"); }
}
const char *gc_opencl_info(void *p) { return p?static_cast<Context *>(p)->info.c_str():"uninitialized"; }
void gc_opencl_destroy(void *p) { delete static_cast<Context *>(p); }
int gc_opencl_upload(void *opaque,const gc_shape *p,const gc_tensor *x,const gc_tensor *w,const gc_tensor *y) {
    if(!opaque) return gc_error(GC_INVALID,"null OpenCL context");
    auto &c=*static_cast<Context *>(opaque);
    c.ready=false; c.output_valid=false;
    int status=gc_validate_tensors(p,x,w,y,GC_CPU,GC_F32);
    if(status!=GC_OK) return status;
    if(p->cin!=p->cout || p->r!=3 || p->s!=3 || p->sh!=1 || p->sw!=1 || p->ph!=1 || p->pw!=1 || p->dh!=1 || p->dw!=1)
        return gc_error(GC_UNSUPPORTED,"OpenCL supports stage B geometry only");
    const gc_tensor *tensors[]={x,w,y};
    for(int i=0;i<3;++i) {
        uint64_t bytes=4;
        for(auto dim:tensors[i]->dims) bytes*=static_cast<uint64_t>(dim);
        if(c.capacity[i]<bytes) {
            if(c.buffers[i]) clReleaseMemObject(c.buffers[i]);
            c.buffers[i]=nullptr; c.capacity[i]=0;
            cl_int rc=0;
            c.buffers[i]=clCreateBuffer(c.context,i==2?CL_MEM_WRITE_ONLY:CL_MEM_READ_ONLY,bytes,nullptr,&rc);
            if(rc!=CL_SUCCESS) return fail(rc,"device allocation");
            c.capacity[i]=bytes;
        }
        if(i<2) {
            cl_int rc=clEnqueueWriteBuffer(c.queue,c.buffers[i],CL_TRUE,0,bytes,tensors[i]->data,0,nullptr,nullptr);
            if(rc!=CL_SUCCESS) return fail(rc,"host to device copy");
        }
    }
    c.shape=*p; c.ready=true;
    return GC_OK;
}
int gc_opencl_run(void *opaque,int impl,int repeats,double *device_us) {
    if(!opaque || !device_us || repeats<1 || repeats>100000) return gc_error(GC_INVALID,"invalid OpenCL run argument");
    auto &c=*static_cast<Context *>(opaque);
    if(!c.ready) return gc_error(GC_INVALID,"OpenCL upload required");
    if(impl<0 || impl>2) return gc_error(GC_UNSUPPORTED,"unknown OpenCL implementation");
    c.output_valid=false;
    cl_kernel kernel=c.kernels[impl];
    for(int i=0;i<3;++i) {
        cl_int rc=clSetKernelArg(kernel,i,sizeof(cl_mem),&c.buffers[i]);
        if(rc!=CL_SUCCESS) return fail(rc,"buffer kernel argument");
    }
    const auto &p=c.shape;
    cl_long args[]={p.n,p.cin,p.h,p.w,p.groups};
    for(int i=0;i<5;++i) {
        cl_int rc=clSetKernelArg(kernel,i+3,sizeof(cl_long),&args[i]);
        if(rc!=CL_SUCCESS) return fail(rc,"shape kernel argument");
    }
    size_t global[3]={}, local[3]={128,1,1};
    cl_uint ndim=1;
    if(impl==1) {
        ndim=3; local[0]=16; local[1]=8;
        global[0]=static_cast<size_t>((p.w+15)/16)*16;
        global[1]=static_cast<size_t>((p.h+7)/8)*8;
        global[2]=static_cast<size_t>(p.n*p.cout);
    } else {
        uint64_t count=static_cast<uint64_t>(p.n*p.cout*p.h)*(impl==2?(p.w+3)/4:p.w);
        global[0]=static_cast<size_t>((count+127)/128)*128;
    }
    // Each event spans a complete logical operator. Repeats are averaged over
    // event durations, excluding inter-launch idle time, never host timestamps.
    double sum=0;
    for(int i=0;i<repeats;++i) {
        cl_event event=nullptr;
        cl_int rc=clEnqueueNDRangeKernel(c.queue,kernel,ndim,nullptr,global,local,0,nullptr,&event);
        if(rc!=CL_SUCCESS) return fail(rc,"kernel enqueue");
        rc=clWaitForEvents(1,&event);
        if(rc!=CL_SUCCESS) { clReleaseEvent(event); return fail(rc,"kernel synchronization"); }
        cl_ulong start=0,end=0;
        cl_int a=clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start),&start,nullptr);
        cl_int b=clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(end),&end,nullptr);
        clReleaseEvent(event);
        if(a!=CL_SUCCESS || b!=CL_SUCCESS || end<start) return gc_error(GC_DEVICE_ERROR,"invalid OpenCL profiling event");
        sum+=static_cast<double>(end-start)/1000.0;
    }
    *device_us=sum/repeats;
    c.output_valid=true;
    return GC_OK;
}
int gc_opencl_download(void *opaque,gc_tensor *y) {
    if(!opaque || !y) return gc_error(GC_INVALID,"invalid download arguments");
    auto &c=*static_cast<Context *>(opaque);
    if(!c.ready || !c.output_valid) return gc_error(GC_INVALID,"successful OpenCL run required before download");
    const auto &p=c.shape;
    int64_t dims[]={p.n,p.cout,p.h,p.w};
    int64_t stride=1;
    if(y->device!=GC_CPU || y->device_index!=0 || y->dtype!=GC_F32 || y->rank!=4 || !y->data || reinterpret_cast<uintptr_t>(y->data)%4)
        return gc_error(GC_INVALID,"download requires aligned CPU FP32 output");
    for(int i=3;i>=0;--i) {
        if(y->dims[i]!=dims[i]) return gc_error(GC_INVALID,"download output dimensions mismatch");
        if(y->strides[i]!=stride) return gc_error(GC_UNSUPPORTED,"download requires contiguous NCHW");
        stride*=dims[i];
    }
    uint64_t bytes=static_cast<uint64_t>(stride)*4;
    if(y->bytes<bytes) return gc_error(GC_INVALID,"download output buffer too small");
    if(y->bytes>UINTPTR_MAX-reinterpret_cast<uintptr_t>(y->data)) return gc_error(GC_INVALID,"download pointer range overflow");
    cl_int rc=clEnqueueReadBuffer(c.queue,c.buffers[2],CL_TRUE,0,bytes,y->data,0,nullptr,nullptr);
    return rc==CL_SUCCESS?GC_OK:fail(rc,"device to host copy");
}
