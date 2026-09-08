// Native, PyTorch-independent CUDA correctness/contract/sanitizer runner.
#include "groupconv.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    float* p = nullptr;
    explicit Buffer(size_t n) { ck(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(float))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t s = nullptr;
    Stream() { ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { if (s) { cudaStreamSynchronize(s); cudaStreamDestroy(s); } }
};
gc_tensor desc(float* data, int64_t a, int64_t b, int64_t c, int64_t d) {
    int dev; ck(cudaGetDevice(&dev));
    gc_tensor t{}; t.data=data; t.bytes=uint64_t(a*b*c*d)*sizeof(float);
    t.dtype=GC_F32; t.device=GC_CUDA; t.device_index=dev; t.rank=4;
    t.dims[0]=a; t.dims[1]=b; t.dims[2]=c; t.dims[3]=d;
    int64_t stride=1;
    for (int k=3;k>=0;--k) { t.strides[k]=stride; stride*=t.dims[k]; }
    return t;
}
struct Fixture {
    gc_shape p;
    int64_t ho=0, wo=0;
    std::vector<float> x,w,y;
    Buffer dx,dw,dy;
    gc_tensor xt,wt,yt;
    // Stream is destroyed first, before buffers and copy source/destination vectors.
    Stream stream;
    static int64_t output(const gc_shape& p, bool height) {
        int64_t h,w;
        if (gc_validate_shape(&p,&h,&w)!=GC_OK) throw std::runtime_error(gc_last_error());
        return height?h:w;
    }
    explicit Fixture(gc_shape q): p(q),ho(output(q,true)),wo(output(q,false)),
        x(p.n*p.cin*p.h*p.w),w(p.cout*(p.cin/p.groups)*p.r*p.s),y(p.n*p.cout*ho*wo),
        dx(x.size()),dw(w.size()),dy(y.size()),
        xt(desc(dx.p,p.n,p.cin,p.h,p.w)),wt(desc(dw.p,p.cout,p.cin/p.groups,p.r,p.s)),
        yt(desc(dy.p,p.n,p.cout,ho,wo)) {}
    void upload() {
        ck(cudaMemcpyAsync(dx.p,x.data(),xt.bytes,cudaMemcpyHostToDevice,stream.s));
        ck(cudaMemcpyAsync(dw.p,w.data(),wt.bytes,cudaMemcpyHostToDevice,stream.s));
        // NaN sentinel detects missed outputs, including every grid-stride tail.
        ck(cudaMemsetAsync(dy.p,0xff,yt.bytes,stream.s));
    }
    int launch(int impl) { return gc_cuda_conv(&p,&xt,&wt,&yt,impl,reinterpret_cast<void*>(stream.s)); }
    void download() {
        ck(cudaMemcpyAsync(y.data(),dy.p,yt.bytes,cudaMemcpyDeviceToHost,stream.s));
        ck(cudaStreamSynchronize(stream.s));
    }
};
int passed=0, unsupported=0, failed=0;
void result(const std::string& name, bool ok, const std::string& detail="") {
    std::cout << (ok?"PASS ":"FAIL ") << name << " " << detail << std::endl;
    if(ok) ++passed; else ++failed;
}
void fill(Fixture& f, const std::string& pattern, uint32_t seed) {
    auto rnd=[&]() { seed=1664525u*seed+1013904223u; return float(int((seed>>16)%33)-16)/16.0f; };
    for(size_t i=0;i<f.x.size();++i) f.x[i]=rnd();
    for(float& v:f.w) v=rnd();
    if(pattern=="zero") std::fill(f.x.begin(),f.x.end(),0.0f);
    if(pattern=="impulse" || pattern=="isolation") {
        std::fill(f.x.begin(),f.x.end(),0.0f);
        // Only the first group receives a signal; all other groups must stay zero.
        f.x[(f.p.h/2)*f.p.w+f.p.w/2]=1.0f;
        if(pattern=="isolation") std::fill(f.w.begin(),f.w.end(),1.0f);
    }
    if(pattern=="cancellation") {
        for(size_t i=0;i<f.x.size();++i) f.x[i]=(i%2? -1.0f:1.0f);
        std::fill(f.w.begin(),f.w.end(),0.125f);
    }
    if(pattern=="stress") {
        for(size_t i=0;i<f.x.size();++i) f.x[i]=float(int((i/(f.p.h*f.p.w))%7)-3)*0.25f;
        for(size_t i=0;i<f.w.size();++i) f.w[i]=float(int(i%5)-2)*0.125f;
    }
}
bool oracle(const Fixture& f, std::string& detail) {
    const auto& p=f.p; const int64_t cpg=p.cin/p.groups, opg=p.cout/p.groups;
    double maxabs=0,maxrel=0,square=0; size_t bad=0;
    for(int64_t n=0;n<p.n;++n) for(int64_t co=0;co<p.cout;++co)
    for(int64_t oh=0;oh<f.ho;++oh) for(int64_t ow=0;ow<f.wo;++ow) {
        double ref=0;
        for(int64_t ci=0;ci<cpg;++ci) for(int64_t r=0;r<p.r;++r) for(int64_t s=0;s<p.s;++s) {
            int64_t ih=oh*p.sh-p.ph+r*p.dh, iw=ow*p.sw-p.pw+s*p.dw;
            if(ih>=0 && ih<p.h && iw>=0 && iw<p.w) {
                size_t xi=((n*p.cin+(co/opg)*cpg+ci)*p.h+ih)*p.w+iw;
                size_t wi=((co*cpg+ci)*p.r+r)*p.s+s;
                ref+=double(f.x[xi])*double(f.w[wi]);
            }
        }
        double got=f.y[((n*p.cout+co)*f.ho+oh)*f.wo+ow];
        double err=std::abs(got-ref);
        if(!std::isfinite(got) || err>1e-4+1e-4*std::abs(ref)) ++bad;
        if(std::isfinite(err)) { maxabs=std::max(maxabs,err); maxrel=std::max(maxrel,err/std::max(std::abs(ref),1e-12)); square+=err*err; }
    }
    detail="outputs="+std::to_string(f.y.size())+" bad="+std::to_string(bad)+" max_abs="+std::to_string(maxabs)+" max_rel="+std::to_string(maxrel)+" rms="+std::to_string(std::sqrt(square/f.y.size()));
    return bad==0;
}
void numerical(const std::string& name, gc_shape p, int impl, const std::string& pattern="random", uint32_t seed=1) {
    Fixture f(p); fill(f,pattern,seed); f.upload();
    int status=f.launch(impl);
    if(status!=GC_OK) { ck(cudaStreamSynchronize(f.stream.s)); result(name,false,"status="+std::to_string(status)+" "+gc_last_error()); return; }
    f.download(); std::string detail; bool ok=oracle(f,detail); result(name,ok,detail);
}
void graph_numerical(gc_shape p, int impl) {
    Fixture f(p); fill(f,"random",73); f.upload();
    // Complete allocation/copies before capture: capture contains the ABI launch.
    ck(cudaStreamSynchronize(f.stream.s));
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    ck(cudaStreamBeginCapture(f.stream.s,cudaStreamCaptureModeThreadLocal));
    int status=f.launch(impl);
    std::string launch_error=gc_last_error();
    // Always end capture, even when the ABI call failed, to restore stream state.
    cudaError_t end_status=cudaStreamEndCapture(f.stream.s,&graph);
    if(status!=GC_OK) {
        if(graph) cudaGraphDestroy(graph);
        result("graph_k"+std::to_string(impl),false,"status="+std::to_string(status)+" "+launch_error);
        cudaGetLastError();
        return;
    }
    ck(end_status);
    try {
        ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        ck(cudaGraphLaunch(executable,f.stream.s));
        f.download();
        std::string detail; bool ok=oracle(f,detail);
        result("graph_k"+std::to_string(impl),ok,detail);
    } catch(...) {
        cudaStreamSynchronize(f.stream.s);
        if(executable) cudaGraphExecDestroy(executable);
        cudaGraphDestroy(graph);
        throw;
    }
    ck(cudaGraphExecDestroy(executable));
    ck(cudaGraphDestroy(graph));
}
gc_shape bshape(int64_t n,int64_t h,int64_t w,int64_t cpg,int64_t groups=2) {
    return {n,cpg*groups,cpg*groups,h,w,3,3,groups,1,1,1,1,1,1};
}
void expect(const std::string& name, Fixture& f, int impl, int expected) {
    int got=f.launch(impl); std::string error=gc_last_error();
    // Invalid metadata calls must not leave work pending; pointer API errors can
    // set the thread's last-error slot, so consume it before the next case.
    cudaGetLastError(); ck(cudaStreamSynchronize(f.stream.s));
    if(got==GC_UNSUPPORTED && got==expected) {
        ++unsupported; std::cout<<"UNSUPPORTED "<<name<<" expected=true "<<error<<std::endl;
    } else result(name,got==expected,"expected="+std::to_string(expected)+" actual="+std::to_string(got)+" "+error);
}
void contracts(int only) {
    Fixture f(bshape(1,7,13,2));
    auto saved=f.xt;
    f.xt.dtype=GC_F64; expect("bad_dtype",f,0,GC_INVALID); f.xt=saved;
    f.xt.rank=3; expect("bad_rank",f,0,GC_INVALID); f.xt=saved;
    f.xt.strides[3]=2; expect("bad_strides",f,0,GC_UNSUPPORTED); f.xt=saved;
    f.xt.device=GC_CPU; expect("bad_device_kind",f,0,GC_INVALID); f.xt=saved;
    f.xt.device_index=-1; expect("bad_device_index",f,0,GC_INVALID); f.xt=saved;
    f.xt.device_index+=1; expect("different_device_indices",f,0,GC_INVALID); f.xt=saved;
    ++f.xt.device_index; ++f.wt.device_index; ++f.yt.device_index;
    expect("pointer_device_mismatch",f,0,GC_INVALID);
    --f.xt.device_index; --f.wt.device_index; --f.yt.device_index;
    f.xt.bytes-=sizeof(float); expect("undersized_declared_buffer",f,0,GC_INVALID); f.xt=saved;
    auto ys=f.yt; f.yt.data=f.xt.data; expect("output_alias_input",f,0,GC_INVALID); f.yt=ys;
    expect("bad_impl_negative",f,-1,GC_UNSUPPORTED); expect("bad_impl_high",f,4,GC_UNSUPPORTED);
    // Pinned host pointer has a portable successful attributes query and is
    // rejected as GC_INVALID, unlike older runtimes' unregistered host queries.
    float* host=nullptr; ck(cudaMallocHost(reinterpret_cast<void**>(&host),f.xt.bytes));
    f.xt.data=host; expect("pinned_host_pointer",f,0,GC_INVALID); f.xt=saved; ck(cudaFreeHost(host));
    f.xt.data=f.x.data();
    int got=f.launch(0); std::string error=gc_last_error(); cudaGetLastError();
    result("pageable_host_pointer",got==GC_INVALID || got==GC_DEVICE_ERROR,"actual="+std::to_string(got)+" "+error);
    f.xt=saved; ck(cudaStreamSynchronize(f.stream.s));
    for(int impl=1;impl<=3;++impl) if(only<0 || only==impl) {
        auto q=bshape(1,7,13,2); q.sh=2; Fixture outside(q);
        expect("outside_B_k"+std::to_string(impl),outside,impl,GC_UNSUPPORTED);
    }
    if(only<0 || only==2) for(int cpg:{3,8}) {
        Fixture other(bshape(1,7,13,cpg)); expect("k2_cpg"+std::to_string(cpg),other,2,GC_UNSUPPORTED);
    }
}
}
int main(int argc,char** argv) {
    bool stress=false,profile=false,negative_only=false,graph_mode=false; int only=-1;
    try {
        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            if(a=="--stress") stress=true;
            else if(a=="--graph") graph_mode=true;
            else if(a=="--profile") profile=true;
            else if(a=="--contracts") negative_only=true;
            else if(a=="--only-impl" && i+1<argc) {
                std::string value=argv[++i];
                if(value.size()!=1 || value[0]<'0' || value[0]>'3') throw std::runtime_error("--only-impl requires 0..3");
                only=value[0]-'0';
            } else throw std::runtime_error("usage: native_cuda [--stress | --profile | --contracts | --graph] [--only-impl 0..3]");
        }
        if(int(stress)+int(profile)+int(negative_only)+int(graph_mode)>1) throw std::runtime_error("choose one run mode");
        if(stress && only>=0 && only!=2) throw std::runtime_error("--stress specifically tests k2");
        if(!gc_cuda_available()) { std::cerr<<"UNAVAILABLE CUDA device; no tests executed\n"; return 77; }
        int dev; ck(cudaGetDevice(&dev)); cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop,dev));
        int driver=0,runtime=0; ck(cudaDriverGetVersion(&driver)); ck(cudaRuntimeGetVersion(&runtime));
        std::cout<<"DEVICE "<<prop.name<<" cc="<<prop.major<<"."<<prop.minor<<" driver="<<driver<<" runtime="<<runtime<<" stream=nonblocking_nondefault\n";
        if(graph_mode) {
            for(int k=0;k<4;++k) if(only<0 || only==k) graph_numerical(bshape(1,7,13,2),k);
        } else if(negative_only) contracts(only);
        else if(stress) for(int cpg:{1,2,4}) {
            // 1x17 has spatial tails on both axes; enough planes force block
            // reuse of shared memory after the 65,535-block grid cap.
            auto p=bshape(65535/(cpg*2*2)+1,1,17,cpg);
            int64_t tiles=p.n*p.cout*2;
            if(tiles<=65535) throw std::runtime_error("stress tile count is insufficient");
            numerical("stress_k2_cpg"+std::to_string(cpg)+"_tiles"+std::to_string(tiles),p,2,"stress");
        } else if(profile) {
            for(int k=0;k<4;++k) if(only<0 || only==k) numerical("profile_k"+std::to_string(k),bshape(1,32,48,4),k);
        } else {
            const int hw[][2]={{1,1},{1,19},{11,1},{7,13},{9,17},{17,33},{8,16}};
            for(int k=0;k<4;++k) if(only<0 || only==k) for(int cpg:{1,2,4,3,8}) {
                if(k==2 && cpg!=1 && cpg!=2 && cpg!=4) continue;
                for(const auto& size:hw) for(int n:{1,4}) {
                    std::string name="k"+std::to_string(k)+"_cpg"+std::to_string(cpg)+"_N"+std::to_string(n)+"_"+std::to_string(size[0])+"x"+std::to_string(size[1]);
                    numerical(name,bshape(n,size[0],size[1],cpg),k,"random",uint32_t(n+size[0]*31+size[1]));
                }
                numerical("G1_k"+std::to_string(k)+"_cpg"+std::to_string(cpg),bshape(1,9,17,cpg,1),k);
                for(const std::string pattern:{"zero","impulse","isolation","cancellation"})
                    numerical(pattern+"_k"+std::to_string(k)+"_cpg"+std::to_string(cpg),bshape(1,7,13,cpg),k,pattern);
            }
            if(only<0 || only==0) {
                for(int kernel:{3,5,7}) for(int stride:{1,2}) for(int dilation:{1,2}) for(int pad:{0,3}) {
                    gc_shape p={1,4,6,17,23,kernel,kernel,2,stride,stride,pad,pad,dilation,dilation};
                    numerical("general_k0_r"+std::to_string(kernel)+"_s"+std::to_string(stride)+"_d"+std::to_string(dilation)+"_p"+std::to_string(pad),p,0);
                }
                numerical("asymmetric_geometry_k0",{1,4,6,17,23,3,5,2,2,1,1,3,1,2},0);
                numerical("depthwise_multiplier2_k0",{4,3,6,7,13,3,3,3,1,1,1,1,1,1},0);
            }
            // Pointer negative tests live in the separate --contracts process.
            for(int k=1;k<=3;++k) if(only<0 || only==k) {
                auto p=bshape(1,7,13,2); p.sh=2; Fixture f(p); expect("outside_B_k"+std::to_string(k),f,k,GC_UNSUPPORTED);
            }
            if(only<0 || only==2) for(int cpg:{3,8}) { Fixture f(bshape(1,7,13,cpg)); expect("k2_cpg"+std::to_string(cpg),f,2,GC_UNSUPPORTED); }
        }
        std::cout<<"SUMMARY pass="<<passed<<" expected_unsupported="<<unsupported<<" fail="<<failed<<std::endl;
        return failed?1:0;
    } catch(const std::exception& e) { std::cerr<<"FAIL exception "<<e.what()<<std::endl; return 1; }
}
