#include "../csrc/common/groupconv.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include <thread>
#include <string>

void require(bool pass,const char *message) {
    if (!pass) { std::cerr << message << ": " << gc_last_error() << '\n'; std::exit(1); }
}
template<class T> gc_tensor desc(std::vector<T> &a,int64_t n,int64_t c,int64_t h,int64_t w) {
    gc_tensor t={}; t.data=a.data(); t.bytes=a.size()*sizeof(T); t.dtype=sizeof(T)==4?GC_F32:GC_F64;
    t.device=GC_CPU; t.rank=4;
    t.dims[0]=n;t.dims[1]=c;t.dims[2]=h;t.dims[3]=w;
    t.strides[3]=1;t.strides[2]=w;t.strides[1]=h*w;t.strides[0]=c*h*w; return t;
}
void compare(gc_shape p) {
    int64_t ho,wo; require(gc_validate_shape(&p,&ho,&wo)==GC_OK,"shape");
    std::vector<float> x(p.n*p.cin*p.h*p.w),w(p.cout*(p.cin/p.groups)*p.r*p.s),y(p.n*p.cout*ho*wo);
    for (size_t i=0;i<x.size();++i) x[i]=float(int(i*17%29)-14)/16;
    for (size_t i=0;i<w.size();++i) w[i]=float(int(i*11%23)-11)/16;
    std::vector<double> xd(x.begin(),x.end()),wd(w.begin(),w.end()),yd(y.size());
    auto a=desc(x,p.n,p.cin,p.h,p.w),b=desc(w,p.cout,p.cin/p.groups,p.r,p.s),c=desc(y,p.n,p.cout,ho,wo);
    auto ad=desc(xd,p.n,p.cin,p.h,p.w),bd=desc(wd,p.cout,p.cin/p.groups,p.r,p.s),cd=desc(yd,p.n,p.cout,ho,wo);
    require(gc_cpu_conv(&p,&ad,&bd,&cd,GC_SERIAL,1)==GC_OK,"oracle");
    for (int impl=0;impl<3;++impl) for (int threads: {1,2,4}) {
        const int status=gc_cpu_conv(&p,&a,&b,&c,impl,threads);
        if (impl && !gc_has_openmp()) { require(status==GC_UNAVAILABLE,"OpenMP unavailable status"); continue; }
        require(status==GC_OK,"CPU implementation");
        for(size_t i=0;i<y.size();++i) require(std::isfinite(y[i]) && std::abs(y[i]-yd[i])<=1e-4+1e-4*std::abs(yd[i]),"FP64 comparison");
    }
}
int main() {
    require(gc_abi_version()==1,"ABI");
    gc_shape p={1,2,2,2,3,1,1,2,1,1,0,0,1,1};
    std::vector<float> x={1,2,3,4,5,6,10,20,30,40,50,60},w={2,3},y(12);
    auto a=desc(x,1,2,2,3),b=desc(w,2,1,1,1),c=desc(y,1,2,2,3);
    require(gc_cpu_conv(&p,&a,&b,&c,0,1)==GC_OK,"manual pointwise");
    for(int i=0;i<12;++i) require(y[i]==x[i]*(i<6?2:3),"manual result");
    for(int i=6;i<12;++i) x[i]=0;
    require(gc_cpu_conv(&p,&a,&b,&c,0,1)==GC_OK,"group isolation");
    for(int i=6;i<12;++i) require(y[i]==0,"group leaked");
    // Hand-computed 2x2 valid cross-correlation, distinct from pointwise test.
    gc_shape q={1,1,1,3,3,2,2,1,1,1,0,0,1,1};
    std::vector<float> xx={1,2,3,4,5,6,7,8,9},ww={1,2,3,4},yy(4);
    auto aa=desc(xx,1,1,3,3),bb=desc(ww,1,1,2,2),cc=desc(yy,1,1,2,2);
    require(gc_cpu_conv(&q,&aa,&bb,&cc,0,1)==GC_OK,"manual spatial");
    require(yy==std::vector<float>({37,47,67,77}),"manual spatial values");
    compare({2,6,10,5,37,3,3,2,1,1,1,1,1,1});
    compare({1,3,6,9,11,3,5,3,2,2,1,2,2,1});
    compare({1,4,8,1,1,3,3,4,1,1,1,1,1,1});
    compare({1,2,3,9,11,7,5,1,1,2,0,0,1,1});
    auto bad=a; bad.strides[2]++; require(gc_cpu_conv(&p,&bad,&b,&c,0,1)==GC_UNSUPPORTED,"layout");
    bad=a;bad.bytes--; require(gc_cpu_conv(&p,&bad,&b,&c,0,1)==GC_INVALID,"small buffer");
    bad=a;bad.rank=3;require(gc_cpu_conv(&p,&bad,&b,&c,0,1)==GC_INVALID,"rank");
    bad=a;bad.device=GC_CUDA;require(gc_cpu_conv(&p,&bad,&b,&c,0,1)==GC_INVALID,"device");
    bad=a;bad.dtype=999;require(gc_cpu_conv(&p,&bad,&b,&c,0,1)==GC_INVALID,"dtype");
    bad=a;bad.data=reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(a.data)+1);require(gc_cpu_conv(&p,&bad,&b,&c,0,1)==GC_INVALID,"alignment");
    bad=a;bad.data=reinterpret_cast<void *>(UINTPTR_MAX-3);require(gc_cpu_conv(&p,&bad,&b,&c,0,1)==GC_INVALID,"pointer range");
    bad=c;bad.data=a.data;require(gc_cpu_conv(&p,&a,&b,&bad,0,1)==GC_INVALID,"alias");
    require(gc_cpu_conv(nullptr,&a,&b,&c,0,1)==GC_INVALID,"null shape");
    require(gc_cpu_conv(&p,nullptr,&b,&c,0,1)==GC_INVALID,"null input");
    require(gc_cpu_conv(&p,&a,nullptr,&c,0,1)==GC_INVALID,"null weights");
    require(gc_cpu_conv(&p,&a,&b,nullptr,0,1)==GC_INVALID,"null output");
    require(gc_cpu_conv(&p,&a,&b,&c,0,0)==GC_INVALID,"threads");
    require(gc_cpu_conv(&p,&a,&b,&c,3,1)==GC_INVALID,"implementation");
    int64_t ho,wo;
    q=p;q.groups=3;require(gc_validate_shape(&q,&ho,&wo)==GC_INVALID,"divisibility");
    q=p;q.ph=INT64_MAX;require(gc_validate_shape(&q,&ho,&wo)==GC_INVALID,"padding overflow");
    q=p;q.n=INT64_MAX;require(gc_validate_shape(&q,&ho,&wo)==GC_INVALID,"element count overflow");
    q=p;q.r=INT64_MAX;q.dh=2;require(gc_validate_shape(&q,&ho,&wo)==GC_INVALID,"effective kernel overflow");
    q=p;q.r=9;require(gc_validate_shape(&q,&ho,&wo)==GC_INVALID,"negative output");
    require(gc_validate_shape(&p,nullptr,&wo)==GC_INVALID,"null extent");
    const std::string main_error=gc_last_error();
    std::thread other([] { gc_error(GC_INVALID,"other thread"); require(std::string(gc_last_error())=="other thread","worker error"); }); other.join();
    require(std::string(gc_last_error())==main_error,"thread local errors");
    std::cout << "native CPU tests passed; OpenMP=" << gc_has_openmp() << '\n';
}
