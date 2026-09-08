#include "cpu_internal.h"
#include <algorithm>
namespace gc_cpu_detail {
namespace {
void tile(const gc_shape &p,const float *x,const float *w,float *y,int64_t ho,int64_t wo,int64_t task,bool blocked) {
    const int64_t tiles=(wo-1)/tile_size+1;
    const int64_t begin=(task%tiles)*tile_size;
    task/=tiles;
    const int64_t oh=task%ho; task/=ho;
    const int64_t co=task%p.cout, n=task/p.cout;
    const int64_t length=std::min(tile_size,wo-begin), base=((n*p.cout+co)*ho+oh)*wo+begin;
    if (!blocked) {
        for (int64_t j=0;j<length;++j) y[base+j]=point(p,x,w,n,co,oh,begin+j);
        return;
    }
    // Same task schedule as c1. Only loop order changes: reuse each weight
    // across a fixed spatial block, keeping accumulators in a small local array.
    float sums[tile_size]={};
    const int64_t cpg=p.cin/p.groups, group=co/(p.cout/p.groups);
    for (int64_t ci=0;ci<cpg;++ci) for (int64_t r=0;r<p.r;++r) {
        const int64_t ih=oh*p.sh-p.ph+r*p.dh;
        if (ih<0 || ih>=p.h) continue;
        for (int64_t s=0;s<p.s;++s) {
            const float weight=w[((co*cpg+ci)*p.r+r)*p.s+s];
            for (int64_t j=0;j<length;++j) {
                const int64_t iw=(begin+j)*p.sw-p.pw+s*p.dw;
                if (iw>=0 && iw<p.w) sums[j]+=x[((n*p.cin+group*cpg+ci)*p.h+ih)*p.w+iw]*weight;
            }
        }
    }
    for (int64_t j=0;j<length;++j) y[base+j]=sums[j];
}
}
void parallel(const gc_shape *p,const gc_tensor *x,const gc_tensor *w,gc_tensor *y,int64_t ho,int64_t wo,int implementation,int threads) {
    const int64_t tasks=p->n*p->cout*ho*((wo-1)/tile_size+1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#else
    (void)threads;
#endif
    for (int64_t t=0;t<tasks;++t)
        tile(*p,static_cast<const float *>(x->data),static_cast<const float *>(w->data),static_cast<float *>(y->data),ho,wo,t,implementation==GC_BLOCKED);
}
}
