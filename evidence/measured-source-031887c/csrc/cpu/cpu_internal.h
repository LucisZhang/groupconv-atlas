#ifndef GC_CPU_INTERNAL_H
#define GC_CPU_INTERNAL_H
#include "groupconv.h"
namespace gc_cpu_detail {
constexpr int64_t tile_size = 32;
// FP32 scalar reference: one complete reduction per output, no shared state.
inline float point(const gc_shape &p, const float *x, const float *w,
            int64_t n, int64_t co, int64_t oh, int64_t ow) {
    const int64_t cpg=p.cin/p.groups, group=co/(p.cout/p.groups);
    float sum=0;
    for (int64_t ci=0; ci<cpg; ++ci)
        for (int64_t r=0; r<p.r; ++r) {
            const int64_t ih=oh*p.sh-p.ph+r*p.dh;
            if (ih<0 || ih>=p.h) continue;
            for (int64_t s=0; s<p.s; ++s) {
                const int64_t iw=ow*p.sw-p.pw+s*p.dw;
                if (iw>=0 && iw<p.w)
                    sum += x[((n*p.cin+group*cpg+ci)*p.h+ih)*p.w+iw] * w[((co*cpg+ci)*p.r+r)*p.s+s];
            }
        }
    return sum;
}
void oracle(const gc_shape &,const double *,const double *,double *,int64_t,int64_t);
void serial(const gc_shape *,const gc_tensor *,const gc_tensor *,gc_tensor *,int64_t,int64_t);
void parallel(const gc_shape *,const gc_tensor *,const gc_tensor *,gc_tensor *,int64_t,int64_t,int,int);
}
#endif
