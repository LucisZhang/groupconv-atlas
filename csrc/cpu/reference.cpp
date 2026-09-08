#include "cpu_internal.h"
namespace gc_cpu_detail {
// Independent oracle implementation: explicit n/group/local-output loops and
// separate addressing, deliberately not instantiated from the FP32 kernel.
void oracle(const gc_shape &p, const double *x, const double *w, double *y, int64_t ho, int64_t wo) {
    const int64_t ic=p.cin/p.groups, oc=p.cout/p.groups;
    for (int64_t b=0;b<p.n;++b) for (int64_t g=0;g<p.groups;++g)
    for (int64_t o=0;o<oc;++o) for (int64_t row=0;row<ho;++row) for (int64_t col=0;col<wo;++col) {
        double value=0;
        for (int64_t c=0;c<ic;++c) for (int64_t kr=0;kr<p.r;++kr) for (int64_t ks=0;ks<p.s;++ks) {
            const int64_t ir=row*p.sh+kr*p.dh-p.ph, is=col*p.sw+ks*p.dw-p.pw;
            if (ir<0 || ir>=p.h || is<0 || is>=p.w) continue;
            const int64_t xi=b*(p.cin*p.h*p.w)+(g*ic+c)*(p.h*p.w)+ir*p.w+is;
            const int64_t wi=(g*oc+o)*(ic*p.r*p.s)+c*(p.r*p.s)+kr*p.s+ks;
            value += x[xi]*w[wi];
        }
        y[((b*p.cout+g*oc+o)*ho+row)*wo+col]=value;
    }
}
void serial(const gc_shape *p,const gc_tensor *x,const gc_tensor *w,gc_tensor *y,int64_t ho,int64_t wo) {
    auto out=static_cast<float *>(y->data);
    for (int64_t n=0;n<p->n;++n) for (int64_t co=0;co<p->cout;++co)
    for (int64_t oh=0;oh<ho;++oh) for (int64_t ow=0;ow<wo;++ow)
        out[((n*p->cout+co)*ho+oh)*wo+ow]=point(*p,static_cast<const float *>(x->data),static_cast<const float *>(w->data),n,co,oh,ow);
}
}
