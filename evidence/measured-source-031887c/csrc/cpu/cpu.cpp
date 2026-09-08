#include "cpu_internal.h"
using namespace gc_cpu_detail;
extern "C" int gc_cpu_conv(const gc_shape *p,const gc_tensor *x,const gc_tensor *w,gc_tensor *y,int implementation,int threads) {
    if (implementation<GC_SERIAL || implementation>GC_BLOCKED || threads<=0)
        return gc_error(GC_INVALID,"invalid CPU implementation or thread count");
    if (!x) return gc_error(GC_INVALID,"null input tensor descriptor");
    int status=gc_validate_tensors(p,x,w,y,GC_CPU,x->dtype);
    if (status) return status;
    if (x->dtype==GC_F64 && implementation!=GC_SERIAL)
        return gc_error(GC_UNSUPPORTED,"FP64 oracle supports GC_SERIAL only");
#ifndef _OPENMP
    if (implementation!=GC_SERIAL) return gc_error(GC_UNAVAILABLE,"OpenMP was not enabled in this build");
#endif
    int64_t ho,wo;
    gc_validate_shape(p,&ho,&wo);
    if (x->dtype==GC_F64) {
        oracle(*p,static_cast<const double *>(x->data),static_cast<const double *>(w->data),static_cast<double *>(y->data),ho,wo);
    } else if (implementation==GC_SERIAL) {
        serial(p,x,w,y,ho,wo);
    } else {
        parallel(p,x,w,y,ho,wo,implementation,threads);
    }
    return gc_error(GC_OK,"");
}
