#include "../common/groupconv.h"

extern "C" int gc_cuda_available(void) { return 0; }
extern "C" int gc_cuda_conv(const gc_shape *, const gc_tensor *,
                             const gc_tensor *, gc_tensor *, int, void *) {
    return gc_error(GC_UNAVAILABLE, "CUDA backend was not built");
}
