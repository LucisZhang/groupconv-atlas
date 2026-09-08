#include "groupconv.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace {
thread_local std::string last_error;
constexpr int64_t limit = INT64_MAX;
bool mul(int64_t a, int64_t b, int64_t &v) {
    if (a < 0 || b < 0 || (a && b > limit / a)) return false;
    v = a * b; return true;
}
bool extent(int64_t input, int64_t kernel, int64_t pad, int64_t dilation,
            int64_t stride, int64_t &out) {
    int64_t halo, effective;
    if (!mul(pad, 2, halo) || input > limit - halo ||
        !mul(kernel - 1, dilation, effective) || effective == limit) return false;
    ++effective;
    const int64_t padded = input + halo;
    if (padded < effective) return false;
    out = (padded - effective) / stride + 1;
    // Also bound the largest intermediate used by direct-kernel coordinates.
    int64_t base;
    return mul(out - 1, stride, base) && base <= limit - (effective - 1);
}
int tensor(const gc_tensor *t, const int64_t dims[4], int device, int dtype) {
    if (!t || !t->data) return gc_error(GC_INVALID, "null tensor descriptor or buffer");
    if (t->rank != 4) return gc_error(GC_INVALID, "tensor rank must be four");
    if (t->device != device || t->device_index < 0 || (device == GC_CPU && t->device_index != 0))
        return gc_error(GC_INVALID, "tensor device or device index mismatch");
    if (t->dtype != dtype) return gc_error(GC_INVALID, "tensor dtype mismatch");
    int64_t count = 1;
    bool dense = true;
    for (int i = 3; i >= 0; --i) {
        if (t->dims[i] != dims[i]) return gc_error(GC_INVALID, "tensor dimensions do not match convolution");
        if (t->strides[i] != count) dense = false;
        if (!mul(count, dims[i], count)) return gc_error(GC_INVALID, "tensor element count overflow");
    }
    const uint64_t element = dtype == GC_F32 ? sizeof(float) : sizeof(double);
    if (static_cast<uint64_t>(count) > static_cast<uint64_t>(PTRDIFF_MAX) / element)
        return gc_error(GC_INVALID, "tensor byte count exceeds pointer indexing range");
    const uint64_t required = static_cast<uint64_t>(count) * element;
    const uintptr_t address = reinterpret_cast<uintptr_t>(t->data);
    if (address % element) return gc_error(GC_INVALID, "tensor data is not element aligned");
    if (t->bytes < required) return gc_error(GC_INVALID, "tensor buffer is too small");
    if (t->bytes > UINTPTR_MAX - address) return gc_error(GC_INVALID, "tensor pointer range overflow");
    if (!dense) return gc_error(GC_UNSUPPORTED, "only dense contiguous NCHW tensors are supported");
    return GC_OK;
}
bool overlaps(const gc_tensor *a, const gc_tensor *b) {
    const uintptr_t ab = reinterpret_cast<uintptr_t>(a->data), bb = reinterpret_cast<uintptr_t>(b->data);
    return ab < bb + b->bytes && bb < ab + a->bytes;
}
}
int gc_error(int status, const char *message) { last_error = message ? message : ""; return status; }
extern "C" {
int gc_abi_version(void) { return GC_ABI_VERSION; }
const char *gc_last_error(void) { return last_error.c_str(); }
int gc_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
const char *gc_build_info(void) {
#ifdef _OPENMP
    return "groupconv-atlas ABI=1; CPU=f32,f64; layout=NCHW; OpenMP=enabled; c1_tile=32; c2_tile=32; FP-contract=off (build flag required)";
#else
    return "groupconv-atlas ABI=1; CPU=f32,f64; layout=NCHW; OpenMP=disabled; c1_tile=32; c2_tile=32; FP-contract=off (build flag required)";
#endif
}
int gc_validate_shape(const gc_shape *p, int64_t *ho, int64_t *wo) {
    if (!p || !ho || !wo) return gc_error(GC_INVALID, "null shape or output extent pointer");
    if (p->n <= 0 || p->cin <= 0 || p->cout <= 0 || p->h <= 0 || p->w <= 0 ||
        p->r <= 0 || p->s <= 0 || p->groups <= 0 || p->sh <= 0 || p->sw <= 0 ||
        p->dh <= 0 || p->dw <= 0 || p->ph < 0 || p->pw < 0)
        return gc_error(GC_INVALID, "dimensions, groups, strides and dilation must be positive; padding nonnegative");
    if (p->cin % p->groups || p->cout % p->groups)
        return gc_error(GC_INVALID, "input and output channels must be divisible by groups");
    int64_t oh, ow;
    if (!extent(p->h, p->r, p->ph, p->dh, p->sh, oh) ||
        !extent(p->w, p->s, p->pw, p->dw, p->sw, ow))
        return gc_error(GC_INVALID, "invalid output geometry or signed index overflow");
    const int64_t dims[3][4] = {{p->n,p->cin,p->h,p->w}, {p->cout,p->cin/p->groups,p->r,p->s}, {p->n,p->cout,oh,ow}};
    for (const auto &d : dims) {
        int64_t count = 1;
        for (int i = 0; i < 4; ++i)
            if (!mul(count, d[i], count)) return gc_error(GC_INVALID, "shape element count overflow");
    }
    *ho = oh; *wo = ow;
    return gc_error(GC_OK, "");
}
int gc_validate_tensors(const gc_shape *p, const gc_tensor *x, const gc_tensor *w,
                       const gc_tensor *y, int device, int dtype) {
    int64_t ho, wo;
    int status = gc_validate_shape(p, &ho, &wo);
    if (status) return status;
    if ((device != GC_CPU && device != GC_CUDA) || (dtype != GC_F32 && dtype != GC_F64))
        return gc_error(GC_INVALID, "unknown expected device or dtype");
    const int64_t xd[4] = {p->n,p->cin,p->h,p->w}, wd[4] = {p->cout,p->cin/p->groups,p->r,p->s}, yd[4] = {p->n,p->cout,ho,wo};
    if ((status = tensor(x,xd,device,dtype)) || (status = tensor(w,wd,device,dtype)) ||
        (status = tensor(y,yd,device,dtype))) return status;
    if (x->device_index != w->device_index || x->device_index != y->device_index)
        return gc_error(GC_INVALID, "tensor device indices differ");
    if (overlaps(y,x) || overlaps(y,w)) return gc_error(GC_INVALID, "output must not alias input or weights");
    return gc_error(GC_OK, "");
}
}
