#ifndef GROUPCONV_ATLAS_H
#define GROUPCONV_ATLAS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* ABI v1. Borrowed buffers; element strides. No function takes ownership. */
#define GC_ABI_VERSION 1
typedef enum { GC_OK=0, GC_INVALID=1, GC_UNSUPPORTED=2, GC_DEVICE_ERROR=3,
               GC_UNAVAILABLE=4 } gc_status;
typedef enum { GC_F32=0, GC_F64=1 } gc_dtype;
typedef enum { GC_CPU=0, GC_CUDA=1 } gc_device;
typedef enum { GC_SERIAL=0, GC_OPENMP=1, GC_BLOCKED=2 } gc_cpu_impl;
typedef struct {
    int64_t n, cin, cout, h, w, r, s, groups, sh, sw, ph, pw, dh, dw;
} gc_shape;
typedef struct {
    void *data;
    uint64_t bytes;
    int32_t dtype, device, device_index, rank;
    int64_t dims[4], strides[4];
} gc_tensor;

int gc_abi_version(void);
const char *gc_last_error(void);
const char *gc_build_info(void);
int gc_has_openmp(void);
int gc_validate_shape(const gc_shape *p, int64_t *ho, int64_t *wo);
int gc_validate_tensors(const gc_shape *p, const gc_tensor *x,
                        const gc_tensor *w, const gc_tensor *y,
                        int device, int dtype);
int gc_cpu_conv(const gc_shape *p, const gc_tensor *x, const gc_tensor *w,
                gc_tensor *y, int implementation, int threads);

/* CUDA buffers and stream belong to the caller. Launch is asynchronous;
   caller checks asynchronous errors at synchronization. No allocation. */
int gc_cuda_available(void);
int gc_cuda_conv(const gc_shape *p, const gc_tensor *x, const gc_tensor *w,
                 gc_tensor *y, int implementation, void *stream);

/* OpenCL owns a context, in-order profiling queue, and reusable device buffers.
   upload/download accept host descriptors; run synchronizes its queue. */
int gc_opencl_create(void **context);
const char *gc_opencl_info(void *context);
int gc_opencl_upload(void *context, const gc_shape *p, const gc_tensor *x,
                     const gc_tensor *w, const gc_tensor *y);
int gc_opencl_run(void *context, int implementation, int repeats,
                  double *device_op_us);
int gc_opencl_download(void *context, gc_tensor *y);
void gc_opencl_destroy(void *context);

#ifdef __cplusplus
}
/* Internal shared error helper; returns the supplied status. */
int gc_error(int status, const char *message);
#endif
#endif
