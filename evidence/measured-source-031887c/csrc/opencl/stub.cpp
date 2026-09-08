#include "groupconv.h"
int gc_opencl_create(void **p) { if(p) *p=nullptr; return gc_error(GC_UNAVAILABLE,"OpenCL was disabled at build time"); }
const char *gc_opencl_info(void *) { return "OpenCL disabled"; }
int gc_opencl_upload(void *,const gc_shape *,const gc_tensor *,const gc_tensor *,const gc_tensor *) { return gc_error(GC_UNAVAILABLE,"OpenCL disabled"); }
int gc_opencl_run(void *,int,int,double *) { return gc_error(GC_UNAVAILABLE,"OpenCL disabled"); }
int gc_opencl_download(void *,gc_tensor *) { return gc_error(GC_UNAVAILABLE,"OpenCL disabled"); }
void gc_opencl_destroy(void *) {}
