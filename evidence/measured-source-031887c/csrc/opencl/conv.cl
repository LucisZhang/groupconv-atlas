// Authored for GroupConv Atlas. FP32, NCHW, B contract only.
#pragma OPENCL FP_CONTRACT OFF
__kernel void naive(__global const float *x, __global const float *w,
                    __global float *y, long N, long C, long H, long W, long G) {
    long idx = get_global_id(0), count = N*C*H*W;
    if (idx >= count) return;
    long ow=idx%W, oh=(idx/W)%H, co=(idx/(H*W))%C, n=idx/(C*H*W);
    long cpg=C/G, group=co/cpg;
    float acc=0.0f;
    for (long ci=0; ci<cpg; ++ci)
      for (int r=0; r<3; ++r)
        for (int s=0; s<3; ++s) {
          long ih=oh+r-1, iw=ow+s-1;
          if (ih>=0 && ih<H && iw>=0 && iw<W)
            acc += x[((n*C+group*cpg+ci)*H+ih)*W+iw]*w[(co*cpg+ci)*9+r*3+s];
        }
    y[idx]=acc;
}

// One 16x8 work-group computes a spatial tile for one output channel.
// Every work-item participates in both barriers, including output tail lanes.
__kernel void shared_tile(__global const float *x, __global const float *w,
                          __global float *y, long N, long C, long H, long W, long G) {
    __local float tile[10*18];
    int lx=get_local_id(0), ly=get_local_id(1), tid=ly*16+lx;
    long bx=get_group_id(0)*16, by=get_group_id(1)*8, nc=get_group_id(2);
    long n=nc/C, co=nc%C, cpg=C/G, group=co/cpg;
    long oh=by+ly, ow=bx+lx;
    float acc=0.0f;
    for (long ci=0; ci<cpg; ++ci) {
      for (int i=tid; i<180; i+=128) {
        long ih=by+i/18-1, iw=bx+i%18-1;
        tile[i]=(ih>=0 && ih<H && iw>=0 && iw<W)
          ? x[((n*C+group*cpg+ci)*H+ih)*W+iw] : 0.0f;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int r=0; r<3; ++r)
        for (int s=0; s<3; ++s)
          acc += tile[(ly+r)*18+lx+s]*w[(co*cpg+ci)*9+r*3+s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (oh<H && ow<W) y[((n*C+co)*H+oh)*W+ow]=acc;
}

// Four adjacent outputs reuse six input values and a row of weights.
__kernel void register_tile(__global const float *x, __global const float *w,
                            __global float *y, long N, long C, long H, long W, long G) {
    long tiles=(W+3)/4, idx=get_global_id(0);
    if (idx>=N*C*H*tiles) return;
    long ow=(idx%tiles)*4, oh=(idx/tiles)%H;
    long co=(idx/(tiles*H))%C, n=idx/(tiles*H*C), cpg=C/G, group=co/cpg;
    float acc[4]={0.0f,0.0f,0.0f,0.0f};
    for (long ci=0; ci<cpg; ++ci)
      for (int r=0; r<3; ++r) {
        long ih=oh+r-1;
        float v[6];
        for (int t=0; t<6; ++t) {
          long iw=ow+t-1;
          v[t]=(ih>=0 && ih<H && iw>=0 && iw<W)
            ? x[((n*C+group*cpg+ci)*H+ih)*W+iw] : 0.0f;
        }
        for (int k=0; k<4; ++k)
          for (int s=0; s<3; ++s)
            acc[k] += v[k+s]*w[(co*cpg+ci)*9+r*3+s];
      }
    for (int k=0; k<4; ++k)
      if (ow+k<W) y[((n*C+co)*H+oh)*W+ow+k]=acc[k];
}
