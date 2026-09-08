// Sustained logical copy bandwidth and explicit FP32 FMA throughput probes.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void initialize(float* x,size_t n) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) x[i]=float(i%1024)*0.001f;
}
__global__ void logical_copy(const float* x,float* y,size_t n) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) y[i]=x[i];
}
__global__ void eight_fma_chains(const float* x,float* y,int iters) {
  const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
  const float b=x[i]*0.000001f;
  float a0=0.01f,a1=0.02f,a2=0.03f,a3=0.04f,a4=0.05f,a5=0.06f,a6=0.07f,a7=0.08f;
  for(int k=0;k<iters;k++) {
    a0=__fmaf_rn(a0,1.000001f,b); a1=__fmaf_rn(a1,1.000001f,b);
    a2=__fmaf_rn(a2,1.000001f,b); a3=__fmaf_rn(a3,1.000001f,b);
    a4=__fmaf_rn(a4,1.000001f,b); a5=__fmaf_rn(a5,1.000001f,b);
    a6=__fmaf_rn(a6,1.000001f,b); a7=__fmaf_rn(a7,1.000001f,b);
  }
  y[i]=((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7));
}

int main(int argc,char** argv) {
 try {
  cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p,0));
  cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  cudaEvent_t start,end; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&end));
  const bool smoke=argc>1 && std::string(argv[1])=="--smoke";
  const int batches=smoke?1:5,samples=smoke?2:30;
  printf("{\"kind\":\"environment\",\"gpu\":\"%s\",\"l2_bytes\":%d,\"sm_count\":%d,\"clock_control\":\"NOT_CONTROLLED\",\"counter_traffic\":\"NOT_MEASURED\"}\n",p.name,p.l2CacheSize,p.multiProcessorCount);
  for(int mode=0;mode<2;mode++) for(int block: {128,256}) for(int scale: {1,2}) {
   const size_t n=mode==0 ? std::max(size_t(64<<20),size_t(p.l2CacheSize)*4)*scale/sizeof(float) : size_t(p.multiProcessorCount)*block*8;
   const int blocks=mode==0 ? p.multiProcessorCount*8 : int(n/block),iters=4096*scale;
   float *x,*y; ck(cudaMalloc(&x,n*4));ck(cudaMalloc(&y,n*4));
   initialize<<<blocks,block,0,stream>>>(x,n);ck(cudaGetLastError());
   auto call=[&]() { if(mode==0)logical_copy<<<blocks,block,0,stream>>>(x,y,n); else eight_fma_chains<<<blocks,block,0,stream>>>(x,y,iters); ck(cudaGetLastError()); };
   for(int k=0;k<10;k++)call();ck(cudaStreamSynchronize(stream));
   std::vector<float> first(16);ck(cudaMemcpy(first.data(),y,first.size()*4,cudaMemcpyDeviceToHost));
   for(int i=0;i<16;i++) {
    float expected=float(i%1024)*0.001f;
    if(mode) {
     float a[8]={0.01f,0.02f,0.03f,0.04f,0.05f,0.06f,0.07f,0.08f};
     float b=expected*0.000001f;
     for(int k=0;k<iters;k++)for(auto& v:a)v=std::fma(v,1.000001f,b);
     expected=((a[0]+a[1])+(a[2]+a[3]))+((a[4]+a[5])+(a[6]+a[7]));
    }
    if(!std::isfinite(first[i]) || std::abs(first[i]-expected)>1e-5f)throw std::runtime_error("microbenchmark output check failed");
   }
   for(int b=0;b<batches;b++)for(int s=0;s<samples;s++) {
    ck(cudaEventRecord(start,stream));call();ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));
    float ms;ck(cudaEventElapsedTime(&ms,start,end));
    printf("{\"kind\":\"sample\",\"operation\":\"%s\",\"block\":%d,\"scale\":%d,\"batch\":%d,\"sample\":%d,\"threads\":%zu,\"buffer_bytes\":%zu,\"read_write_bytes\":%zu,\"fp32_fma_flops\":%zu,\"iterations\":%d,\"t_device_us\":%.9g,\"configuration_check\":\"PASS\",\"validation_scope\":\"first_16_after_warmup\"}\n",mode?"eight_fma_chains":"logical_copy",block,scale,b,s,n,n*4,mode?size_t(0):n*8,mode?n*size_t(iters)*16:size_t(0),mode?iters:0,ms*1000.0);
   }
   ck(cudaFree(x));ck(cudaFree(y));
  }
  ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));ck(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}
}
