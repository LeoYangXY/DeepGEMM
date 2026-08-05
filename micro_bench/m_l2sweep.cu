#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e){printf("err %d %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)
__global__ void st(const float* g,unsigned mask,float* out){
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned step=gridDim.x*blockDim.x;
  float s=0;
#pragma unroll 8
  for(int k=0;k<256;++k){ s+=g[i&mask]; i+=step; }
  if(s==1234.5f) out[0]=s;
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("SM=%d L2=%.1fMB clk=%.2fGHz\n",p.multiProcessorCount,p.l2CacheSize/1048576.0,p.clockRate/1e6);
  size_t big=1UL<<29; float* g; CK(cudaMalloc(&g,big)); CK(cudaMemset(g,0,big));
  float* out; CK(cudaMalloc(&out,16));
  int thr=256, blk=p.multiProcessorCount*8;
  double acc=(double)thr*blk*256*4;
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  printf(" WS(KB)     GB/s\n");
  for(int e=14;e<=29;++e){
    unsigned n=(1u<<e)/4, mask=n-1;
    st<<<blk,thr>>>(g,mask,out); CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for(int r=0;r<5;++r) st<<<blk,thr>>>(g,mask,out);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b);
    printf("%7d %8.0f\n",(1<<e)/1024,acc*5/1e9/(ms/1000));
  }
  return 0;
}
