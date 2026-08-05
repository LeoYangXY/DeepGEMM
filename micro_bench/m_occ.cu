#include <cstdio>
#include <cuda_runtime.h>
extern __shared__ char sh[];
__global__ void bw(const float4* __restrict__ in,float4* out,size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  size_t s=(size_t)gridDim.x*blockDim.x;
  float4 a=make_float4(0,0,0,0);
  for(;i<n;i+=s){ float4 v=in[i]; a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w; }
  if(a.x==1e30f){ out[0]=a; sh[0]=1; }
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  int SM=p.multiProcessorCount, maxs=232448;
  cudaFuncSetAttribute(bw,cudaFuncAttributeMaxDynamicSharedMemorySize,maxs);
  size_t n=(size_t)1<<26; float4* in; cudaMalloc(&in,n*16); cudaMemset(in,0,n*16);
  float4* out; cudaMalloc(&out,16);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  int bs[]={1,2,3,4,6,8,12,16}; int thr=128;
  printf("blk/SM warp/SM occAPI   GB/s\n");
  for(int q=0;q<8;++q){
    int bpsm=bs[q]; int sm=(bpsm>=16)?0:(maxs/bpsm-256);
    int api; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api,bw,thr,sm);
    while(api<bpsm&&sm>1024){sm-=1024;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api,bw,thr,sm);}
    int grid=bpsm*SM;
    bw<<<grid,thr,sm>>>(in,out,n); cudaDeviceSynchronize();
    cudaEventRecord(a); bw<<<grid,thr,sm>>>(in,out,n); cudaEventRecord(b);
    cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);
    printf("%6d %7d %6d %6.0f\n",bpsm,bpsm*thr/32,api,n*16.0/1e9/(ms/1000));
  }
  return 0;
}
