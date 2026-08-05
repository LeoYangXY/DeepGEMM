#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
extern __shared__ char shm[];
template<int STAGES> __global__ void pk(const float4* __restrict__ g,float4* o,int nTile){
  float4* sh=(float4*)shm;
  const int T=blockDim.x;
  int gbase=blockIdx.x*nTile*T;
  float4 acc=make_float4(0,0,0,0);
  int fetch=0,compute=0;
  for(;compute<nTile;++compute){
    for(;fetch<nTile&&fetch<compute+STAGES;++fetch){
      __pipeline_memcpy_async(&sh[(fetch%STAGES)*T+threadIdx.x],
        &g[gbase+fetch*T+threadIdx.x],16,0);
      __pipeline_commit();
    }
    __pipeline_wait_prior(STAGES-1);
    __syncthreads();
    float4 v=sh[(compute%STAGES)*T+threadIdx.x];
    acc.x+=v.x;acc.y+=v.y;acc.z+=v.z;acc.w+=v.w;
    __syncthreads();
  }
  if(acc.x==1e30f)o[0]=acc;
}
__global__ void plain(const float4* __restrict__ g,float4* o,int nTile){
  const int T=blockDim.x; int gbase=blockIdx.x*nTile*T;
  float4 acc=make_float4(0,0,0,0);
  for(int i=0;i<nTile;++i){ float4 v=g[gbase+i*T+threadIdx.x];
    acc.x+=v.x;acc.y+=v.y;acc.z+=v.z;acc.w+=v.w; }
  if(acc.x==1e30f)o[0]=acc;
}
float4 *G,*O; int SM,nTile=64,THR=256;
template<int S> void run(){
  int grid=SM*4, smem=S*THR*16;
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  pk<S><<<grid,THR,smem>>>(G,O,nTile); cudaDeviceSynchronize();
  cudaEventRecord(a); pk<S><<<grid,THR,smem>>>(G,O,nTile); cudaEventRecord(b);
  cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  double B=(double)grid*nTile*THR*16;
  printf("cp.async stages=%d  %8.0f GB/s\n",S,B/1e9/(ms/1000));
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0); SM=p.multiProcessorCount;
  size_t N=(size_t)SM*4*nTile*THR;
  cudaMalloc(&G,N*16); cudaMemset(G,0,N*16); cudaMalloc(&O,16);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  plain<<<SM*4,THR>>>(G,O,nTile); cudaDeviceSynchronize();
  cudaEventRecord(a); plain<<<SM*4,THR>>>(G,O,nTile); cudaEventRecord(b);
  cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("plain global read   %8.0f GB/s\n",(double)N*16/1e9/(ms/1000));
  run<1>(); run<2>(); run<3>(); run<4>(); run<6>(); run<8>(); run<12>();
  return 0;
}
