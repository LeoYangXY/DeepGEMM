#include <cstdio>
#include <cuda_runtime.h>
__global__ void rd(const float4* __restrict__ g,float4* o,size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x;
  float4 a=make_float4(0,0,0,0);
  for(;i<n;i+=s){float4 v=g[i];a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w;}
  if(a.x==1e30f)o[0]=a;
}
__global__ void lat(const long long* b,int hops,long long* o){
  long long i=0,t0=clock64();
  for(int k=0;k<hops;++k) i=b[i];
  o[0]=(clock64()-t0)/hops; o[1]=i;
}
__global__ void ini(long long* b,long long stride,long long hops){
  long long k=blockIdx.x*(long long)blockDim.x+threadIdx.x;
  if(k<hops) b[k*stride]=((k+1)%hops)*stride;
}
int main(){
  int can01=0,can10=0;
  cudaDeviceCanAccessPeer(&can01,0,1); cudaDeviceCanAccessPeer(&can10,1,0);
  printf("canAccessPeer 0->1=%d 1->0=%d\n",can01,can10);
  cudaSetDevice(0); cudaDeviceEnablePeerAccess(1,0);
  cudaSetDevice(1); cudaDeviceEnablePeerAccess(0,0);
  size_t N=1UL<<28;
  cudaSetDevice(0); char* d0; cudaMalloc(&d0,N); cudaMemset(d0,1,N);
  cudaSetDevice(1); char* d1; cudaMalloc(&d1,N); cudaMemset(d1,2,N);
  cudaSetDevice(0);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  printf("== cudaMemcpyPeer 0->1 ==\n  size(KB)   GB/s\n");
  for(int e=12;e<=28;e+=2){
    size_t sz=1UL<<e; int rep=(sz<(1UL<<20))?200:20;
    cudaMemcpyPeerAsync(d1,1,d0,0,sz,0); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for(int r=0;r<rep;++r) cudaMemcpyPeerAsync(d1,1,d0,0,sz,0);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
    printf("%10zu %7.1f\n",sz/1024,(double)sz*rep/1e9/(ms/1000));
  }
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  size_t n=N/16; float4* o; cudaMalloc(&o,16);
  printf("== kernel read bandwidth ==\n");
  rd<<<p.multiProcessorCount*8,256>>>((float4*)d0,o,n); cudaDeviceSynchronize();
  cudaEventRecord(a); rd<<<p.multiProcessorCount*8,256>>>((float4*)d0,o,n);
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("local  HBM read : %7.1f GB/s\n",(double)N/1e9/(ms/1000));
  rd<<<p.multiProcessorCount*8,256>>>((float4*)d1,o,n); cudaDeviceSynchronize();
  cudaEventRecord(a); rd<<<p.multiProcessorCount*8,256>>>((float4*)d1,o,n);
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("remote NVLink rd: %7.1f GB/s\n",(double)N/1e9/(ms/1000));
  long long* lo; cudaMalloc(&lo,16); long long ho[2];
  long long stride=1024, hops=4096;
  ini<<<(hops+255)/256,256>>>((long long*)d0,stride,hops);
  ini<<<(hops+255)/256,256>>>((long long*)d1,stride,hops);
  cudaDeviceSynchronize();
  lat<<<1,1>>>((long long*)d0,(int)hops,lo); cudaDeviceSynchronize();
  lat<<<1,1>>>((long long*)d0,(int)hops,lo);
  cudaMemcpy(ho,lo,16,cudaMemcpyDeviceToHost);
  printf("local  HBM latency : %5lld cyc  %6.1f ns\n",ho[0],ho[0]*1e6/p.clockRate);
  lat<<<1,1>>>((long long*)d1,(int)hops,lo); cudaDeviceSynchronize();
  lat<<<1,1>>>((long long*)d1,(int)hops,lo);
  cudaMemcpy(ho,lo,16,cudaMemcpyDeviceToHost);
  printf("remote NVLink lat  : %5lld cyc  %6.1f ns\n",ho[0],ho[0]*1e6/p.clockRate);
  return 0;
}
