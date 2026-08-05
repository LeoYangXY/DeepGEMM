#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__global__ void touch(float* p,size_t n,float* o){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;
  float a=0; for(;i<n;i+=s) a+=p[i];
  if(a==1e30f)o[0]=a;
}
static float bwmb(size_t bytes,float ms){ return bytes/1e9f/(ms/1000.f); }
int main(){
  cudaSetDevice(0);
  size_t N=1UL<<30;
  char *pg=(char*)malloc(N), *pn; cudaHostAlloc(&pn,N,cudaHostAllocDefault);
  char* d; cudaMalloc(&d,N);
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("asyncEngineCount=%d unifiedAddr=%d\n",p.asyncEngineCount,p.unifiedAddressing);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  printf("  size(KB)  pgH2D pinH2D  pgD2H pinD2H (GB/s)\n");
  for(int e=12;e<=30;e+=3){
    size_t sz=1UL<<e; int rep=(sz<(1UL<<22))?100:10; float r[4];
    for(int m=0;m<4;++m){
      char* h=(m&1)?pn:pg;
      cudaMemcpyKind k=(m<2)?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost;
      void* dst=(m<2)?(void*)d:(void*)h; void* src=(m<2)?(void*)h:(void*)d;
      cudaMemcpy(dst,src,sz,k); cudaEventRecord(a);
      for(int i=0;i<rep;++i) cudaMemcpy(dst,src,sz,k);
      cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
      r[m]=bwmb(sz*rep,ms);
    }
    printf("%10zu %6.2f %6.2f %6.2f %6.2f\n",sz/1024,r[0],r[1],r[2],r[3]);
  }
  size_t sz=1UL<<28;
  cudaStream_t s1,s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
  char* pn2; cudaHostAlloc(&pn2,sz,cudaHostAllocDefault);
  cudaEventRecord(a);
  cudaMemcpyAsync(d,pn,sz,cudaMemcpyHostToDevice,s1);
  cudaMemcpyAsync(pn2,d+sz,sz,cudaMemcpyDeviceToHost,s2);
  cudaStreamSynchronize(s1); cudaStreamSynchronize(s2);
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("bidir H2D+D2H concurrent : %.2f GB/s aggregate\n",bwmb(sz*2,ms));
  float* mm; cudaMallocManaged(&mm,sz); float* o; cudaMalloc(&o,4);
  memset(mm,0,sz);
  cudaEventRecord(a); touch<<<624,256>>>(mm,sz/4,o); cudaDeviceSynchronize();
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("UM first-touch on GPU    : %.2f GB/s\n",bwmb(sz,ms));
  cudaMemPrefetchAsync(mm,sz,0); cudaDeviceSynchronize();
  cudaEventRecord(a); touch<<<624,256>>>(mm,sz/4,o); cudaDeviceSynchronize();
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("UM after prefetch        : %.2f GB/s\n",bwmb(sz,ms));
  return 0;
}
