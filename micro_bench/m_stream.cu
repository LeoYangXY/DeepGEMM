#include <cstdio>
#include <cuda_runtime.h>
__global__ void big(float* o,int it){
  float x=threadIdx.x*1e-3f+1.f,y=x+1,z=x+2,w=x+3;
  for(int i=0;i<it;++i){x=fmaf(x,y,z);y=fmaf(y,z,w);z=fmaf(z,w,x);w=fmaf(w,x,y);}
  if(x+y+z+w==1e30f) o[0]=x;
}
__global__ void small(float* o,int it,long long* tk){
  long long t0=clock64();
  float x=threadIdx.x*1e-3f+1.f,y=x+1,z=x+2,w=x+3;
  for(int i=0;i<it;++i){x=fmaf(x,y,z);y=fmaf(y,z,w);z=fmaf(z,w,x);w=fmaf(w,x,y);}
  if(x+y+z+w==1e30f) o[0]=x;
  if(threadIdx.x==0&&blockIdx.x==0)*tk=clock64()-t0;
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0); int SM=p.multiProcessorCount;
  float* o; cudaMalloc(&o,16); long long* tk; cudaMalloc(&tk,8); long long t;
  cudaStream_t s1,s2,shi,slo; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
  int lo,hi; cudaDeviceGetStreamPriorityRange(&lo,&hi);
  cudaStreamCreateWithPriority(&shi,0,hi); cudaStreamCreateWithPriority(&slo,0,lo);
  printf("priority range lo=%d hi=%d\n",lo,hi);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  int IT=100000;
  big<<<SM,256>>>(o,IT); cudaDeviceSynchronize();
  cudaEventRecord(a); big<<<SM,256>>>(o,IT); cudaEventRecord(b);
  cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("1 kernel (SM blocks)          : %.3f ms\n",ms);
  cudaEventRecord(a); big<<<SM,256,0,s1>>>(o,IT); big<<<SM,256,0,s2>>>(o,IT);
  cudaEventRecord(b); cudaDeviceSynchronize(); cudaEventElapsedTime(&ms,a,b);
  printf("2 kernels 2 streams (half SM) : %.3f ms\n",ms);
  small<<<1,256>>>(o,IT/10,tk); cudaDeviceSynchronize();
  cudaMemcpy(&t,tk,8,cudaMemcpyDeviceToHost);
  printf("small kernel alone           : %lld cyc\n",t);
  big<<<SM*8,256,0,slo>>>(o,IT);
  cudaEventRecord(a); small<<<1,256,0,shi>>>(o,IT/10,tk);
  cudaEventRecord(b); cudaDeviceSynchronize(); cudaEventElapsedTime(&ms,a,b);
  cudaMemcpy(&t,tk,8,cudaMemcpyDeviceToHost);
  printf("small behind full-GPU big    : %lld cyc, wall %.3f ms\n",t,ms);
  return 0;
}
