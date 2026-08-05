#include <cstdio>
#include <cuda_runtime.h>
__global__ void ksync(long long* t,int it){
  long long a=clock64();
  for(int i=0;i<it;++i){ __syncthreads(); }
  long long b=clock64();
  if(threadIdx.x==0)*t=(b-a)/it;
}
__global__ void kwarp(long long* t,int it){
  long long a=clock64();
  for(int i=0;i<it;++i){ __syncwarp(); }
  long long b=clock64();
  if(threadIdx.x==0)*t=(b-a)/it;
}
__global__ void kfence(long long* t,int it,int mode){
  long long a=clock64();
  for(int i=0;i<it;++i){ if(mode) __threadfence(); else __threadfence_block(); }
  long long b=clock64();
  if(threadIdx.x==0)*t=(b-a)/it;
}
__shared__ int sarr[32];
__global__ void kbar(long long* t,int it,int n){
  long long a=clock64();
  for(int i=0;i<it;++i) asm volatile("bar.sync 1, %0;"::"r"(n));
  long long b=clock64();
  if(threadIdx.x==0)*t=(b-a)/it;
}
__global__ void kmbar(long long* t,int it){
  __shared__ unsigned long long mb;
  if(threadIdx.x==0) asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"l"(__cvta_generic_to_shared(&mb)),"r"(blockDim.x));
  __syncthreads();
  long long a=clock64();
  for(int i=0;i<it;++i){
    unsigned long long st; unsigned long long p=__cvta_generic_to_shared(&mb);
    asm volatile("mbarrier.arrive.shared.b64 %0,[%1];":"=l"(st):"l"(p));
    asm volatile("{.reg .pred q; WT: mbarrier.test_wait.shared.b64 q,[%0],%1; @!q bra WT;}"::"l"(p),"l"(st));
  }
  long long b=clock64();
  if(threadIdx.x==0)*t=(b-a)/it;
}
long long* T; long long t;
double base=0;
void R(const char* nm,int thr,long long v){ printf("%-16s thr=%4d  cyc=%6lld\n",nm,thr,v); }
int main(){
  cudaMalloc(&T,8); const int IT=2000;
  int thrs[]={32,64,128,256,512,1024};
  for(int i=0;i<6;++i){
    ksync<<<1,thrs[i]>>>(T,IT); cudaDeviceSynchronize();
    ksync<<<1,thrs[i]>>>(T,IT); cudaMemcpy(&t,T,8,cudaMemcpyDeviceToHost);
    R("__syncthreads",thrs[i],t);
  }
  for(int i=0;i<6;++i){
    kbar<<<1,thrs[i]>>>(T,IT,thrs[i]); cudaDeviceSynchronize();
    kbar<<<1,thrs[i]>>>(T,IT,thrs[i]); cudaMemcpy(&t,T,8,cudaMemcpyDeviceToHost);
    R("bar.sync named",thrs[i],t);
  }
  for(int i=0;i<6;++i){
    kwarp<<<1,thrs[i]>>>(T,IT); cudaDeviceSynchronize();
    kwarp<<<1,thrs[i]>>>(T,IT); cudaMemcpy(&t,T,8,cudaMemcpyDeviceToHost);
    R("__syncwarp",thrs[i],t);
  }
  for(int m=0;m<2;++m) for(int i=0;i<3;++i){
    int th=thrs[i*2+1];
    kfence<<<1,th>>>(T,IT,m); cudaDeviceSynchronize();
    kfence<<<1,th>>>(T,IT,m); cudaMemcpy(&t,T,8,cudaMemcpyDeviceToHost);
    R(m?"threadfence":"threadfence_blk",th,t);
  }
  for(int i=0;i<6;++i){
    kmbar<<<1,thrs[i]>>>(T,IT); cudaDeviceSynchronize();
    kmbar<<<1,thrs[i]>>>(T,IT); cudaMemcpy(&t,T,8,cudaMemcpyDeviceToHost);
    R("mbarrier",thrs[i],t);
  }
  return 0;
}
