#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define MK(NM,EX) __global__ void NM(float* o,long long* t,int it){\
 float x0=threadIdx.x*1e-3f+1.f,x1=x0+.1f,x2=x0+.2f,x3=x0+.3f;\
 asm volatile("":::"memory"); long long a=clock64();\
 for(int i=0;i<it;++i){x0=EX(x0);x1=EX(x1);x2=EX(x2);x3=EX(x3);}\
 long long b=clock64(); asm volatile("":::"memory");\
 o[threadIdx.x]=x0+x1+x2+x3; if(threadIdx.x==0)*t=b-a;}
#define MKI(NM,EX) __global__ void NM(int* o,long long* t,int it){\
 int d=o[1000]; int x0=threadIdx.x+1,x1=x0+1,x2=x0+2,x3=x0+3;\
 asm volatile("":::"memory"); long long a=clock64();\
 for(int i=0;i<it;++i){x0=EX(x0);x1=EX(x1);x2=EX(x2);x3=EX(x3);}\
 long long b=clock64(); asm volatile("":::"memory");\
 o[threadIdx.x]=x0+x1+x2+x3; if(threadIdx.x==0)*t=b-a;}
__device__ __forceinline__ float f1(float x){return fmaf(x,1.000001f,0.5f);}
__device__ __forceinline__ float f2(float x){return x+1.000001f;}
__device__ __forceinline__ float f3(float x){return x*1.000001f;}
__device__ __forceinline__ float f4(float x){return __frcp_rn(x)+1.5f;}
__device__ __forceinline__ float f5(float x){return __fdividef(3.f,x)+1.f;}
__device__ __forceinline__ float f6(float x){return 3.f/x+1.f;}
__device__ __forceinline__ float f7(float x){return rsqrtf(x)+1.f;}
__device__ __forceinline__ float f8(float x){return sqrtf(x)+1.f;}
__device__ __forceinline__ float f9(float x){return __sinf(x)+2.f;}
__device__ __forceinline__ float fa(float x){return sinf(x)+2.f;}
__device__ __forceinline__ float fb(float x){return __expf(x*1e-3f);}
__device__ __forceinline__ float fc(float x){return expf(x*1e-3f);}
__device__ __forceinline__ float fd(float x){return __logf(x)+2.f;}
__device__ __forceinline__ float fe(float x){return tanhf(x)+1.f;}
__device__ __forceinline__ float ff(float x){return powf(x,1.5f)+1.f;}
__device__ __forceinline__ float fg(float x){double d=x;return (float)fma(d,1.000001,0.5);}
MK(kf1,f1) MK(kf2,f2) MK(kf3,f3) MK(kf4,f4) MK(kf5,f5) MK(kf6,f6) MK(kf7,f7) MK(kf8,f8)
MK(kf9,f9) MK(kfa,fa) MK(kfb,fb) MK(kfc,fc) MK(kfd,fd) MK(kfe,fe) MK(kff,ff) MK(kfg,fg)
__device__ __forceinline__ int i1(int x,int d){return x+d;}
__device__ __forceinline__ int i2(int x,int d){return x*d+7;}
__device__ __forceinline__ int i3(int x,int d){return x*d;}
__device__ __forceinline__ int i4(int x,int d){return x/d+1;}
__device__ __forceinline__ int i5(int x,int d){return x%d+1;}
__device__ __forceinline__ int i6(int x,int d){return __popc(x)+d;}
__device__ __forceinline__ int i7(int x,int d){return __clz(x)+d;}
__device__ __forceinline__ int i8(int x,int d){return (int)__brev(x)+d;}
__device__ __forceinline__ int i9(int x,int d){return __funnelshift_l(x,x,3)+d;}
__device__ __forceinline__ int ia(int x,int d){return max(x,d)+1;}
#define MI(NM,EX) __global__ void NM(int* o,long long* t,int it){\
 int d=o[1000]; int x0=threadIdx.x+1,x1=x0+1,x2=x0+2,x3=x0+3;\
 asm volatile("":::"memory"); long long a=clock64();\
 for(int i=0;i<it;++i){x0=EX(x0,d);x1=EX(x1,d);x2=EX(x2,d);x3=EX(x3,d);}\
 long long b=clock64(); asm volatile("":::"memory");\
 o[threadIdx.x]=x0+x1+x2+x3; if(threadIdx.x==0)*t=b-a;}
MI(ki1,i1) MI(ki2,i2) MI(ki3,i3) MI(ki4,i4) MI(ki5,i5)
MI(ki6,i6) MI(ki7,i7) MI(ki8,i8) MI(ki9,i9) MI(kia,ia)
float* FO; int* IO; long long* T; double base=0; const int IT=4000;
static void pr(const char* nm,long long t1,long long t2){
  double c=(double)t1/(IT*4.0), ipc=32.0*4*IT/(double)t2;
  if(base==0) base=ipc;
  printf("%-9s %9.2f %10.3f %8.2f\n",nm,c,ipc,base/ipc);
}
void RF(void(*k)(float*,long long*,int),const char* nm){
  long long t1,t2; k<<<1,32>>>(FO,T,IT); cudaDeviceSynchronize();
  k<<<1,32>>>(FO,T,IT); cudaMemcpy(&t1,T,8,cudaMemcpyDeviceToHost);
  k<<<1,1024>>>(FO,T,IT); cudaDeviceSynchronize();
  k<<<1,1024>>>(FO,T,IT); cudaMemcpy(&t2,T,8,cudaMemcpyDeviceToHost); pr(nm,t1,t2);
}
void RI(void(*k)(int*,long long*,int),const char* nm){
  long long t1,t2; k<<<1,32>>>(IO,T,IT); cudaDeviceSynchronize();
  k<<<1,32>>>(IO,T,IT); cudaMemcpy(&t1,T,8,cudaMemcpyDeviceToHost);
  k<<<1,1024>>>(IO,T,IT); cudaDeviceSynchronize();
  k<<<1,1024>>>(IO,T,IT); cudaMemcpy(&t2,T,8,cudaMemcpyDeviceToHost); pr(nm,t1,t2);
}
#include "extra_ops.cuh"
int main(){
  cudaMalloc(&FO,4096*4); cudaMalloc(&IO,4096*4); cudaMalloc(&T,8);
  int sev=7; cudaMemcpy(IO+1000,&sev,4,cudaMemcpyHostToDevice);
  printf("op         cyc/op(1w) IPC/SM(32w)   xFFMA\n");
  RF(kf1,"ffma"); RF(kf2,"fadd"); RF(kf3,"fmul"); RF(kf4,"rcp_rn");
  RF(kf5,"fdividef"); RF(kf6,"fdiv_prec"); RF(kf7,"rsqrtf"); RF(kf8,"sqrtf");
  RF(kf9,"__sinf"); RF(kfa,"sinf"); RF(kfb,"__expf"); RF(kfc,"expf");
  RF(kfd,"__logf"); RF(kfe,"tanhf"); RF(kff,"powf"); RF(kfg,"dfma_f64");
  RI(ki1,"iadd"); RI(ki2,"imad"); RI(ki3,"imul"); RI(ki4,"idiv");
  RI(ki5,"imod"); RI(ki6,"popc"); RI(ki7,"clz"); RI(ki8,"brev");
  RI(ki9,"funnelshf"); RI(kia,"imax");
  printf("-- cyclic-dep (fold-proof) --\n");
  RI(kc1,"iadd_c"); RI(kc2,"imad_c"); RI(kc3,"imul_c"); RI(kc4,"i64add_c");
  RI(kc5,"i64mul_c"); RI(kc6,"dfma_c"); RI(kc7,"ffma_c"); RI(kc8,"hfma2_c");
  return 0;
}
