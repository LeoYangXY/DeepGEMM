#include <cstdio>
#include <cuda_runtime.h>
template<int C> __global__ void k(float* out,long long* tk,int it){
  float a[C];
#pragma unroll
  for(int i=0;i<C;++i) a[i]=threadIdx.x*0.01f+i;
  float b=1.000001f;
  asm volatile("":::"memory"); long long t0=clock64();
  for(int j=0;j<it;++j){
#pragma unroll
    for(int i=0;i<C;++i) a[i]=fmaf(a[i],b,0.5f);
  }
  long long t1=clock64(); asm volatile("":::"memory");
  float s=0;
#pragma unroll
  for(int i=0;i<C;++i) s+=a[i];
  out[blockIdx.x*blockDim.x+threadIdx.x]=s;
  if(threadIdx.x==0) tk[blockIdx.x]=t1-t0;
}
float* O; long long* T; cudaDeviceProp P;
template<int C> void run(int nw){
  int it=20000; long long t;
  k<C><<<1,nw*32>>>(O,T,it); cudaDeviceSynchronize();
  k<C><<<1,nw*32>>>(O,T,it); cudaMemcpy(&t,T,8,cudaMemcpyDeviceToHost);
  double cpi=(double)t/((double)it*C);
  printf("%3d %5d %9.2f %9.2f %9.2f\n",C,nw,cpi,(double)C/cpi,(double)nw/cpi);
}
int main(){
  cudaGetDeviceProperties(&P,0);
  cudaMalloc(&O,4096*4); cudaMalloc(&T,64*8);
  printf("== 1 warp/SM (ILP scan) ==\n chains warps cyc/FFMA FFMA/cyc  IPC/SM\n");
  run<1>(1); run<2>(1); run<4>(1); run<6>(1); run<8>(1); run<12>(1); run<16>(1);
  printf("== ILP=1, warp scan (TLP) ==\n chains warps cyc/FFMA FFMA/cyc  IPC/SM\n");
  run<1>(2); run<1>(4); run<1>(8); run<1>(16); run<1>(32);
  printf("== ILP=4, warp scan ==\n chains warps cyc/FFMA FFMA/cyc  IPC/SM\n");
  run<4>(2); run<4>(4); run<4>(8); run<4>(16);
  return 0;
}
