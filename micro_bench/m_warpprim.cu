#include <cstdio>
#include <cuda_runtime.h>
#define MW(NM,EX) __global__ void NM(int* o,long long* t,int it){\
 int x0=threadIdx.x+1,x1=x0+1,x2=x0+2,x3=x0+3;\
 asm volatile("":::"memory"); long long a=clock64();\
 for(int i=0;i<it;++i){x0=EX(x0);x1=EX(x1);x2=EX(x2);x3=EX(x3);}\
 long long b=clock64(); asm volatile("":::"memory");\
 o[threadIdx.x]=x0+x1+x2+x3; if(threadIdx.x==0)*t=b-a;}
__shared__ int SM[1024];
__device__ __forceinline__ int w_shfl(int x){return __shfl_sync(0xffffffff,x,(x+1)&31);}
__device__ __forceinline__ int w_xor(int x){return __shfl_xor_sync(0xffffffff,x,1);}
__device__ __forceinline__ int w_down(int x){return __shfl_down_sync(0xffffffff,x,1)+1;}
__device__ __forceinline__ int w_ball(int x){return (int)__ballot_sync(0xffffffff,x&1)+x;}
__device__ __forceinline__ int w_any(int x){return __any_sync(0xffffffff,x&1)+x;}
__device__ __forceinline__ int w_match(int x){return (int)__match_any_sync(0xffffffff,x&3)+x;}
__device__ __forceinline__ int w_red(int x){return __reduce_add_sync(0xffffffff,x&7)+x;}
__device__ __forceinline__ int w_popc(int x){return __popc(x)+x;}
__device__ __forceinline__ int w_smem(int x){int l=threadIdx.x&31,b=threadIdx.x&~31;SM[threadIdx.x]=x;__syncwarp();return SM[b+((l+1)&31)];}
__device__ __forceinline__ int w_add(int x){int r;asm volatile("add.s32 %0,%1,7;":"=r"(r):"r"(x));return r;}
MW(k_add,w_add) MW(k_shfl,w_shfl) MW(k_xor,w_xor) MW(k_down,w_down)
MW(k_ball,w_ball) MW(k_any,w_any) MW(k_match,w_match) MW(k_red,w_red)
MW(k_popc,w_popc) MW(k_smem,w_smem)
int* O; long long* T; double base=0;
void R(void(*k)(int*,long long*,int),const char* nm){
  const int IT=5000; long long t1,t2;
  k<<<1,32>>>(O,T,IT); cudaDeviceSynchronize();
  k<<<1,32>>>(O,T,IT); cudaMemcpy(&t1,T,8,cudaMemcpyDeviceToHost);
  k<<<1,1024>>>(O,T,IT); cudaDeviceSynchronize();
  k<<<1,1024>>>(O,T,IT); cudaMemcpy(&t2,T,8,cudaMemcpyDeviceToHost);
  double c=(double)t1/(IT*4.0), ipc=32.0*4*IT/(double)t2;
  if(base==0) base=ipc;
  printf("%-8s %9.2f %10.2f %8.2f\n",nm,c,ipc,base/ipc);
}
int main(){
  cudaMalloc(&O,4096*4); cudaMalloc(&T,8);
  printf("op        cyc/op(1w) IPC/SM(32w)  xIADD\n");
  R(k_add,"iadd"); R(k_shfl,"shfl_idx"); R(k_xor,"shfl_xor"); R(k_down,"shfl_down");
  R(k_ball,"ballot"); R(k_any,"any"); R(k_match,"match_any"); R(k_red,"redux_add");
  R(k_popc,"popc"); R(k_smem,"smem_xchg");
  return 0;
}
