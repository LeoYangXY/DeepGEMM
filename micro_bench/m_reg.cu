// m_reg.cu -- 寄存器压力与溢出 (register spilling)
// 用 32 个独立 FMA 链强制高寄存器占用。__launch_bounds__(256,MINB) 要求
// 至少 MINB 个 block 同驻一 SM, 当 MINB 大到寄存器不够时, 编译器把多余
// 累加器溢出到 local memory (即 HBM), 吞吐骤降。
// 教训: 不要盲目调高占用率, 寄存器不够时"高占用"反而因溢出更慢。
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e){printf("ERR %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
template<int MINB>
__global__ __launch_bounds__(256, MINB)
void reg_pressure(float* o, long long it){
  float v[32];
  for(int i=0;i<32;++i) v[i]=(float)(threadIdx.x+i);
  for(long long r=0;r<it;++r)
    for(int i=0;i<32;++i) v[i]=v[i]*1.0001f+1.0f;
  float s=0; for(int i=0;i<32;++i) s+=v[i];
  o[blockIdx.x*blockDim.x+threadIdx.x]=s;
}

int main(){
  int SM=0; cudaDeviceProp p; cudaGetDeviceProperties(&p,0); SM=p.multiProcessorCount;
  float* o; cudaMalloc(&o,SM*4*256*4);
  int thr=256, blk=SM*4; long long it=200000;
  double flop=(double)blk*thr*it*32.0*2.0;
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  printf("MINB  regs  GFLOP/s\n");
  int minbs[]={1,2,4,8};
  for(int k=0;k<4;++k){ int MB=minbs[k];
    cudaFuncAttributes at; const void* f=0;
    if(MB==1) f=(const void*)reg_pressure<1>; else if(MB==2) f=(const void*)reg_pressure<2>;
    else if(MB==4) f=(const void*)reg_pressure<4>; else if(MB==8) f=(const void*)reg_pressure<8>;
    else f=(const void*)reg_pressure<8>;
    cudaFuncGetAttributes(&at,f);
    if(MB==1) reg_pressure<1><<<blk,thr>>>(o,it);
    else if(MB==2) reg_pressure<2><<<blk,thr>>>(o,it);
    else if(MB==4) reg_pressure<4><<<blk,thr>>>(o,it);
    else if(MB==8) reg_pressure<8><<<blk,thr>>>(o,it);
    else reg_pressure<8><<<blk,thr>>>(o,it);
    CK(cudaGetLastError());
    cudaEventRecord(a);
    if(MB==1) reg_pressure<1><<<blk,thr>>>(o,it);
    else if(MB==2) reg_pressure<2><<<blk,thr>>>(o,it);
    else if(MB==4) reg_pressure<4><<<blk,thr>>>(o,it);
    else if(MB==8) reg_pressure<8><<<blk,thr>>>(o,it);
    else reg_pressure<8><<<blk,thr>>>(o,it);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms,a,b);
    printf("  %2d    %3d   %7.1f\n", MB, at.numRegs, flop/1e9/(ms/1000));
  }
  return 0;
}
