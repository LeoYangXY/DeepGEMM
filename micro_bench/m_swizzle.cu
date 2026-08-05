#include <cstdio>
#include <cuda_runtime.h>
__shared__ unsigned short A[64*66];
template<int MODE> __global__ void sw(float* o,long long* tk,int it){
  int lane=threadIdx.x&31, wid=threadIdx.x>>5;
  for(int i=threadIdx.x;i<64*66;i+=blockDim.x) A[i]=(unsigned short)i;
  __syncthreads();
  unsigned acc=0, sb=(unsigned)__cvta_generic_to_shared(A);
  long long t0=clock64();
  for(int j=0;j<it;++j){
#pragma unroll
    for(int u=0;u<8;++u){
      int row=lane, col=(u*4+wid)&63, idx;
      if(MODE==0) idx=row*64+col;
      else if(MODE==1) idx=row*66+col;
      else idx=row*64+((((col>>3)^(row&7))<<3)+(col&7));
      unsigned v, ad=sb+(unsigned)idx*2u;
      asm volatile("ld.volatile.shared.u16 %0,[%1];":"=r"(v):"r"(ad));
      acc+=v;
    }
  }
  long long t1=clock64();
  o[threadIdx.x]=(float)acc; if(threadIdx.x==0)*tk=t1-t0;
}
float* O; long long* TK;
template<int M> void run(const char* nm){
  int thr=256,it=4000; long long t;
  sw<M><<<1,thr>>>(O,TK,it); cudaDeviceSynchronize();
  sw<M><<<1,thr>>>(O,TK,it); cudaMemcpy(&t,TK,8,cudaMemcpyDeviceToHost);
  double inst=(double)it*8*(thr/32);
  printf("%-22s cyc/LDS.u16 = %6.2f  (smem %d B)\n",nm,(double)t/inst,
    (M==1?64*66:64*64)*2);
}
int main(){
  cudaMalloc(&O,1024*4); cudaMalloc(&TK,8);
  printf("column read of 64x64 half tile, 8 warps\n");
  run<0>("naive [64][64]");
  run<1>("padded [64][66]");
  run<2>("XOR swizzle 8x8");
  return 0;
}
