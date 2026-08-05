#include <cstdio>
#include <cuda_runtime.h>
extern __shared__ char shm[];
template<int W,int C> __global__ void kl(float* o,long long* tk,int it){
  unsigned lane=threadIdx.x&31, wid=threadIdx.x>>5;
  unsigned widx = wid*1024u + (C==0?0u:(unsigned)(lane*C));
  unsigned base=(unsigned)__cvta_generic_to_shared(shm)+widx*4u;
  unsigned stride=4096u;
  unsigned q[8][4];
#pragma unroll
  for(int u=0;u<8;++u){q[u][0]=0;q[u][1]=0;q[u][2]=0;q[u][3]=0;}
  long long a=clock64();
  for(int j=0;j<it;++j){
#pragma unroll
    for(int u=0;u<8;++u){
      unsigned ad=base+u*stride;
      if(W==1) asm volatile("ld.volatile.shared.b32 %0,[%1];":"=r"(q[u][0]):"r"(ad));
      else if(W==2) asm volatile("ld.volatile.shared.v2.b32 {%0,%1},[%2];":"=r"(q[u][0]),"=r"(q[u][1]):"r"(ad));
      else asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3},[%4];":"=r"(q[u][0]),"=r"(q[u][1]),"=r"(q[u][2]),"=r"(q[u][3]):"r"(ad));
    }
  }
  long long b=clock64();
  unsigned s=0;
#pragma unroll
  for(int u=0;u<8;++u) s+=q[u][0]+q[u][1]+q[u][2]+q[u][3];
  o[threadIdx.x]=(float)s; if(threadIdx.x==0)*tk=b-a;
}
float* O; long long* T;
template<int W,int C> void run(const char* nm,int nw){
  int thr=nw*32, it=4000, smem=((thr>>5)+8)*4096;
  cudaFuncSetAttribute(kl<W,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,232448);
  long long t; kl<W,C><<<1,thr,smem>>>(O,T,it); cudaDeviceSynchronize();
  kl<W,C><<<1,thr,smem>>>(O,T,it); cudaMemcpy(&t,T,8,cudaMemcpyDeviceToHost);
  double inst=(double)it*8*nw;
  double B=(double)it*8*thr*W*4;
  printf("%-10s nw=%2d cyc/inst=%7.2f  B/clk/SM=%7.1f\n",nm,nw,(double)t/inst,B/(double)t);
}
int main(){
  cudaMalloc(&O,4096*4); cudaMalloc(&T,8);
  printf("== width scan (conflict-free) ==\n");
  run<1,1>("LDS.32",1); run<1,1>("LDS.32",8); run<1,1>("LDS.32",32);
  run<2,2>("LDS.64",1); run<2,2>("LDS.64",8); run<2,2>("LDS.64",32);
  run<4,4>("LDS.128",1); run<4,4>("LDS.128",8); run<4,4>("LDS.128",32);
  printf("== bank conflict scan (LDS.32, 8 warps) ==\n");
  run<1,0>("bcast",8); run<1,1>("1way",8); run<1,2>("2way",8);
  run<1,4>("4way",8); run<1,8>("8way",8); run<1,16>("16way",8); run<1,32>("32way",8);
  return 0;
}
