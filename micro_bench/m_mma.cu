#include <cstdio>
#include <mma.h>
#include <cuda_runtime.h>
using namespace nvcuda;
template<typename AB,typename CT,int M,int N,int K>
__global__ void mk(float* o,long long* tk,int it){
  wmma::fragment<wmma::matrix_a,M,N,K,AB,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,M,N,K,AB,wmma::col_major> b;
  wmma::fragment<wmma::accumulator,M,N,K,CT> c0,c1,c2,c3;
  wmma::fill_fragment(a,1); wmma::fill_fragment(b,1);
  wmma::fill_fragment(c0,0); wmma::fill_fragment(c1,0);
  wmma::fill_fragment(c2,0); wmma::fill_fragment(c3,0);
  asm volatile("":::"memory"); long long t0=clock64();
  for(int i=0;i<it;++i){
    wmma::mma_sync(c0,a,b,c0); wmma::mma_sync(c1,a,b,c1);
    wmma::mma_sync(c2,a,b,c2); wmma::mma_sync(c3,a,b,c3);
  }
  long long t1=clock64(); asm volatile("":::"memory");
  float s=0;
  for(int i=0;i<c0.num_elements;++i) s+=(float)c0.x[i]+(float)c1.x[i]+(float)c2.x[i]+(float)c3.x[i];
  o[blockIdx.x*blockDim.x+threadIdx.x]=s;
  if(threadIdx.x==0&&blockIdx.x==0)*tk=t1-t0;
}
float* O; long long* TK; int SM;
template<typename AB,typename CT,int M,int N,int K>
void run(const char* nm,const char* unit){
  int it=2000,thr=256,blk=SM*2,nw=thr/32; long long t;
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  mk<AB,CT,M,N,K><<<blk,thr>>>(O,TK,it); cudaDeviceSynchronize();
  cudaEventRecord(a); mk<AB,CT,M,N,K><<<blk,thr>>>(O,TK,it); cudaEventRecord(b);
  cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  cudaMemcpy(&t,TK,8,cudaMemcpyDeviceToHost);
  double ops=(double)blk*nw*it*4*2.0*M*N*K;
  printf("%-16s m%dn%dk%-3d  %7.1f T%s/s   cyc/mma(1warp-view)=%6.2f\n",
    nm,M,N,K,ops/1e12/(ms/1000),unit,(double)t/(it*4.0));
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0); SM=p.multiProcessorCount;
  cudaMalloc(&O,SM*2*256*4); cudaMalloc(&TK,8);
  printf("H20 SM=%d clk=%.2fGHz\n",SM,p.clockRate/1e6);
  run<half,float,16,16,16>("fp16.f32acc","FLOP");
  run<half,half,16,16,16>("fp16.f16acc","FLOP");
  run<__nv_bfloat16,float,16,16,16>("bf16.f32acc","FLOP");
  run<wmma::precision::tf32,float,16,16,8>("tf32.f32acc","FLOP");
  run<signed char,int,16,16,16>("int8.s32acc","OP");
  return 0;
}
