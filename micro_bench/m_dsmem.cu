#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
template<int CS,int MODE>
__global__ void __cluster_dims__(CS,1,1) dk(int* o,long long* tk,int it){
  __shared__ int sh[1024];
  cg::cluster_group cl=cg::this_cluster();
  unsigned r=cl.block_rank(), nb=cl.num_blocks();
  for(int i=threadIdx.x;i<1024;i+=blockDim.x) sh[i]=i;
  cl.sync();
  int* p=(MODE==0)?sh:cl.map_shared_rank(sh,(r+1)%nb);
  unsigned base=threadIdx.x&1023; int acc=0;
  long long t0=clock64();
  for(int j=0;j<it;++j){
#pragma unroll
    for(int u=0;u<8;++u) acc+=((volatile int*)p)[(base+u*128u)&1023u];
  }
  long long t1=clock64();
  cl.sync();
  o[blockIdx.x*blockDim.x+threadIdx.x]=acc;
  if(threadIdx.x==0&&blockIdx.x==0)*tk=t1-t0;
}
template<int CS,int MODE>
__global__ void __cluster_dims__(CS,1,1) dl(int* o,long long* tk,int hops){
  __shared__ int sh[1024];
  cg::cluster_group cl=cg::this_cluster();
  unsigned r=cl.block_rank(), nb=cl.num_blocks();
  for(int i=threadIdx.x;i<1024;i+=blockDim.x) sh[i]=(i+37)&1023;
  cl.sync();
  int* p=(MODE==0)?sh:cl.map_shared_rank(sh,(r+1)%nb);
  int i=0; long long t0=clock64();
  for(int k=0;k<hops;++k) i=((volatile int*)p)[i];
  long long t1=clock64();
  cl.sync();
  if(threadIdx.x==0){ o[blockIdx.x]=i; if(blockIdx.x==0)*tk=(t1-t0)/hops; }
}
int* O; long long* TK;
template<int CS,int MODE> void run(const char* nm){
  int thr=256,it=3000,grid=CS*8; long long t;
  dk<CS,MODE><<<grid,thr>>>(O,TK,it); cudaDeviceSynchronize();
  dk<CS,MODE><<<grid,thr>>>(O,TK,it); cudaMemcpy(&t,TK,8,cudaMemcpyDeviceToHost);
  double inst=(double)it*8*(thr/32);
  double B=(double)it*8*thr*4;
  dl<CS,MODE><<<grid,32>>>(O,TK,2000); cudaDeviceSynchronize();
  long long l; dl<CS,MODE><<<grid,32>>>(O,TK,2000);
  cudaMemcpy(&l,TK,8,cudaMemcpyDeviceToHost);
  printf("%-8s cluster=%2d  cyc/LDS=%5.2f  B/clk/SM=%6.1f  lat=%4lld cyc\n",
    nm,CS,(double)t/inst,B/(double)t,l);
}
int main(){
  cudaMalloc(&O,1024*256*4); cudaMalloc(&TK,8);
  int mx=0; cudaLaunchConfig_t cfg={}; cfg.gridDim=dim3(16); cfg.blockDim=dim3(256);
  cudaOccupancyMaxPotentialClusterSize(&mx,(void*)dk<2,1>,&cfg);
  printf("maxPotentialClusterSize=%d\n",mx);
  run<1,0>("local"); run<2,0>("local"); run<2,1>("remote");
  run<4,1>("remote"); run<8,1>("remote");
  return 0;
}
