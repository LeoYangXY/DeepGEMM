#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#define CK(x) do{cudaError_t e=(x);if(e){printf("err %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__global__ void cp_copy(const int* g,int n,int mode,long long* out){
 __shared__ int sh[1024];
 int tid=threadIdx.x, gid=blockIdx.x*blockDim.x+threadIdx.x;
 int steps=n/(gridDim.x*blockDim.x*4);
 int acc=0;
 for(int s=0;s<steps;++s){
  int idx=(gid+s*gridDim.x*blockDim.x)*4;
  int* d4=(int*)sh+tid*4; const int* s4=(const int*)g+idx;
  if(mode==0){ for(int k=0;k<4;++k) d4[k]=s4[k]; }
  else if(mode==1){
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned)(unsigned long long)d4), "l"(s4) : "memory");
   asm volatile("cp.async.commit_group;");
   asm volatile("cp.async.wait_group 0;");
  } else {
   __pipeline_memcpy_async((void*)d4,(const void*)s4,16,16);
   __pipeline_commit();
   __pipeline_wait_prior(0);
  }
  acc = acc + *(volatile int*)&d4[0];
 }
 out[0]=acc;
}
int main(){
 CK(cudaSetDevice(0));
 int n=1<<26; int* g; CK(cudaMalloc(&g,n*4)); CK(cudaMemset(g,1,n*4));
 long long* out; CK(cudaMalloc(&out,8));
 int thr=256, blk=312; cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
 const char* nm[]={"manual ld+sts","cp.async 16B","bulk(TMA) 16B"};
 for(int m=0;m<3;++m){
  double bytes=(double)n*4;
  cp_copy<<<blk,thr>>>(g,n,m,out);
  CK(cudaGetLastError());
  cudaEventRecord(a); cp_copy<<<blk,thr>>>(g,n,m,out); cudaEventRecord(b); CK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms,a,b);
  printf("%-16s %8.1f GB/s  (ms=%.3f)\n",nm[m],bytes/1e9/(ms/1000),ms);
 }
 return 0;
}
