#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e){printf("err %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__global__ void div(int mode,int iters,float* out){
 int t=threadIdx.x+blockIdx.x*blockDim.x; int lane=t&31;
 float c[8]; for(int j=0;j<8;++j) c[j]=(j+lane)*0.017f;
 float x=t*0.001f; float s=0;
 for(int i=0;i<iters;++i){
  float w=c[i&7];
  if(mode==0) s=s+x*w;
  else if(mode==1){ if(t<(1<<30)) s=s+x*w; else s=s+x*w*2; }
  else { if(t&1) s=s+x*w; else s=s+x*w*2; }
  x=x*1.0001f+w;
 }
 out[0]=s;
}
int main(){
 CK(cudaSetDevice(0));
 float* out; CK(cudaMalloc(&out,256));
 int thr=256, blk=312, iters=2000000;
 cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
 const char* nm[]={"no branch","uniform branch","divergent branch"};
 float base=0;
 for(int m=0;m<3;++m){
  div<<<blk,thr>>>(m,iters,out);
  cudaEventRecord(a); div<<<blk,thr>>>(m,iters,out); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b);
  if(m==0) base=ms;
  printf("%-18s %7.2f ms  (x%.2f)\n",nm[m],ms,ms/base);
 }
 return 0;
}
