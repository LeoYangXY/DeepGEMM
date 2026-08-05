#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e){printf("err %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__global__ void gmem(int mode,const float* g,const int* perm,int n,long long* out){
 int gid=blockIdx.x*blockDim.x+threadIdx.x;
 int BS=blockDim.x*gridDim.x;
 int i=gid; float s=0;
 for(int k=0;k<200;++k){
  float v;
  if(mode==0) v=g[i&(n-1)];
  else if(mode==1) v=g[(i*32)&(n-1)];
  else if(mode==2) v=g[perm[i&(n-1)]&(n-1)];
  else { const float4* f4=(const float4*)g; float4 w=f4[i&(n/4-1)]; v=w.x+w.y+w.z+w.w; }
  s+=v; i=(i+BS)&(n-1);
 }
 *out=(long long)s;
}
int main(){
 CK(cudaSetDevice(0));
 int n=1<<26; float* g; int* perm; CK(cudaMalloc(&g,n*4)); CK(cudaMalloc(&perm,n*4));
 CK(cudaMemset(g,1,n*4)); int* h=(int*)malloc(n*4); for(int i=0;i<n;++i) h[i]=rand()%n; CK(cudaMemcpy(perm,h,n*4,cudaMemcpyHostToDevice));
 long long* out; CK(cudaMalloc(&out,8));
 int thr=256, blk=1920; cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
 const char* nm[]={"contig scalar","stride32","random perm","vec float4"};
 for(int m=0;m<4;++m){
  int bpe=(m==3)?16:4; double bytes=(double)thr*blk*bpe*200;
  gmem<<<blk,thr>>>(m,g,perm,n,out);
  cudaEventRecord(a); gmem<<<blk,thr>>>(m,g,perm,n,out); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b);
  printf("%-14s %7.3f ms  %7.1f GB/s\n",nm[m],ms,bytes/1e9/ms*1000);
 }
 return 0;
}
