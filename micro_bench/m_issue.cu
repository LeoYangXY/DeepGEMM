#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e){printf("err %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__global__ void issue(int ilp,int warps,int iters,long long* out){
 int w=threadIdx.x>>5; if(w>=warps) return;
 int lane=threadIdx.x&31;
 float a[16]; float cc[16]; for(int j=0;j<ilp;++j){ a[j]=(j+lane)*0.01f; cc[j]=1.0f+j*0.013f; }
 float b=1.000001f;
 unsigned long long t0=clock64();
 for(int it=0;it<iters;++it) for(int j=0;j<ilp;++j) a[j]=fmaf(a[j],b,cc[(j+it)&15]);
 unsigned long long t1=clock64();
 float sum=0; for(int j=0;j<ilp;++j) sum+=a[j];
 if(lane==0){ out[w]=(long long)(t1-t0); out[w+16]=(long long)(sum*1e6f); }
}
int main(){
 CK(cudaSetDevice(0));
 long long* out; CK(cudaMalloc(&out,256)); long long cy,hh[4];
 int iters=1000000;
 printf("ILP sweep (1 warp):\n");
 for(int ilp=1;ilp<=16;ilp<<=1){
  issue<<<2,64>>>(ilp,1,iters,out); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&cy,out,8,cudaMemcpyDeviceToHost));
  printf(" ilp=%2d  %lld cyc  IPC=%.3f\n",ilp,cy,(double)iters*ilp/cy);
 }
 printf("warps/SMSP (ilp=16):\n");
 for(int wp=1;wp<=2;++wp){
  issue<<<2,64>>>(16,wp,iters,out); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hh,out,wp*8,cudaMemcpyDeviceToHost));
  long long tot=0; for(int i=0;i<wp;++i) tot+=hh[i];
  printf(" warps=%d  sum_cyc=%lld  IPC_smspp=%.3f\n",wp,tot,(double)iters*16*wp/tot);
 }
 return 0;
}
