#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e){printf("err %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__global__ void chase(volatile const long long* a,int n,int start,long long* out){
 long long p=start; unsigned long long t0=clock64();
 for(int i=0;i<500000;++i) p=a[p];
 unsigned long long t1=clock64();
 *out=(long long)(t1-t0);
}
__global__ void arith(int op,int n,long long* out){
 float s=0; double d=0; int x=0;
 float c[8]; for(int j=0;j<8;++j) c[j]=1.0f+j*0.013f;
 unsigned long long t0=clock64();
 for(int i=0;i<n;++i){
  int k=i&7;
  if(op==0) s=fmaf(s,c[k],c[(k+1)&7]);
  else if(op==1) s=s+c[k];
  else if(op==2) x=x+(int)c[k];
  else d=d+c[k];
 }
 unsigned long long t1=clock64();
 out[0]=(long long)(t1-t0);
 out[1]=(long long)(s*1e6f+(float)d+x);
}
int main(){
 cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0)); double MHz=p.clockRate/1000.0;
 printf("clock %.0f MHz\n",MHz);
 int n=1<<26; long long* a; CK(cudaMalloc(&a,n*8)); long long* h=(long long*)malloc(n*8);
 for(int i=0;i<n;++i) h[i]=(long long)((i+1)%n);
 CK(cudaMemcpy(a,h,n*8,cudaMemcpyHostToDevice));
 long long* out; CK(cudaMalloc(&out,16)); long long cy;
 chase<<<1,1>>>(a,n,0,out); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&cy,out,8,cudaMemcpyDeviceToHost));
 printf("DRAM(>L2)   %lld cyc = %.1f ns  (per ld %.2f cyc)\n",cy,(double)cy/MHz*1000,(double)cy/500000);
 int n2=1<<22; for(int i=0;i<n2;++i) h[i]=(long long)((i+1)%n2); CK(cudaMemcpy(a,h,n2*8,cudaMemcpyHostToDevice));
 chase<<<1,1>>>(a,n2,0,out); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&cy,out,8,cudaMemcpyDeviceToHost));
 printf("L2(<L2)     %lld cyc = %.1f ns  (per ld %.2f cyc)\n",cy,(double)cy/MHz*1000,(double)cy/500000);
 const char* an[4]={"FFMA","FADD","IMAD","DFMA"};
 for(int op=0;op<4;++op){ arith<<<1,1>>>(op,20000000,out); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&cy,out,8,cudaMemcpyDeviceToHost)); printf("%-5s %.3f ns/op\n",an[op],(double)cy/20000000/MHz*1000); }
 return 0;
}
