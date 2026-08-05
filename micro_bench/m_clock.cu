#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__global__ void fp32(float* o,int it){
  float x=threadIdx.x*1e-3f+1.f,y=x+1,z=x+2,w=x+3;
  for(int i=0;i<it;++i){x=fmaf(x,y,z);y=fmaf(y,z,w);z=fmaf(z,w,x);w=fmaf(w,x,y);}
  if(x+y+z+w==1e30f) o[0]=x;
}
__global__ void mem(const float4* g,float4* o,size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;
  float4 a=make_float4(0,0,0,0);
  for(;i<n;i+=s){float4 v=g[i];a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w;}
  if(a.x==1e30f)o[0]=a;
}
int main(int argc,char** argv){
  int mode=argc>1?atoi(argv[1]):0, secs=argc>2?atoi(argv[2]):30;
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  int SM=p.multiProcessorCount;
  float* o; cudaMalloc(&o,16);
  size_t n=(size_t)1<<26; float4* g; cudaMalloc(&g,n*16); cudaMemset(g,0,n*16);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  double t=0; int r=0;
  while(t<secs*1000.0){
    cudaEventRecord(a);
    if(mode==0) fp32<<<SM*8,256>>>(o,200000);
    else mem<<<SM*8,256>>>(g,(float4*)o,n);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
    t+=ms; ++r;
    if(mode==0) printf("%3d %8.2fms  %6.2f TFLOPS\n",r,ms,(double)SM*8*256*200000*8/1e12/(ms/1000));
    else printf("%3d %8.2fms  %6.0f GB/s\n",r,ms,(double)n*16/1e9/(ms/1000));
    fflush(stdout);
  }
  return 0;
}
