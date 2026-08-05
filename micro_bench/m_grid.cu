#include <cstdio>
#include <cuda_runtime.h>
__global__ void work(float* o,int it){
  float x=threadIdx.x*1e-3f+1.f,y=x+1,z=x+2,w=x+3;
  for(int i=0;i<it;++i){x=fmaf(x,y,z);y=fmaf(y,z,w);z=fmaf(z,w,x);w=fmaf(w,x,y);}
  if(x+y+z+w==1e30f) o[0]=x;
}
__global__ void pers(float* o,int it,int nblk){
  float x=threadIdx.x*1e-3f+1.f,y=x+1,z=x+2,w=x+3;
  for(int b=blockIdx.x;b<nblk;b+=gridDim.x)
    for(int i=0;i<it;++i){x=fmaf(x,y,z);y=fmaf(y,z,w);z=fmaf(z,w,x);w=fmaf(w,x,y);}
  if(x+y+z+w==1e30f) o[0]=x;
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  int SM=p.multiProcessorCount; printf("SM=%d\n",SM);
  float* o; cudaMalloc(&o,4);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  int thr=256, it=4000, SMEM=150000;
  cudaFuncSetAttribute(work,cudaFuncAttributeMaxDynamicSharedMemorySize,232448);
  printf("== tail effect: 1 block/SM per wave (thr=256,smem=0) ==\n blocks   waves      ms  eff%%\n");
  double base=0;
  int gs[]={SM,SM+1,2*SM,2*SM+1,3*SM,4*SM,4*SM+1,6*SM,8*SM,8*SM+1,12*SM,16*SM};
  for(int i=0;i<12;++i){
    int g=gs[i];
    work<<<g,thr,SMEM>>>(o,it); cudaDeviceSynchronize();
    cudaEventRecord(a); work<<<g,thr,SMEM>>>(o,it); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
    if(i==0) base=ms;
    printf("%7d %7.2f %7.3f %5.0f\n",g,(double)g/SM,ms,100.0*(base*g/SM)/ms);
  }
  return 0;
}
