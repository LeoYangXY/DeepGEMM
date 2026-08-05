#include <cstdio>
#include <cuda_runtime.h>
__global__ void ag(int* d,int nadd,int it){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  int idx=(nadd==0)?0:(t%nadd);
  for(int i=0;i<it;++i) atomicAdd(&d[idx],1);
}
__global__ void agret(int* d,int nadd,int it,int* o){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  int idx=(nadd==0)?0:(t%nadd); int s=0;
  for(int i=0;i<it;++i) s+=atomicAdd(&d[idx],1);
  if(s==-1) o[0]=s;
}
__global__ void as(int* o,int nadd,int it){
  __shared__ int sh[1024];
  int idx=(nadd==0)?0:(threadIdx.x%nadd);
  sh[threadIdx.x]=0; __syncthreads();
  int v=0;
  for(int i=0;i<it;++i) v+=atomicAdd(&sh[idx],1);
  __syncthreads(); if(v==-7) o[0]=1;
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  int SM=p.multiProcessorCount, thr=256, blk=SM*4, it=200;
  int* d; cudaMalloc(&d,1<<20); cudaMemset(d,0,1<<20);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  double tot=(double)thr*blk*it;
  int ns[]={0,1,2,4,8,16,32,64,256,1024,65536};
  printf("== global atomicAdd, %d threads ==\n uniqAddr    Gop/s\n",thr*blk);
  for(int i=0;i<11;++i){
    ag<<<blk,thr>>>(d,ns[i],it); cudaDeviceSynchronize();
    cudaEventRecord(a); ag<<<blk,thr>>>(d,ns[i],it); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
    printf("%9d %8.2f\n",ns[i]?ns[i]:1,tot/1e9/(ms/1000));
  }
  printf("== global atomicAdd WITH return value ==\n uniqAddr    Gop/s\n");
  int* o; cudaMalloc(&o,4);
  for(int i=0;i<11;++i){
    agret<<<blk,thr>>>(d,ns[i],it,o); cudaDeviceSynchronize();
    cudaEventRecord(a); agret<<<blk,thr>>>(d,ns[i],it,o); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
    printf("%9d %8.2f\n",ns[i]?ns[i]:1,tot/1e9/(ms/1000));
  }
  printf("== shared atomicAdd ==\n uniqAddr    Gop/s\n");
  int ns2[]={0,1,2,4,8,16,32,64,256};
  for(int i=0;i<9;++i){
    as<<<blk,thr>>>(o,ns2[i],it); cudaDeviceSynchronize();
    cudaEventRecord(a); as<<<blk,thr>>>(o,ns2[i],it); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
    printf("%9d %8.2f\n",ns2[i]?ns2[i]:1,tot/1e9/(ms/1000));
  }
  return 0;
}
