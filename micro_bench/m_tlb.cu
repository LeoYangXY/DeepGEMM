#include <cstdio>
#include <cuda_runtime.h>
__global__ void ini(int* b,size_t stride,size_t hops){
  size_t k=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(k<hops) b[k*stride]=(int)(((k+1)%hops)*stride);
}
__global__ void chase(const int* b,int hops,long long* out){
  int i=0; __syncwarp(); long long t0=clock64();
  for(int k=0;k<hops;++k) i=b[i];
  long long t1=clock64();
  out[0]=(t1-t0)/hops; out[1]=i;
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  size_t N=1UL<<29;
  int* d; cudaMalloc(&d,N*4);
  long long* o; cudaMalloc(&o,16); long long ho[2];
  printf("stride(KB)  hops  cyc/hop    ns\n");
  for(int s=12;s<=27;++s){
    size_t stride=(1UL<<s)/4, hops=512;
    while(stride*hops>N) hops/=2;
    if(hops<8) break;
    ini<<<(hops+255)/256,256>>>(d,stride,hops);
    chase<<<1,1>>>(d,(int)hops,o); cudaDeviceSynchronize();
    chase<<<1,1>>>(d,(int)hops,o);
    cudaMemcpy(ho,o,16,cudaMemcpyDeviceToHost);
    printf("%9zu %5zu %8lld %6.1f\n",(1UL<<s)/1024,hops,ho[0],ho[0]*1e6/p.clockRate);
  }
  return 0;
}
