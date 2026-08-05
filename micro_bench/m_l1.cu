// m_l1.cu -- Hopper L1 / Shared 256KB 可配置切分 (opt-in)
// 每 SM 的 256KB 在 L1 data cache 与 shared memory 之间分配。
// 默认每块只能用 48KB shared, 余下 ~208KB 是 L1; 通过
// cudaFuncAttributeMaxDynamicSharedMemorySize opt-in 可把 shared 提到 227KB
// (此时留给 L1 的只剩 ~29KB)。这是 Hopper 上决定 GEMM tile 大小的关键旋钮。
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e){printf("ERR %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
extern __shared__ char sbuf[];
__global__ void probe(float* o){ if(threadIdx.x==0) o[0]=sbuf[0]; }

int main(){
  int SM=0; cudaDeviceProp p; cudaGetDeviceProperties(&p,0); SM=p.multiProcessorCount;
  int def=0,opt=0;
  cudaDeviceGetAttribute(&def,cudaDevAttrMaxSharedMemoryPerBlock,0);
  cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,0);
  printf("H20 SM=%d  默认 maxShared/block=%dKB  opt-in 上限=%dKB  (L1+Shared 共256KB)\n", SM,def/1024,opt/1024);
  float* o; cudaMalloc(&o,4);
  printf("-- 动态 shared 容量 (kernel 是否 opt-in MaxDynamicSharedMemorySize) --\n");
  for(int setopt=0; setopt<=1; ++setopt){
    int cap = setopt? opt : def;
    CK(cudaFuncSetAttribute((const void*)probe, cudaFuncAttributeMaxDynamicSharedMemorySize, cap));
    int kbs[]={32,64,100,164,200,227};
    for(int i=0;i<6;++i){ int kb=kbs[i];
      probe<<<1,128,(size_t)kb*1024>>>(o); cudaError_t e=cudaGetLastError(); cudaDeviceSynchronize();
      printf("  opt-in=%s  申请%3dKB : %s\n", setopt?"ON ":"OFF", kb, e==cudaSuccess?"OK":"FAIL");
    }
  }
  return 0;
}
