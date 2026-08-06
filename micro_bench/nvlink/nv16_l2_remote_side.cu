// ============================================================================
// nv16_l2_remote_side —— 远端请求到达 GPU1 后, 是命中 GPU1 的 L2 还是穿透到
//                        GPU1 的 HBM?
//
// 难点: nv15 已证明 GPU0 的本地 L2 会缓存 peer 内存。要隔离"远端侧"的效应,
//       必须让 GPU0 侧一定 miss。做法:
//         - 用 ld.global.cv (nv15 表 A 证明 .cv 在任何 WS 都是 ~1583 cyc,
//           即完全绕过 GPU0 本地缓存)。这样测到的差异只能来自 GPU1 侧。
//         - 同时 GPU0 侧再跑一个 L2 冲刷 kernel 作双保险。
//
// 变量: GPU1 侧 L2 状态
//   warm : 测量前在 GPU1 上跑 kernel 把整个 WS 读一遍 -> 尽量驻留 GPU1 L2
//   cold : 测量前在 GPU1 上流式读一个 256MB 的无关 buffer -> 冲掉 GPU1 L2
// WS 从 1MB 扫到 256MB, 跨过 GPU1 的 60MB L2。
//
// 若 warm 与 cold 在 WS<60MB 时有稳定差值 D -> D 即"远端 L2 命中 vs 远端 HBM"
// 的差; 若无差 -> 远端请求不查 GPU1 L2 (直接到 HBM) 或 L2 无法为 peer 请求保留。
// ============================================================================
#include "nvl_common.cuh"
#include <vector>
#include <algorithm>
#include <random>

__device__ __forceinline__ unsigned ld_cv_(const unsigned* p) {
  unsigned v; asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned ld_g_(const unsigned* p) {
  unsigned v; asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

template<int MOD>
__global__ void k_chase(const unsigned* chain, int iters, long long* out) {
  unsigned cur = 0;
  for (int i=0;i<256;++i) cur = MOD? ld_cv_(chain+cur) : ld_g_(chain+cur);
  long long t0 = clk();
  #pragma unroll 8
  for (int i=0;i<iters;++i) cur = MOD? ld_cv_(chain+cur) : ld_g_(chain+cur);
  long long t1 = clk();
  out[0]=t1-t0; out[1]=cur;
}

// 焐热 / 冲刷: 流式读
__global__ void k_touch(const uint4* __restrict__ p, size_t n, uint4* sink) {
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x*blockDim.x;
  uint4 acc = make_uint4(0,0,0,0);
  for (; i<n; i+=s) { uint4 v = ld128(p+i); acc.x^=v.x; acc.y^=v.y; acc.z^=v.z; acc.w^=v.w; }
  if (acc.x==0xdeadbeefu && acc.y==0xfeedfaceu) *sink=acc;
}
// 带宽 kernel(远端流式读, 用于带宽视角)
__global__ void k_stream_v(const uint4* __restrict__ src, size_t n, uint4* sink) {
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x*blockDim.x;
  uint4 acc = make_uint4(0,0,0,0);
  for (; i<n; i+=s) { uint4 v = ld128_v(src+i); acc.x^=v.x; acc.y^=v.y; acc.z^=v.z; acc.w^=v.w; }
  if (acc.x==0xdeadbeefu && acc.y==0xfeedfaceu) *sink=acc;
}

static std::vector<unsigned> build_chain(size_t bytes, size_t stride, unsigned seed) {
  size_t nnode = bytes/stride; if (nnode<8){nnode=8;stride=bytes/8;}
  std::vector<unsigned> h(bytes/4,0u);
  std::vector<unsigned> idx(nnode);
  for (size_t i=0;i<nnode;++i) idx[i]=(unsigned)(i*stride/4);
  std::vector<unsigned> perm(idx.begin()+1, idx.end());
  std::mt19937 rng(seed); std::shuffle(perm.begin(),perm.end(),rng);
  unsigned cur=idx[0];
  for (unsigned p:perm){h[cur]=p;cur=p;}
  h[cur]=idx[0];
  return h;
}

int main() {
  NvlEnv env = nvl_init(2);
  const double GHZ=1.98;
  printf("# nv16_l2_remote_side — %s, GPU1 侧 L2 状态对远端读延迟的影响\n", env.name);
  printf("GPU0 侧一律用 ld.global.cv (nv15 已证明该 modifier 完全绕过 GPU0 本地缓存)\n");
  nvl_enable_peers(env.ndev);

  CK(cudaSetDevice(0));
  long long* d_out; CK(cudaMalloc(&d_out,64));
  uint4* sink0=(uint4*)nvl_alloc(0,1024);
  uint4* sink1=(uint4*)nvl_alloc(1,1024);
  // GPU1 上的 L2 冲刷用大 buffer (>60MB)
  const size_t FLUSH = 256ull<<20;
  uint4* flush1 = (uint4*)nvl_alloc(1, FLUSH);
  // GPU0 上的 L2 冲刷 buffer (双保险)
  uint4* flush0 = (uint4*)nvl_alloc(0, FLUSH);
  CK(cudaSetDevice(0));

  int grid=env.sm*4, blk=256;
  const int ITERS=2048;

  size_t ws[]={1ull<<20,4ull<<20,8ull<<20,16ull<<20,32ull<<20,48ull<<20,
               64ull<<20,128ull<<20,256ull<<20};
  const int NW=sizeof(ws)/sizeof(ws[0]);

  hdr("A) 远端 pointer-chase 延迟: GPU1 L2 warm vs cold (GPU0 用 .cv 强制 miss)");
  printf("| GPU1 WS | warm (cyc) | cold (cyc) | cold-warm (cyc) | warm ns | cold ns |\n");
  printf("|---|---|---|---|---|---|\n");
  for (int w=0;w<NW;++w) {
    size_t B=ws[w];
    std::vector<unsigned> h = build_chain(B, 8192, 4242+w);
    unsigned* d=(unsigned*)nvl_alloc(1,B);
    CK(cudaMemcpy(d,h.data(),B,cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());

    double warm=1e30, cold=1e30;
    long long hh[2];
    for (int k=0;k<4;++k) {
      // ---- warm: GPU1 自己把 WS 读一遍进它的 L2
      CK(cudaSetDevice(1));
      k_touch<<<grid,blk>>>((uint4*)d, B/16, sink1);
      CK(cudaDeviceSynchronize());
      CK(cudaSetDevice(0));
      // GPU0 侧冲刷(双保险)
      k_touch<<<grid,blk>>>(flush0, FLUSH/16, sink0); CK(cudaDeviceSynchronize());
      k_chase<1><<<1,1>>>(d,ITERS,d_out); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hh,d_out,sizeof hh,cudaMemcpyDeviceToHost));
      warm=fmin(warm,(double)hh[0]/ITERS);

      // ---- cold: GPU1 读 256MB 无关 buffer 冲掉自己 L2
      CK(cudaSetDevice(1));
      k_touch<<<grid,blk>>>(flush1, FLUSH/16, sink1);
      CK(cudaDeviceSynchronize());
      CK(cudaSetDevice(0));
      k_touch<<<grid,blk>>>(flush0, FLUSH/16, sink0); CK(cudaDeviceSynchronize());
      k_chase<1><<<1,1>>>(d,ITERS,d_out); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hh,d_out,sizeof hh,cudaMemcpyDeviceToHost));
      cold=fmin(cold,(double)hh[0]/ITERS);
    }
    printf("| %5zu MB | %10.1f | %10.1f | %14.1f | %7.1f | %7.1f |\n",
           B>>20, warm, cold, cold-warm, cyc2ns(warm,GHZ), cyc2ns(cold,GHZ));
    fflush(stdout);
    CK(cudaFree(d)); CK(cudaSetDevice(0));
  }

  hdr("B) 对照: 本地(GPU0) pointer-chase 在同样 warm/cold 下的差 —— 验证冲刷手法有效");
  printf("| WS | local warm (cyc) | local cold (cyc) | diff |\n|---|---|---|---|\n");
  for (int w=0;w<NW;++w) {
    size_t B=ws[w];
    if (B > (64ull<<20)) continue;
    std::vector<unsigned> h=build_chain(B,8192,777+w);
    unsigned* d=(unsigned*)nvl_alloc(0,B);
    CK(cudaMemcpy(d,h.data(),B,cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize()); CK(cudaSetDevice(0));
    double warm=1e30, cold=1e30; long long hh[2];
    for (int k=0;k<4;++k) {
      k_touch<<<grid,blk>>>((uint4*)d,B/16,sink0); CK(cudaDeviceSynchronize());
      k_chase<0><<<1,1>>>(d,ITERS,d_out); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hh,d_out,sizeof hh,cudaMemcpyDeviceToHost)); warm=fmin(warm,(double)hh[0]/ITERS);
      k_touch<<<grid,blk>>>(flush0,FLUSH/16,sink0); CK(cudaDeviceSynchronize());
      k_chase<0><<<1,1>>>(d,ITERS,d_out); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hh,d_out,sizeof hh,cudaMemcpyDeviceToHost)); cold=fmin(cold,(double)hh[0]/ITERS);
    }
    printf("| %5zu MB | %10.1f | %10.1f | %8.1f |\n", B>>20, warm, cold, cold-warm);
    fflush(stdout);
    CK(cudaFree(d)); CK(cudaSetDevice(0));
  }

  hdr("C) 带宽视角: 远端流式读(volatile, 绕 GPU0 缓存) 在 GPU1 warm/cold 下");
  printf("| GPU1 WS | warm GB/s | cold GB/s | 提升 |\n|---|---|---|---|\n");
  for (int w=0;w<NW;++w) {
    size_t B=ws[w];
    unsigned* d=(unsigned*)nvl_alloc(1,B);
    CK(cudaSetDevice(0));
    size_t n=B/16;
    double gw, gc;
    { // warm
      double best=1e30;
      for (int k=0;k<3;++k){
        CK(cudaSetDevice(1)); k_touch<<<grid,blk>>>((uint4*)d,n,sink1); CK(cudaDeviceSynchronize());
        CK(cudaSetDevice(0));
        Timer t; t.start(); k_stream_v<<<grid,blk>>>((uint4*)d,n,sink0); best=fmin(best,t.stop_ms());
      }
      gw=gbps((double)B,best);
    }
    { // cold
      double best=1e30;
      for (int k=0;k<3;++k){
        CK(cudaSetDevice(1)); k_touch<<<grid,blk>>>(flush1,FLUSH/16,sink1); CK(cudaDeviceSynchronize());
        CK(cudaSetDevice(0));
        Timer t; t.start(); k_stream_v<<<grid,blk>>>((uint4*)d,n,sink0); best=fmin(best,t.stop_ms());
      }
      gc=gbps((double)B,best);
    }
    printf("| %5zu MB | %9.1f | %9.1f | %5.2fx |\n", B>>20, gw, gc, gc>0?gw/gc:0);
    fflush(stdout);
    CK(cudaFree(d)); CK(cudaSetDevice(0));
  }

  printf("\n[done]\n");
  return 0;
}
