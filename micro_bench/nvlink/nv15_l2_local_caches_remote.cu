// ============================================================================
// nv15_l2_local_caches_remote —— 本地 L2/L1 到底会不会缓存远端(peer)数据?
//
// 三条独立证据链:
//  (A) 延迟曲线: buffer 在 GPU1, GPU0 做 pointer-chase, WS 从 4KB 扫到 256MB。
//      如果小 WS 明显更快 -> 本地缓存生效; 曲线平坦 -> 远端访问 uncached。
//      同时跑 ld.global(可缓存) / ld.global.cv(强制 miss) / ld.global.nc 三种。
//  (B) 带宽曲线: 多线程流式读同一个远端 WS, WS 小则命中本地 cache -> 带宽 >> 370GB/s
//  (C) *决定性证据* NVLink 硬件字节计数器:
//      取一个 W 字节的远端 buffer, 用 GPU0 反复读 N 遍。
//      若 wire 上 GPU0 的 dataRx ≈ N*W  -> 每遍都真的走了 NVLink, 没被本地缓存
//      若 dataRx ≈ 1*W                  -> 只拉了一遍, 后续命中本地 L2
//      对 W << L2(60MB) 和 W >> L2 各做一次, 对比比值。
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"
#include <vector>
#include <algorithm>
#include <random>

__device__ __forceinline__ unsigned ld_g(const unsigned* p) {
  unsigned v; asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned ld_cv_(const unsigned* p) {
  unsigned v; asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned ld_nc_(const unsigned* p) {
  unsigned v; asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

template <int MOD>
__global__ void k_chase(const unsigned* chain, int iters, long long* out) {
  unsigned cur = 0;
  for (int i = 0; i < 512; ++i)
    cur = (MOD==0)?ld_g(chain+cur):((MOD==1)?ld_cv_(chain+cur):ld_nc_(chain+cur));
  long long t0 = clk();
  #pragma unroll 8
  for (int i = 0; i < iters; ++i)
    cur = (MOD==0)?ld_g(chain+cur):((MOD==1)?ld_cv_(chain+cur):ld_nc_(chain+cur));
  long long t1 = clk();
  out[0] = t1 - t0; out[1] = cur;
}

// 流式读: 全 grid 反复扫同一个 WS
__global__ void k_stream(const uint4* __restrict__ src, size_t n, int reps,
                         uint4* sink) {
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s  = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0,0,0,0);
  for (int r = 0; r < reps; ++r)
    for (size_t i = i0; i < n; i += s) {
      uint4 v = ld128(src + i);
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  if (acc.x == 0xdeadbeefu && acc.y == 0xfeedfaceu) *sink = acc;
}
// volatile 版(强制每次都到 memory)
__global__ void k_stream_v(const uint4* __restrict__ src, size_t n, int reps,
                           uint4* sink) {
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s  = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0,0,0,0);
  for (int r = 0; r < reps; ++r)
    for (size_t i = i0; i < n; i += s) {
      uint4 v = ld128_v(src + i);
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  if (acc.x == 0xdeadbeefu && acc.y == 0xfeedfaceu) *sink = acc;
}

static std::vector<unsigned> build_chain(size_t bytes, size_t stride, unsigned seed) {
  size_t nnode = bytes / stride; if (nnode < 8) { nnode = 8; stride = bytes/8; }
  std::vector<unsigned> h(bytes/4, 0u);
  std::vector<unsigned> idx(nnode);
  for (size_t i=0;i<nnode;++i) idx[i] = (unsigned)(i*stride/4);
  std::vector<unsigned> perm(idx.begin()+1, idx.end());
  std::mt19937 rng(seed); std::shuffle(perm.begin(), perm.end(), rng);
  unsigned cur = idx[0];
  for (unsigned p : perm) { h[cur]=p; cur=p; }
  h[cur]=idx[0];
  return h;
}

int main() {
  NvlEnv env = nvl_init(2);
  const double GHZ = 1.98;
  printf("# nv15_l2_local_caches_remote — %s x%d, L2=60MB\n", env.name, env.ndev);
  nvl_enable_peers(env.ndev);
  CK(cudaSetDevice(0));
  long long* d_out; CK(cudaMalloc(&d_out, 64));
  uint4* sink = (uint4*)nvl_alloc(0, 1024);
  CK(cudaSetDevice(0));

  const int ITERS = 4096;
  size_t ws[] = {4ull<<10, 16ull<<10, 64ull<<10, 256ull<<10, 1ull<<20,
                 4ull<<20, 16ull<<20, 32ull<<20, 48ull<<20, 64ull<<20,
                 96ull<<20, 128ull<<20, 256ull<<20};
  const int NW = sizeof(ws)/sizeof(ws[0]);

  // ================================================== A) 延迟 vs WS
  hdr("A) GPU0 pointer-chase GPU1 的 buffer: 延迟 vs working set");
  printf("| WS | remote ld.global (cyc) | remote .cv (cyc) | remote .nc (cyc) | local ld.global (cyc) |\n");
  printf("|---|---|---|---|---|\n");
  double remA[NW], remCV[NW], remNC[NW], locA[NW];
  for (int w = 0; w < NW; ++w) {
    size_t B = ws[w];
    size_t stride = (B <= (256ull<<10)) ? 256 : 8192;
    std::vector<unsigned> h = build_chain(B, stride, 999+w);
    long long hh[2];
    for (int side = 0; side < 2; ++side) {
      int dev = side ? 0 : 1;
      unsigned* d = (unsigned*)nvl_alloc(dev, B);
      CK(cudaMemcpy(d, h.data(), B, cudaMemcpyHostToDevice));
      CK(cudaDeviceSynchronize()); CK(cudaSetDevice(0));
      double b0=1e30,b1=1e30,b2=1e30;
      for (int k=0;k<3;++k){
        k_chase<0><<<1,1>>>(d,ITERS,d_out); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hh,d_out,sizeof hh,cudaMemcpyDeviceToHost)); b0=fmin(b0,(double)hh[0]/ITERS);
        if (!side) {
          k_chase<1><<<1,1>>>(d,ITERS,d_out); CK(cudaDeviceSynchronize());
          CK(cudaMemcpy(hh,d_out,sizeof hh,cudaMemcpyDeviceToHost)); b1=fmin(b1,(double)hh[0]/ITERS);
          k_chase<2><<<1,1>>>(d,ITERS,d_out); CK(cudaDeviceSynchronize());
          CK(cudaMemcpy(hh,d_out,sizeof hh,cudaMemcpyDeviceToHost)); b2=fmin(b2,(double)hh[0]/ITERS);
        }
      }
      if (!side) { remA[w]=b0; remCV[w]=b1; remNC[w]=b2; } else locA[w]=b0;
      CK(cudaFree(d)); CK(cudaSetDevice(0));
    }
    char wsn[32];
    if (ws[w] < (1u<<20)) snprintf(wsn,32,"%zu KB", ws[w]>>10);
    else snprintf(wsn,32,"%zu MB", ws[w]>>20);
    printf("| %-8s | %6.1f | %6.1f | %6.1f | %6.1f |\n", wsn, remA[w], remCV[w], remNC[w], locA[w]);
    fflush(stdout);
  }

  // ================================================== B) 带宽 vs WS
  hdr("B) GPU0 流式读 GPU1 buffer: 有效带宽 vs working set (reps 使总量恒定)");
  printf("| WS | ld.global GB/s | ld.volatile GB/s | ratio |\n|---|---|---|---|\n");
  int grid = env.sm*4, blk = 256;
  for (int w = 0; w < NW; ++w) {
    size_t B = ws[w];
    size_t n = B/16;
    // 让每次 kernel 至少搬 256MB 逻辑量
    int reps = (int)((256ull<<20)/B); if (reps<1) reps=1; if (reps>4096) reps=4096;
    unsigned* d = (unsigned*)nvl_alloc(1, B);
    CK(cudaSetDevice(0));
    double t1 = bench_ms([&]{ k_stream<<<grid,blk>>>((uint4*)d,n,reps,sink); }, 3);
    double t2 = bench_ms([&]{ k_stream_v<<<grid,blk>>>((uint4*)d,n,reps,sink); }, 3);
    double g1 = gbps((double)B*reps, t1), g2 = gbps((double)B*reps, t2);
    char wsn[32];
    if (B<(1u<<20)) snprintf(wsn,32,"%zu KB",B>>10); else snprintf(wsn,32,"%zu MB",B>>20);
    printf("| %-8s | %10.1f | %10.1f | %5.2fx |\n", wsn, g1, g2, g2>0?g1/g2:0);
    fflush(stdout);
    CK(cudaFree(d)); CK(cudaSetDevice(0));
  }

  // ================================================== C) 计数器决定性验证
  hdr("C) [决定性] NVLink 硬件计数器: 反复读 N 遍远端 WS, 线上 Rx 字节是 1x 还是 Nx?");
  printf("如果 dataRx/(1*WS) ~= N  -> 每遍都走 NVLink, 未被本地缓存\n");
  printf("如果 dataRx/(1*WS) ~= 1  -> 只拉了一遍, 后续命中 GPU0 本地 L2\n\n");
  printf("| WS | N(reps) | 逻辑读取量 MB | GPU0 dataRx MB | dataRx/WS | 判定 |\n|---|---|---|---|---|---|\n");
  struct CC { size_t B; int reps; };
  CC cc[] = {{4ull<<20, 64}, {16ull<<20, 32}, {32ull<<20, 16},
             {48ull<<20, 12}, {128ull<<20, 8}, {256ull<<20, 4}};
  for (auto& c : cc) {
    unsigned* d = (unsigned*)nvl_alloc(1, c.B);
    CK(cudaSetDevice(0));
    size_t n = c.B/16;
    // warmup(把数据拉一遍进本地 cache, 使后续测量不含 cold miss)
    k_stream<<<grid,blk>>>((uint4*)d,n,1,sink); CK(cudaDeviceSynchronize());
    NvlCounters a = nvl_read_counters(0);
    k_stream<<<grid,blk>>>((uint4*)d,n,c.reps,sink);
    CK(cudaDeviceSynchronize());
    NvlCounters b = nvl_read_counters(0);
    NvlCounters dd = nvl_diff(a,b);
    double logical = (double)c.B*c.reps;
    double ratio = (double)dd.dataRx/(double)c.B;
    const char* verdict = (ratio < c.reps*0.5) ? "**被本地缓存**" : "未缓存(每遍上线)";
    char wsn[32]; snprintf(wsn,32,"%zu MB", c.B>>20);
    printf("| %-7s | %3d | %9.0f | %10.1f | %6.2f | %s |\n", wsn, c.reps,
           logical/1e6, dd.dataRx/1e6, ratio, verdict);
    fflush(stdout);
    CK(cudaFree(d)); CK(cudaSetDevice(0));
  }

  hdr("C2) 同上但用 ld.volatile (应当强制每遍都上线, 作为阳性对照)");
  printf("| WS | N | GPU0 dataRx MB | dataRx/WS |\n|---|---|---|---|\n");
  for (auto& c : cc) {
    if (c.B > (48ull<<20)) continue;  // volatile 慢, 只跑小的
    unsigned* d = (unsigned*)nvl_alloc(1, c.B);
    CK(cudaSetDevice(0));
    size_t n = c.B/16;
    k_stream_v<<<grid,blk>>>((uint4*)d,n,1,sink); CK(cudaDeviceSynchronize());
    NvlCounters a = nvl_read_counters(0);
    k_stream_v<<<grid,blk>>>((uint4*)d,n,c.reps,sink); CK(cudaDeviceSynchronize());
    NvlCounters dd = nvl_diff(a, nvl_read_counters(0));
    char wsn[32]; snprintf(wsn,32,"%zu MB", c.B>>20);
    printf("| %-7s | %3d | %10.1f | %6.2f |\n", wsn, c.reps, dd.dataRx/1e6,
           (double)dd.dataRx/(double)c.B);
    fflush(stdout);
    CK(cudaFree(d)); CK(cudaSetDevice(0));
  }

  printf("\n[done]\n");
  return 0;
}
