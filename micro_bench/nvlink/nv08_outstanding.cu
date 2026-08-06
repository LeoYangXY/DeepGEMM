// ============================================================================
// nv08_outstanding —— 每线程未完成(in-flight)请求数扫描
//
// 核心问题: 一个线程能同时挂起多少个远端 NVLink 访存? 拐点在哪?
//
// 方法:
//   一个 block (1 warp 或 N warp), 每个线程在循环体内连发 K 个**互不依赖**的
//   访存 (地址两两错开 >=128B 保证不合并成同一个事务), 用 #pragma unroll 强制
//   编译期展开成 K 条连续的 LDG/STG 指令。
//
//   store 版: st.global.v4  (fire-and-forget, 由 LSU write buffer / MSHR 限制)
//   load  版: ld.volatile.global.v4, 返回值 XOR 累加到 acc, **循环体内不消费**,
//             只在 kernel 末尾写回 sink。这样 K 条 LDG 之间无依赖, 可以全部
//             同时挂起; 只有当硬件槽位用尽才会 stall。
//
//   读法: 记 T(K) = 每次外层迭代的 cycle 数。
//     - 若 K 翻倍而 T(K) 基本不变 -> 这 K 个请求完全并行排队, 未打满槽位
//     - 若 T(K) 开始 ~ 线性增长于 K -> 槽位打满, 后续请求要等前面退休
//     拐点 K* = 每线程最大可挂起请求数。
//
//   对照组: 同样的 kernel 打本地 HBM。本地拐点 = LSU/MSHR 通用上限;
//           远端拐点若明显更小 -> NVLink 侧另有更紧的 credit/队列限制。
//
// 注意: 循环 trip count 固定, 总请求数 = ITER*K 归一化后比较 cycles/request。
// ============================================================================
#include "nvl_common.cuh"

#define ITER 512  // 外层循环次数(每个线程)

// ---------------------------------------------------------------- store kernel
// 每线程 K 个独立 store。地址布局: 线程 t 的第 k 个地址 =
//   base + (t * K + k) * STRIDE_U4  (STRIDE_U4=1 -> 相邻 16B, warp 内合并)
// 为了让每个线程的 K 个请求真正是 K 个独立事务, 让 k 维步长 = 32*16B = 512B,
// 即 warp 内 32 线程在同一 k 上连续(合并成 2 个 256B 事务), 不同 k 相隔一个
// warp-tile。这是最真实的 "每线程 K 个 outstanding" 形态。
template <int K>
__global__ __launch_bounds__(1024) void k_st_out(uint4* __restrict__ base,
                                                 uint4* __restrict__ sink,
                                                 long long* __restrict__ tout,
                                                 int nthr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint4 v = make_uint4(tid, 2, 3, 4);
  // 每轮 K 个 tile, 每 tile nthr 个 uint4
  uint4* p = base + tid;
  const int TILE = nthr;  // uint4 步进

  __syncthreads();
  long long t0 = clk();
#pragma unroll 1
  for (int it = 0; it < ITER; ++it) {
    uint4* q = p + (long long)it * K * TILE;
#pragma unroll
    for (int k = 0; k < K; ++k) st128(q + (long long)k * TILE, v);
  }
  long long t1 = clk();
  if (threadIdx.x == 0) tout[blockIdx.x] = t1 - t0;
  if (tid == 0x7fffffff) *sink = v;
}

// ---------------------------------------------------------------- load kernel
template <int K>
__global__ __launch_bounds__(1024) void k_ld_out(const uint4* __restrict__ base,
                                                 uint4* __restrict__ sink,
                                                 long long* __restrict__ tout,
                                                 int nthr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint4* p = base + tid;
  const int TILE = nthr;
  uint4 acc = make_uint4(0, 0, 0, 0);

  __syncthreads();
  long long t0 = clk();
#pragma unroll 1
  for (int it = 0; it < ITER; ++it) {
    const uint4* q = p + (long long)it * K * TILE;
    uint4 tmp[K];
#pragma unroll
    for (int k = 0; k < K; ++k) tmp[k] = ld128_v(q + (long long)k * TILE);
    // 消费放在发射之后 -> K 条 LDG 之间无依赖
#pragma unroll
    for (int k = 0; k < K; ++k) {
      acc.x ^= tmp[k].x; acc.y ^= tmp[k].y; acc.z ^= tmp[k].z; acc.w ^= tmp[k].w;
    }
  }
  long long t1 = clk();
  if (threadIdx.x == 0) tout[blockIdx.x] = t1 - t0;
  if (acc.x == 0xdeadbeefu && acc.y == 0xdeadu) *sink = acc;
}

// ---------------------------------------------------------------- driver
struct Res { double cyc_iter; double cyc_req; double gbs; };

template <int K, bool STORE>
Res run(uint4* buf, uint4* sink, long long* tout, int nblk, int blk,
        double clkGHz) {
  int nthr = nblk * blk;
  double best = 1e30;
  long long h[512];
  for (int rep = 0; rep < 5; ++rep) {
    CK(cudaMemset(tout, 0, sizeof(long long) * nblk));
    if (STORE) k_st_out<K><<<nblk, blk>>>(buf, sink, tout, nthr);
    else       k_ld_out<K><<<nblk, blk>>>(buf, sink, tout, nthr);
    CK(cudaDeviceSynchronize());
    CKLAST();
    CK(cudaMemcpy(h, tout, sizeof(long long) * nblk, cudaMemcpyDeviceToHost));
    double mx = 0;
    for (int b = 0; b < nblk; ++b) mx = fmax(mx, (double)h[b]);
    if (rep > 0 && mx < best) best = mx;  // 丢掉第一次(warmup)
  }
  Res r;
  r.cyc_iter = best / ITER;             // 每轮(K 个请求)的 cycle
  r.cyc_req  = best / ((double)ITER * K);
  double ns = cyc2ns(best, clkGHz);
  double bytes = (double)ITER * K * nthr * 16.0;
  r.gbs = bytes / ns;                   // bytes/ns == GB/s
  return r;
}

template <bool STORE>
void sweep(const char* title, uint4* buf, uint4* sink, long long* tout,
           int nblk, int blk, double clkGHz) {
  printf("\n--- %s  (grid=%d, block=%d, %d thr) ---\n", title, nblk, blk,
         nblk * blk);
  printf("%4s %14s %14s %12s %10s\n", "K", "cyc/iter", "cyc/req", "ns/req",
         "GB/s");
  double base_iter = 0;
#define DO(KK)                                                                 \
  {                                                                            \
    Res r = run<KK, STORE>(buf, sink, tout, nblk, blk, clkGHz);                \
    if (KK == 1) base_iter = r.cyc_iter;                                       \
    printf("%4d %14.1f %14.2f %12.3f %10.2f   %s\n", KK, r.cyc_iter,           \
           r.cyc_req, cyc2ns(r.cyc_req, clkGHz), r.gbs,                        \
           (r.cyc_iter < base_iter * KK * 0.6) ? "parallel" : "serializing");  \
  }
  DO(1) DO(2) DO(4) DO(8) DO(16) DO(32) DO(64) DO(128)
#undef DO
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv08_outstanding — %s, %d SM, SM clock %.3f GHz\n", env.name,
         env.sm, env.clkGHz);
  printf("# ITER=%d, 每线程每轮发 K 个独立 128-bit 访存 (地址步长 = nthr*16B)\n",
         ITER);

  // 缓冲要足够大: ITER * Kmax * nthr * 16B。nthr 最大 1024 -> 512*128*1024*16
  // = 1 GiB。为了省显存, warp 实验用 nthr=32 -> 32 MiB; block 实验 nthr=1024
  // 需要 1 GiB, 可接受(97GB 卡)。
  const size_t BIG = 1200ull << 20;
  uint4* local = (uint4*)nvl_alloc(0, BIG);
  uint4* remote = (uint4*)nvl_alloc(1, BIG);
  CK(cudaSetDevice(0));
  uint4* sink = (uint4*)nvl_alloc(0, 4096);
  long long* tout = (long long*)nvl_alloc(0, 4096);
  CK(cudaSetDevice(0));

  hdr("A) 单 warp (grid=1, block=32) —— 最纯净的每线程 outstanding");
  printf("[远端 NVLink GPU0->GPU1]");
  sweep<true>("STORE remote", remote, sink, tout, 1, 32, env.clkGHz);
  sweep<false>("LOAD  remote", remote, sink, tout, 1, 32, env.clkGHz);
  printf("[本地 HBM 对照]");
  sweep<true>("STORE local", local, sink, tout, 1, 32, env.clkGHz);
  sweep<false>("LOAD  local", local, sink, tout, 1, 32, env.clkGHz);

  hdr("B) 单 block 8 warp (grid=1, block=256) —— 有 warp 级并行掩盖时");
  sweep<true>("STORE remote", remote, sink, tout, 1, 256, env.clkGHz);
  sweep<false>("LOAD  remote", remote, sink, tout, 1, 256, env.clkGHz);
  sweep<false>("LOAD  local ", local, sink, tout, 1, 256, env.clkGHz);

  printf("\n[说明] cyc/iter 是每轮 K 个请求的耗时。若 K 翻倍 cyc/iter 不变 ->\n");
  printf("       K 个请求完全重叠; 若 cyc/iter 正比于 K -> 槽位打满在排队。\n");
  printf("       'parallel' 标记 = cyc/iter < K*cyc/iter(K=1)*0.6\n");
  printf("\n[done]\n");
  return 0;
}
