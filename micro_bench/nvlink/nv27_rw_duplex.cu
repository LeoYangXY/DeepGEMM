// ============================================================================
// nv27_rw_duplex —— NVLink 全双工验证与读写混合
//
// (a) GPU0 只写 GPU1 / 只读 GPU1 / 同时读写 GPU1
//     若「同时读写」总带宽 ~= 2x 单向 -> 链路全双工, 读写各占一个物理方向
// (b) 双向对打: GPU0 写 GPU1 的同时 GPU1 写 GPU0
// (c) 读写比例扫描 0/25/50/75/100%, 看混合惩罚
// 全程用 nvl_counters 验证 Tx/Rx 方向的实际字节分配。
//
// 注意 nv26 已知: LOAD 受 outstanding 限制 solo 只有 285 GB/s(8warp/78blk),
// 所以本实验读侧要开足并发(更大 grid)才能逼近链路上限, 否则会低估读带宽,
// 进而低估"双工总和"。这里读写都用 grid = 4*SM 打满。
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"
#include <algorithm>
#include <vector>

__global__ void k_wr(uint4* __restrict__ dst, size_t n, int rep) {
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += s) st128(dst + i, v);
}

__global__ void k_rd(const uint4* __restrict__ src, size_t n, int rep,
                     uint4* sink) {
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += s) {
      uint4 v = ld128_v(src + i);
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  if (acc.x == 0xdeadbeefu && acc.y == 0x12345678u) *sink = acc;
}

// 读写混合: 按 block 划分角色, wrFrac_x256 决定多少比例的 block 做写
// 读和写打不同 buffer, 避免读到自己刚写的数据(会命中 L2)
__global__ void k_mix(uint4* __restrict__ wbuf, const uint4* __restrict__ rbuf,
                      size_t n, int rep, int wrBlocks, uint4* sink) {
  bool isWriter = (int)blockIdx.x < wrBlocks;
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  if (isWriter) {
    uint4 v = make_uint4(1, 2, 3, 4);
    for (int r = 0; r < rep; ++r)
      for (size_t i = i0; i < n; i += s) st128(wbuf + i, v);
  } else {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int r = 0; r < rep; ++r)
      for (size_t i = i0; i < n; i += s) {
        uint4 v = ld128_v(rbuf + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
      }
    if (acc.x == 0xdeadbeefu && acc.y == 0x12345678u) *sink = acc;
  }
}

int main() {
  NvlEnv env = nvl_init(2);
  printf("# nv27_rw_duplex — %s x%d, %d SM, %.3f GHz\n", env.name, env.ndev,
         env.sm, env.clkGHz);
  nvl_enable_peers(env.ndev);

  const size_t BUF = 256ull << 20;
  const size_t N = BUF / sizeof(uint4);
  int blk = 256, grid = env.sm * 4;   // 打满并发, 让读也能逼近链路上限
  const int REP = 12;                 // 保证 >= 50ms

  // GPU1 上两块 buffer: 一块给 GPU0 写, 一块给 GPU0 读
  void* g1_wr = nvl_alloc(1, BUF);
  void* g1_rd = nvl_alloc(1, BUF, 0x5a);
  // GPU0 上的 buffer: 给 GPU1 写(双向对打用) + sink
  void* g0_wr = nvl_alloc(0, BUF);
  void* g0_local = nvl_alloc(0, BUF, 0x3c);
  CK(cudaSetDevice(0));
  uint4* sink0 = (uint4*)nvl_alloc(0, 4096);
  CK(cudaSetDevice(1));
  uint4* sink1 = (uint4*)nvl_alloc(1, 4096);
  CK(cudaSetDevice(0));

  cudaStream_t s0a, s0b, s1a;
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&s0a));
  CK(cudaStreamCreate(&s0b));
  CK(cudaSetDevice(1));
  CK(cudaStreamCreate(&s1a));
  CK(cudaSetDevice(0));

  cudaEvent_t ea0, eb0, ea1, eb1, ea2, eb2;
  CK(cudaSetDevice(0));
  CK(cudaEventCreate(&ea0)); CK(cudaEventCreate(&eb0));
  CK(cudaEventCreate(&ea1)); CK(cudaEventCreate(&eb1));
  CK(cudaSetDevice(1));
  CK(cudaEventCreate(&ea2)); CK(cudaEventCreate(&eb2));
  CK(cudaSetDevice(0));

  const double PASS = (double)BUF * REP;

  // ============================================================ (a) 单向 vs 双工
  hdr("a) 单向 vs 同时读写 (GPU0 <-> GPU1)");

  // -- 只写
  CK(cudaSetDevice(0));
  k_wr<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, N, REP);
  CK(cudaStreamSynchronize(s0a));
  double bw_w = 0, bw_r = 0;
  {
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
      CK(cudaEventRecord(ea0, s0a));
      k_wr<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, N, REP);
      CK(cudaEventRecord(eb0, s0a));
      CK(cudaEventSynchronize(eb0));
      float ms; CK(cudaEventElapsedTime(&ms, ea0, eb0));
      best = std::min(best, (double)ms);
    }
    bw_w = gbps(PASS, best);
  }
  // -- 只读
  {
    k_rd<<<grid, blk, 0, s0a>>>((const uint4*)g1_rd, N, REP, sink0);
    CK(cudaStreamSynchronize(s0a));
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
      CK(cudaEventRecord(ea0, s0a));
      k_rd<<<grid, blk, 0, s0a>>>((const uint4*)g1_rd, N, REP, sink0);
      CK(cudaEventRecord(eb0, s0a));
      CK(cudaEventSynchronize(eb0));
      float ms; CK(cudaEventElapsedTime(&ms, ea0, eb0));
      best = std::min(best, (double)ms);
    }
    bw_r = gbps(PASS, best);
  }
  // -- 同时读写(两条 stream, 都在 GPU0)
  // 【重要】读侧必须开满 grid: nv26 已证明 LOAD 受 outstanding 请求数限制,
  // 如果给读只分一半 SM, 读带宽会被并发度而非链路卡住, 导致低估双工总和。
  // 这里读写各开满 grid(超订 SM 也无妨, 我们要的是各自尽量灌满自己的方向)。
  double bw_rw_w = 0, bw_rw_r = 0, bw_rw_agg = 0;
  {
    // warmup
    k_wr<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, N, REP);
    k_rd<<<grid, blk, 0, s0b>>>((const uint4*)g1_rd, N, REP, sink0);
    CK(cudaDeviceSynchronize());
    double bestagg = 0;
    for (int t = 0; t < 3; ++t) {
      CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(ea0, s0a));
      CK(cudaEventRecord(ea1, s0b));
      k_wr<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, N, REP);
      k_rd<<<grid, blk, 0, s0b>>>((const uint4*)g1_rd, N, REP, sink0);
      CK(cudaEventRecord(eb0, s0a));
      CK(cudaEventRecord(eb1, s0b));
      CK(cudaDeviceSynchronize());
      float mw, mr;
      CK(cudaEventElapsedTime(&mw, ea0, eb0));
      CK(cudaEventElapsedTime(&mr, ea1, eb1));
      // 两条流重叠执行, 聚合带宽用较长的那个作为墙钟窗口
      double agg = gbps(PASS * 2, std::max(mw, mr));
      if (agg > bestagg) {
        bestagg = agg; bw_rw_w = gbps(PASS, mw); bw_rw_r = gbps(PASS, mr);
      }
    }
    bw_rw_agg = bestagg;
  }

  printf("| 场景 | 写 GB/s | 读 GB/s | 总 GB/s | 相对单向 |\n|---|---|---|---|---|\n");
  printf("| 只写 GPU0->GPU1 | %.1f | - | %.1f | 1.00x |\n", bw_w, bw_w);
  printf("| 只读 GPU0<-GPU1 | - | %.1f | %.1f | %.2fx |\n", bw_r, bw_r, bw_r / bw_w);
  printf("| 同时读+写 | %.1f | %.1f | %.1f | %.2fx |\n", bw_rw_w, bw_rw_r,
         bw_rw_agg, bw_rw_agg / bw_w);
  printf("\n判据: 同时读写总带宽 / 只写带宽 = %.2f  (接近 2.0 => 全双工)\n",
         bw_rw_agg / bw_w);

  // 计数器验证方向分配
  hdr("a2) 硬件计数器验证方向 (GPU0 视角, Tx=发出 Rx=收到)");
  auto count_run = [&](const char* tag, double payload, auto&& fn) {
    CK(cudaDeviceSynchronize());
    NvlCounters c0 = nvl_read_counters(0);
    fn();
    CK(cudaDeviceSynchronize());
    NvlCounters c1 = nvl_read_counters(0);
    nvl_report(tag, payload, nvl_diff(c0, c1));
  };
  count_run("只写", PASS * 2, [&] {
    for (int i = 0; i < 2; ++i) k_wr<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, N, REP);
  });
  count_run("只读", PASS * 2, [&] {
    for (int i = 0; i < 2; ++i)
      k_rd<<<grid, blk, 0, s0a>>>((const uint4*)g1_rd, N, REP, sink0);
  });
  count_run("同时读写", PASS * 4, [&] {
    for (int i = 0; i < 2; ++i) {
      k_wr<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, N, REP);
      k_rd<<<grid, blk, 0, s0b>>>((const uint4*)g1_rd, N, REP, sink0);
    }
  });
  printf("(只写: dataTx 应 >> dataRx；只读: dataRx 应 >> dataTx；\n");
  printf(" 同时读写: 两者应都很大 => 两个物理方向同时在跑)\n");

  // ============================================================ (b) 双向对打
  hdr("b) 双向对打: GPU0 写 GPU1 同时 GPU1 写 GPU0");
  double bi_01 = 0, bi_10 = 0, bi_agg = 0;
  {
    // warmup
    CK(cudaSetDevice(0));
    k_wr<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, N, REP);
    CK(cudaSetDevice(1));
    k_wr<<<grid, blk, 0, s1a>>>((uint4*)g0_wr, N, REP);
    CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());

    double bestagg = 0;
    for (int t = 0; t < 3; ++t) {
      CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
      CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());
      // 先都 launch 出去
      CK(cudaSetDevice(0));
      CK(cudaEventRecord(ea0, s0a));
      k_wr<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, N, REP);
      CK(cudaEventRecord(eb0, s0a));
      CK(cudaSetDevice(1));
      CK(cudaEventRecord(ea2, s1a));
      k_wr<<<grid, blk, 0, s1a>>>((uint4*)g0_wr, N, REP);
      CK(cudaEventRecord(eb2, s1a));
      // 统一同步
      CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
      CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());
      float m01, m10;
      CK(cudaEventElapsedTime(&m01, ea0, eb0));
      CK(cudaEventElapsedTime(&m10, ea2, eb2));
      double agg = gbps(PASS * 2, std::max(m01, m10));
      if (agg > bestagg) {
        bestagg = agg; bi_01 = gbps(PASS, m01); bi_10 = gbps(PASS, m10);
      }
    }
    bi_agg = bestagg;
    CK(cudaSetDevice(0));
  }
  printf("| 场景 | 0->1 GB/s | 1->0 GB/s | 总 GB/s | 相对单向 |\n|---|---|---|---|---|\n");
  printf("| 单向 0->1 | %.1f | - | %.1f | 1.00x |\n", bw_w, bw_w);
  printf("| 双向对打 | %.1f | %.1f | %.1f | %.2fx |\n", bi_01, bi_10, bi_agg,
         bi_agg / bw_w);
  printf("\n公平性: |0->1 - 1->0| / 均值 = %.1f%%\n",
         100.0 * fabs(bi_01 - bi_10) / ((bi_01 + bi_10) / 2));

  // ============================================================ (c) 读写比例扫描
  const int pcts[] = {0, 25, 50, 75, 100};
  double mixbw[5];
  for (int pi = 0; pi < 5; ++pi) {
    int wrB = grid * pcts[pi] / 100;
    k_mix<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, (const uint4*)g1_rd, N, REP,
                                 wrB, sink0);
    CK(cudaStreamSynchronize(s0a));
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
      CK(cudaEventRecord(ea0, s0a));
      k_mix<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, (const uint4*)g1_rd, N, REP,
                                   wrB, sink0);
      CK(cudaEventRecord(eb0, s0a));
      CK(cudaEventSynchronize(eb0));
      float ms; CK(cudaEventElapsedTime(&ms, ea0, eb0));
      best = std::min(best, (double)ms);
    }
    // 每个 block 都 grid-stride 覆盖自己那份, 总逻辑字节固定 = PASS
    mixbw[pi] = gbps(PASS, best);
  }
  // 两趟输出, 保证归一化基准 mixbw[4](纯写) 已算好
  printf("| 写占比 | 写 block | 读 block | 总有效 GB/s | vs 纯写 | vs 纯读 |\n");
  printf("|---|---|---|---|---|---|\n");
  for (int pi = 0; pi < 5; ++pi)
    printf("| %d%% | %d | %d | %.1f | %.2fx | %.2fx |\n", pcts[pi],
           grid * pcts[pi] / 100, grid - grid * pcts[pi] / 100, mixbw[pi],
           mixbw[pi] / mixbw[4], mixbw[pi] / mixbw[0]);
  printf("\n注: 单 kernel 内混合, 写 block 和读 block 各自 grid-stride 遍历,\n");
  printf("    总搬运字节固定 = %.0f MB, 所以这里直接可比。\n", PASS / 1e6);

  // 混合的计数器验证
  hdr("c2) 50/50 混合的方向字节分配");
  count_run("50%写50%读", PASS * 2, [&] {
    for (int i = 0; i < 2; ++i)
      k_mix<<<grid, blk, 0, s0a>>>((uint4*)g1_wr, (const uint4*)g1_rd, N, REP,
                                   grid / 2, sink0);
  });

  printf("\n[done]\n");
  return 0;
}
