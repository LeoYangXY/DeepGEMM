// ============================================================================
// nv28_multi_peer —— 一对多带宽分配 (GPU0 同时写 1/2/3 个 peer)
//
// 核心问题: GPU0 的 18 条 link 是「按目标静态划分」还是「全部聚合、
//           任意目标都能用满」?
//   - 若单目标就能到 370 且 3 目标总和还是 ~370 -> 出口共享聚合,
//     370 是 GPU0 的注入上限(与目标数无关)
//   - 若 3 目标总和 ~= 3x 某值 且 > 单目标 -> link 按目标静态划分
//
// 两种发起方式都测:
//   (A) 多 stream: 每个目标一个 stream, 各自独立 kernel
//   (B) 单 kernel 多目标: 一个 kernel 内按 block 分组打不同目标
//       (排除 stream 调度/SM 分配的干扰)
// 再测公平性: 各目标份额的标准差。
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"
#include <algorithm>
#include <cmath>

__global__ void k_wr(uint4* __restrict__ dst, size_t n, int rep) {
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += s) st128(dst + i, v);
}

// 单 kernel 打多个目标: block 按 blockIdx % npeer 分组
__global__ void k_wr_multi(uint4* __restrict__ d0, uint4* __restrict__ d1,
                           uint4* __restrict__ d2, int npeer, size_t n,
                           int rep) {
  int grp = blockIdx.x % npeer;
  uint4* dst = (grp == 0) ? d0 : (grp == 1 ? d1 : d2);
  // 该组内的第几个 block
  size_t bIn = blockIdx.x / npeer;
  size_t nb = (gridDim.x + npeer - 1 - grp) / npeer;  // 该组的 block 总数
  size_t i0 = bIn * (size_t)blockDim.x + threadIdx.x;
  size_t s = nb * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += s) st128(dst + i, v);
}

int main() {
  NvlEnv env = nvl_init(4);
  printf("# nv28_multi_peer — %s x%d, %d SM, %.3f GHz\n", env.name, env.ndev,
         env.sm, env.clkGHz);
  nvl_enable_peers(env.ndev);

  const size_t BUF = 256ull << 20;
  const size_t N = BUF / sizeof(uint4);
  int blk = 256, grid = env.sm * 4;
  const int REP = 12;
  const double PASS = (double)BUF * REP;

  // 在 GPU1/2/3 上各开一块
  uint4* peer[3];
  for (int d = 1; d < 4; ++d) peer[d - 1] = (uint4*)nvl_alloc(d, BUF);
  CK(cudaSetDevice(0));

  cudaStream_t st[3];
  cudaEvent_t ea[3], eb[3];
  for (int i = 0; i < 3; ++i) {
    CK(cudaStreamCreate(&st[i]));
    CK(cudaEventCreate(&ea[i]));
    CK(cudaEventCreate(&eb[i]));
  }

  // ============================================================ A) 多 stream
  hdr("A) 多 stream 一对多写 (每个目标一个 stream, 各自 grid=4*SM)");
  printf("| 目标数 | GPU1 | GPU2 | GPU3 | 各目标和 | 墙钟聚合 | vs 单目标 | 份额std%% |\n");
  printf("|---|---|---|---|---|---|---|---|\n");
  double single_bw = 0;
  for (int np = 1; np <= 3; ++np) {
    // warmup
    for (int i = 0; i < np; ++i)
      k_wr<<<grid, blk, 0, st[i]>>>(peer[i], N, REP);
    CK(cudaDeviceSynchronize());

    double bw[3] = {0, 0, 0}, bestagg = 0, bestwall = 0;
    for (int t = 0; t < 3; ++t) {
      CK(cudaDeviceSynchronize());
      Timer wall;
      wall.start(0);
      // 先都 launch 出去
      for (int i = 0; i < np; ++i) CK(cudaEventRecord(ea[i], st[i]));
      for (int i = 0; i < np; ++i)
        k_wr<<<grid, blk, 0, st[i]>>>(peer[i], N, REP);
      for (int i = 0; i < np; ++i) CK(cudaEventRecord(eb[i], st[i]));
      double wms = wall.stop_ms(0);
      CK(cudaDeviceSynchronize());
      double sum = 0, mx = 0;
      double tmp[3];
      for (int i = 0; i < np; ++i) {
        float ms; CK(cudaEventElapsedTime(&ms, ea[i], eb[i]));
        tmp[i] = gbps(PASS, ms);
        sum += tmp[i];
        mx = std::max(mx, (double)ms);
      }
      if (sum > bestagg) {
        bestagg = sum; bestwall = gbps(PASS * np, mx);
        for (int i = 0; i < np; ++i) bw[i] = tmp[i];
        (void)wms;
      }
    }
    if (np == 1) single_bw = bestagg;
    // 份额标准差
    double mean = bestagg / np, var = 0;
    for (int i = 0; i < np; ++i) var += (bw[i] - mean) * (bw[i] - mean);
    double sd = np > 1 ? sqrt(var / np) : 0;

    printf("| %d ", np);
    for (int i = 0; i < 3; ++i) {
      if (i < np) printf("| %.1f ", bw[i]); else printf("| - ");
    }
    printf("| %.1f | %.1f | %.2fx | %.1f%% |\n", bestagg, bestwall,
           bestagg / single_bw, 100.0 * sd / mean);
  }

  // ============================================================ B) 单 kernel
  hdr("B) 单 kernel 多目标 (block 轮转分配给不同目标, 排除 stream 调度干扰)");
  printf("| 目标数 | 总有效 GB/s | vs 单目标 |\n|---|---|---|\n");
  double sk1 = 0;
  for (int np = 1; np <= 3; ++np) {
    k_wr_multi<<<grid, blk, 0, st[0]>>>(peer[0], peer[1], peer[2], np, N, REP);
    CK(cudaStreamSynchronize(st[0]));
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
      CK(cudaEventRecord(ea[0], st[0]));
      k_wr_multi<<<grid, blk, 0, st[0]>>>(peer[0], peer[1], peer[2], np, N, REP);
      CK(cudaEventRecord(eb[0], st[0]));
      CK(cudaEventSynchronize(eb[0]));
      float ms; CK(cudaEventElapsedTime(&ms, ea[0], eb[0]));
      best = std::min(best, (double)ms);
    }
    // 每个目标都被完整写了 N 个元素 x REP 轮 => 总字节 = PASS * np
    double bw = gbps(PASS * np, best);
    if (np == 1) sk1 = bw;
    printf("| %d | %.1f | %.2fx |\n", np, bw, bw / sk1);
  }

  // ============================================================ C) 计数器
  hdr("C) 硬件计数器: GPU0 出口总字节 vs 目标数");
  for (int np = 1; np <= 3; ++np) {
    CK(cudaDeviceSynchronize());
    NvlCounters c0 = nvl_read_counters(0);
    for (int i = 0; i < np; ++i) k_wr<<<grid, blk, 0, st[i]>>>(peer[i], N, REP);
    CK(cudaDeviceSynchronize());
    NvlCounters c1 = nvl_read_counters(0);
    char tag[32];
    snprintf(tag, sizeof tag, "%d 个目标", np);
    nvl_report(tag, PASS * np, nvl_diff(c0, c1));
  }
  printf("(dataTx 应 = np * %.0f MB; 若 GPU0 出口聚合共享, 时间会线性变长)\n",
         PASS / 1e6);

  // ============================================================ D) 判读
  hdr("D) 判读");
  printf("单目标 %.1f GB/s, 3 目标各流之和 (见 A 表最后一行)\n", single_bw);
  printf("若 3 目标之和 ~= 单目标 -> GPU0 出口是共享聚合的, "
         "%.0f GB/s 是全卡注入上限\n", single_bw);
  printf("若 3 目标之和 ~= 3x 单目标 -> 18 条 link 按目标静态划分\n");

  printf("\n[done]\n");
  return 0;
}
