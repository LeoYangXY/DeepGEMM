// ============================================================================
// nv30_hbm_contend —— 远端 NVLink 流量与本地 HBM 访存的相互干扰
//
// (a) GPU1 本地 HBM 带宽随 GPU0->GPU1 的 NVLink 入流量的下降
// (b) NVLink 带宽随 GPU1 本地 HBM 压力的下降
// 输出二维表 (NVLink 强度 x 本地强度)
// 关键: 代价系数 = 每 1 GB/s NVLink 入流量吃掉 GPU1 多少 GB/s 本地带宽
//       -> 这是「远端写是否穿透到远端 HBM」的重要旁证
// (c) 发起端 GPU0 侧: 发远端访问会不会吃掉 GPU0 自己的本地带宽
//
// 强度调节: 用 grid 大小 (nv26 证明对 store 有效, warp 数对 store 无效)
// ============================================================================
#include "nvl_common.cuh"
#include <algorithm>

__global__ void k_wr(uint4* __restrict__ dst, size_t n, int rep) {
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += s) st128(dst + i, v);
}

// 本地 HBM 压力: 读+写混合(流式), buffer 远大于 60MB L2 保证真打 HBM
__global__ void k_hbm(uint4* __restrict__ a, const uint4* __restrict__ b,
                      size_t n, int rep, uint4* sink) {
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += s) {
      uint4 v = ld128_v(b + i);
      acc.x ^= v.x; acc.y ^= v.y;
      st128(a + i, v);
    }
  if (acc.x == 0xdeadbeefu && acc.y == 0x12345678u) *sink = acc;
}

int main() {
  NvlEnv env = nvl_init(2);
  printf("# nv30_hbm_contend — %s x%d, %d SM, %.3f GHz\n", env.name, env.ndev,
         env.sm, env.clkGHz);
  nvl_enable_peers(env.ndev);

  // buffer 必须 >> L2(60MB), 用 512MB 保证流式访问真的打到 HBM
  const size_t BUF = 512ull << 20;
  const size_t N = BUF / sizeof(uint4);
  int blk = 256;
  int FG = env.sm * 4;      // full grid
  const int REP_NVL = 10, REP_HBM = 10;
  const double PASS = (double)BUF;

  // GPU1: NVLink 目标 + 本地压力用的两块
  uint4* g1_nvl = (uint4*)nvl_alloc(1, BUF);
  uint4* g1_a = (uint4*)nvl_alloc(1, BUF);
  uint4* g1_b = (uint4*)nvl_alloc(1, BUF, 0x5a);
  CK(cudaSetDevice(1));
  uint4* sink1 = (uint4*)nvl_alloc(1, 4096);
  // GPU0: 本地压力两块
  uint4* g0_a = (uint4*)nvl_alloc(0, BUF);
  uint4* g0_b = (uint4*)nvl_alloc(0, BUF, 0x3c);
  CK(cudaSetDevice(0));
  uint4* sink0 = (uint4*)nvl_alloc(0, 4096);
  CK(cudaSetDevice(0));

  cudaStream_t s0, s1;
  CK(cudaSetDevice(0)); CK(cudaStreamCreate(&s0));
  CK(cudaSetDevice(1)); CK(cudaStreamCreate(&s1));
  cudaEvent_t e0a, e0b, e1a, e1b;
  CK(cudaSetDevice(0)); CK(cudaEventCreate(&e0a)); CK(cudaEventCreate(&e0b));
  CK(cudaSetDevice(1)); CK(cudaEventCreate(&e1a)); CK(cudaEventCreate(&e1b));
  CK(cudaSetDevice(0));

  auto sync2 = [&]() {
    CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(0));
  };

  // HBM kernel 每轮搬 2*BUF (读 b 写 a)
  const double HBM_BYTES = (double)BUF * 2 * REP_HBM;
  const double NVL_BYTES = (double)BUF * REP_NVL;

  // ---- 单独基线
  hdr("0) 单独基线");
  double hbm_solo = 0, nvl_solo = 0;
  {
    CK(cudaSetDevice(1));
    k_hbm<<<FG, blk, 0, s1>>>(g1_a, g1_b, N, REP_HBM, sink1);
    CK(cudaStreamSynchronize(s1));
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
      CK(cudaEventRecord(e1a, s1));
      k_hbm<<<FG, blk, 0, s1>>>(g1_a, g1_b, N, REP_HBM, sink1);
      CK(cudaEventRecord(e1b, s1));
      CK(cudaEventSynchronize(e1b));
      float ms; CK(cudaEventElapsedTime(&ms, e1a, e1b));
      best = std::min(best, (double)ms);
    }
    hbm_solo = gbps(HBM_BYTES, best);
    CK(cudaSetDevice(0));
    k_wr<<<FG, blk, 0, s0>>>(g1_nvl, N, REP_NVL);
    CK(cudaStreamSynchronize(s0));
    best = 1e30;
    for (int t = 0; t < 3; ++t) {
      CK(cudaEventRecord(e0a, s0));
      k_wr<<<FG, blk, 0, s0>>>(g1_nvl, N, REP_NVL);
      CK(cudaEventRecord(e0b, s0));
      CK(cudaEventSynchronize(e0b));
      float ms; CK(cudaEventElapsedTime(&ms, e0a, e0b));
      best = std::min(best, (double)ms);
    }
    nvl_solo = gbps(NVL_BYTES, best);
  }
  printf("GPU1 本地 HBM 读写混合 (grid=%d): %.1f GB/s\n", FG, hbm_solo);
  printf("GPU0->GPU1 NVLink 写   (grid=%d): %.1f GB/s\n", FG, nvl_solo);

  // ---- 强度档位 (grid 大小)
  const int gdiv[] = {0, 8, 4, 2, 1};  // 0 表示不跑
  const char* glbl[] = {"0(off)", "1/8", "1/4", "1/2", "full"};
  const int NG = 5;

  // 先标定各档单独跑的带宽, 作为"无干扰参照"
  hdr("1) 强度档位标定 (各自单独跑)");
  double hbm_ref[NG], nvl_ref[NG];
  printf("| 档位 | GPU1本地HBM grid | HBM GB/s | NVLink grid | NVLink GB/s |\n");
  printf("|---|---|---|---|---|\n");
  for (int g = 0; g < NG; ++g) {
    if (gdiv[g] == 0) { hbm_ref[g] = 0; nvl_ref[g] = 0;
      printf("| %s | 0 | 0.0 | 0 | 0.0 |\n", glbl[g]); continue; }
    int gg = FG / gdiv[g];
    CK(cudaSetDevice(1));
    k_hbm<<<gg, blk, 0, s1>>>(g1_a, g1_b, N, REP_HBM, sink1);
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventRecord(e1a, s1));
    k_hbm<<<gg, blk, 0, s1>>>(g1_a, g1_b, N, REP_HBM, sink1);
    CK(cudaEventRecord(e1b, s1));
    CK(cudaEventSynchronize(e1b));
    float m1; CK(cudaEventElapsedTime(&m1, e1a, e1b));
    hbm_ref[g] = gbps(HBM_BYTES, m1);

    CK(cudaSetDevice(0));
    k_wr<<<gg, blk, 0, s0>>>(g1_nvl, N, REP_NVL);
    CK(cudaStreamSynchronize(s0));
    CK(cudaEventRecord(e0a, s0));
    k_wr<<<gg, blk, 0, s0>>>(g1_nvl, N, REP_NVL);
    CK(cudaEventRecord(e0b, s0));
    CK(cudaEventSynchronize(e0b));
    float m0; CK(cudaEventElapsedTime(&m0, e0a, e0b));
    nvl_ref[g] = gbps(NVL_BYTES, m0);
    printf("| %s | %d | %.1f | %d | %.1f |\n", glbl[g], gg, hbm_ref[g], gg,
           nvl_ref[g]);
  }

  // ---- 二维扫描
  hdr("2) 二维干扰表: GPU1 本地 HBM 带宽 (GB/s)  [行=NVLink入流量强度, 列=本地强度]");
  printf("| NVL\\本地 |");
  for (int h = 1; h < NG; ++h) printf(" %s |", glbl[h]);
  printf("\n|---|---|---|---|---|\n");
  double hbmM[NG][NG], nvlM[NG][NG];
  for (int nv = 0; nv < NG; ++nv) {
    printf("| **%s** |", glbl[nv]);
    for (int h = 1; h < NG; ++h) {
      int gh = FG / gdiv[h];
      double bh = 0, bn = 0;
      sync2();
      if (gdiv[nv] == 0) {
        // 只跑本地
        CK(cudaSetDevice(1));
        CK(cudaEventRecord(e1a, s1));
        k_hbm<<<gh, blk, 0, s1>>>(g1_a, g1_b, N, REP_HBM, sink1);
        CK(cudaEventRecord(e1b, s1));
        sync2();
        float m1; CK(cudaSetDevice(1));
        CK(cudaEventElapsedTime(&m1, e1a, e1b));
        bh = gbps(HBM_BYTES, m1); bn = 0;
      } else {
        int gn = FG / gdiv[nv];
        // 先都 launch 出去, 再统一同步
        CK(cudaSetDevice(1));
        CK(cudaEventRecord(e1a, s1));
        k_hbm<<<gh, blk, 0, s1>>>(g1_a, g1_b, N, REP_HBM, sink1);
        CK(cudaEventRecord(e1b, s1));
        CK(cudaSetDevice(0));
        CK(cudaEventRecord(e0a, s0));
        k_wr<<<gn, blk, 0, s0>>>(g1_nvl, N, REP_NVL);
        CK(cudaEventRecord(e0b, s0));
        sync2();
        float m1, m0;
        CK(cudaSetDevice(1)); CK(cudaEventElapsedTime(&m1, e1a, e1b));
        CK(cudaSetDevice(0)); CK(cudaEventElapsedTime(&m0, e0a, e0b));
        bh = gbps(HBM_BYTES, m1);
        bn = gbps(NVL_BYTES, m0);
      }
      hbmM[nv][h] = bh; nvlM[nv][h] = bn;
      printf(" %.0f |", bh);
    }
    printf("\n");
  }
  printf("\n(第一行 NVL=0(off) 是无 NVLink 干扰的本地基线)\n");

  hdr("3) 二维干扰表: NVLink 带宽 (GB/s)  [行=NVLink强度, 列=GPU1本地强度]");
  printf("| NVL\\本地 |");
  for (int h = 1; h < NG; ++h) printf(" %s |", glbl[h]);
  printf("\n|---|---|---|---|---|\n");
  for (int nv = 1; nv < NG; ++nv) {
    printf("| **%s** |", glbl[nv]);
    for (int h = 1; h < NG; ++h) printf(" %.0f |", nvlM[nv][h]);
    printf("\n");
  }
  printf("\n参照: 各 NVLink 强度单独跑 = ");
  for (int g = 1; g < NG; ++g) printf("%s:%.0f  ", glbl[g], nvl_ref[g]);
  printf("\n");

  // ---- 代价系数
  hdr("4) 代价系数: 每 1 GB/s NVLink 入流量吃掉 GPU1 多少本地带宽");
  printf("固定本地强度 = full, 扫 NVLink 强度:\n");
  printf("| NVLink 强度 | NVLink 实测 GB/s | GPU1 本地 GB/s | 本地损失 GB/s | 代价系数 |\n");
  printf("|---|---|---|---|---|\n");
  double base_local = hbmM[0][NG - 1];
  for (int nv = 1; nv < NG; ++nv) {
    double loss = base_local - hbmM[nv][NG - 1];
    double coef = nvlM[nv][NG - 1] > 1 ? loss / nvlM[nv][NG - 1] : 0;
    printf("| %s | %.0f | %.0f | %.0f | %.2f |\n", glbl[nv], nvlM[nv][NG - 1],
           hbmM[nv][NG - 1], loss, coef);
  }
  printf("\n(代价系数 ~1.0 => 远端写完全穿透到 GPU1 的 HBM, 1:1 吃本地带宽;\n");
  printf("  <1.0 => 部分被 GPU1 的 L2 吸收)\n");

  // ---- (c) 发起端 GPU0 侧的代价
  hdr("5) 发起端代价: GPU0 发远端写时, 自己的本地 HBM 带宽损失多少");
  double g0_solo = 0, g0_busy = 0, nvl_when_g0busy = 0;
  {
    CK(cudaSetDevice(0));
    k_hbm<<<FG, blk, 0, s0>>>(g0_a, g0_b, N, REP_HBM, sink0);
    CK(cudaStreamSynchronize(s0));
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
      CK(cudaEventRecord(e0a, s0));
      k_hbm<<<FG, blk, 0, s0>>>(g0_a, g0_b, N, REP_HBM, sink0);
      CK(cudaEventRecord(e0b, s0));
      CK(cudaEventSynchronize(e0b));
      float ms; CK(cudaEventElapsedTime(&ms, e0a, e0b));
      best = std::min(best, (double)ms);
    }
    g0_solo = gbps(HBM_BYTES, best);

    // GPU0 本地压力 + GPU0 发远端写 (两条 stream 都在 GPU0)
    cudaStream_t s0b_;
    CK(cudaStreamCreate(&s0b_));
    cudaEvent_t ex, ey;
    CK(cudaEventCreate(&ex)); CK(cudaEventCreate(&ey));
    k_hbm<<<FG / 2, blk, 0, s0>>>(g0_a, g0_b, N, REP_HBM, sink0);
    k_wr<<<FG / 2, blk, 0, s0b_>>>(g1_nvl, N, REP_NVL);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0a, s0));
    k_hbm<<<FG / 2, blk, 0, s0>>>(g0_a, g0_b, N, REP_HBM, sink0);
    CK(cudaEventRecord(e0b, s0));
    CK(cudaEventRecord(ex, s0b_));
    k_wr<<<FG / 2, blk, 0, s0b_>>>(g1_nvl, N, REP_NVL);
    CK(cudaEventRecord(ey, s0b_));
    CK(cudaDeviceSynchronize());
    float mh, mn;
    CK(cudaEventElapsedTime(&mh, e0a, e0b));
    CK(cudaEventElapsedTime(&mn, ex, ey));
    g0_busy = gbps(HBM_BYTES, mh);
    nvl_when_g0busy = gbps(NVL_BYTES, mn);
  }
  printf("| 场景 | GPU0 本地 HBM GB/s | NVLink 出流量 GB/s |\n|---|---|---|\n");
  printf("| GPU0 只跑本地 (grid=%d) | %.1f | - |\n", FG, g0_solo);
  printf("| GPU0 本地(1/2) + 远端写(1/2) | %.1f | %.1f |\n", g0_busy,
         nvl_when_g0busy);
  printf("\n发起端本地带宽损失 = %.1f GB/s, 每 1 GB/s 出流量代价 = %.2f\n",
         g0_solo - g0_busy,
         nvl_when_g0busy > 1 ? (g0_solo - g0_busy) / nvl_when_g0busy : 0);

  printf("\n[done]\n");
  return 0;
}
