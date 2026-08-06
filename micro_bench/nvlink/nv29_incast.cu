// ============================================================================
// nv29_incast —— 多对一拥塞与公平性 (GPU1/2/3 同时写 GPU0)
//
// (a) 接收端总带宽是否等于单源带宽 -> 接收端是不是瓶颈
// (b) 各源份额是否公平 (标准差)
// (c) 有没有拥塞崩溃 (总带宽随源数反而下降)
// (d) 非对称: 一个源全力打 + 一个源轻载, 看轻载源的延迟被拖累多少
//     (轻载源用一个探针 kernel 测单次远端写的往返延迟)
//
// 多 GPU 计时姿势: 每个 device 自己的 stream + 自己的 event,
// 先都 launch 再统一同步; 聚合带宽 = 总字节 / max(各自时间)。
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

// 探针: 单线程做 N 次「远端写 + 读回」往返, 测平均延迟(cycles)
// 用 ld.acquire.sys / st.release.sys 保证真的走到对端而不是停在本地队列
__global__ void k_probe(unsigned* remote, long long* out, int iters) {
  if (threadIdx.x || blockIdx.x) return;
  long long t0 = clk();
  unsigned acc = 0;
  for (int i = 0; i < iters; ++i) {
    st_release_sys(remote, i);
    acc += ld_acquire_sys(remote);  // 强制等到对端可见, 形成往返依赖
  }
  long long t1 = clk();
  out[0] = (t1 - t0) / iters;
  out[1] = acc;  // sink
}

int main() {
  NvlEnv env = nvl_init(4);
  printf("# nv29_incast — %s x%d, %d SM, %.3f GHz\n", env.name, env.ndev,
         env.sm, env.clkGHz);
  nvl_enable_peers(env.ndev);

  const size_t BUF = 256ull << 20;
  const size_t N = BUF / sizeof(uint4);
  int blk = 256, grid = env.sm * 4;
  const int REP = 12;
  const double PASS = (double)BUF * REP;

  // GPU0 上给每个源开一块独立的接收 buffer(不相交, 避免写冲突)
  uint4* dst[3];
  for (int i = 0; i < 3; ++i) dst[i] = (uint4*)nvl_alloc(0, BUF);
  CK(cudaSetDevice(0));
  unsigned* probe_tgt = (unsigned*)nvl_alloc(0, 4096);

  // 每个源 GPU (1,2,3) 自己的 stream / event
  cudaStream_t st[3];
  cudaEvent_t ea[3], eb[3];
  for (int i = 0; i < 3; ++i) {
    CK(cudaSetDevice(i + 1));
    CK(cudaStreamCreate(&st[i]));
    CK(cudaEventCreate(&ea[i]));
    CK(cudaEventCreate(&eb[i]));
  }
  CK(cudaSetDevice(0));

  auto sync_all = [&]() {
    for (int d = 0; d < 4; ++d) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); }
    CK(cudaSetDevice(0));
  };

  // ============================================================ (a)(b)(c)
  hdr("a/b/c) N 打 1 incast: GPU1..N 同时写 GPU0");
  printf("| 源数 | GPU1 | GPU2 | GPU3 | 各流之和 | 墙钟聚合 | vs 单源 | 份额std%% |\n");
  printf("|---|---|---|---|---|---|---|---|\n");
  double single = 0;
  for (int ns = 1; ns <= 3; ++ns) {
    // warmup
    for (int i = 0; i < ns; ++i) {
      CK(cudaSetDevice(i + 1));
      k_wr<<<grid, blk, 0, st[i]>>>(dst[i], N, REP);
    }
    sync_all();

    double bw[3] = {0, 0, 0}, bestwall = 0, sumAt = 0;
    for (int t = 0; t < 3; ++t) {
      sync_all();
      // 先都 launch 出去
      for (int i = 0; i < ns; ++i) {
        CK(cudaSetDevice(i + 1));
        CK(cudaEventRecord(ea[i], st[i]));
        k_wr<<<grid, blk, 0, st[i]>>>(dst[i], N, REP);
        CK(cudaEventRecord(eb[i], st[i]));
      }
      // 统一同步
      sync_all();
      double mx = 0, sum = 0, tmp[3];
      for (int i = 0; i < ns; ++i) {
        CK(cudaSetDevice(i + 1));
        float ms; CK(cudaEventElapsedTime(&ms, ea[i], eb[i]));
        tmp[i] = gbps(PASS, ms);
        sum += tmp[i];
        mx = std::max(mx, (double)ms);
      }
      CK(cudaSetDevice(0));
      double wall = gbps(PASS * ns, mx);
      if (wall > bestwall) {
        bestwall = wall; sumAt = sum;
        for (int i = 0; i < ns; ++i) bw[i] = tmp[i];
      }
    }
    if (ns == 1) single = bestwall;
    double mean = sumAt / ns, var = 0;
    for (int i = 0; i < ns; ++i) var += (bw[i] - mean) * (bw[i] - mean);
    double sd = ns > 1 ? sqrt(var / ns) : 0;

    printf("| %d ", ns);
    for (int i = 0; i < 3; ++i) {
      if (i < ns) printf("| %.1f ", bw[i]); else printf("| - ");
    }
    printf("| %.1f | %.1f | %.2fx | %.1f%% |\n", sumAt, bestwall,
           bestwall / single, 100.0 * sd / mean);
  }
  printf("\n判据: 墙钟聚合列若恒定 => 接收端(GPU0)是瓶颈;\n");
  printf("      若随源数下降 => 拥塞崩溃\n");

  // 计数器: GPU0 接收侧
  hdr("a2) GPU0 接收侧硬件计数器");
  for (int ns = 1; ns <= 3; ++ns) {
    sync_all();
    NvlCounters c0 = nvl_read_counters(0);
    for (int i = 0; i < ns; ++i) {
      CK(cudaSetDevice(i + 1));
      k_wr<<<grid, blk, 0, st[i]>>>(dst[i], N, REP);
    }
    sync_all();
    NvlCounters c1 = nvl_read_counters(0);
    char tag[32];
    snprintf(tag, sizeof tag, "%d 源打 GPU0", ns);
    nvl_report(tag, PASS * ns, nvl_diff(c0, c1));
  }
  printf("(GPU0 的 dataRx 应 = ns * %.0f MB)\n", PASS / 1e6);

  // ============================================================ (d) 非对称
  hdr("d) 非对称负载: GPU1 全力打 GPU0, GPU2 用探针测延迟");
  const int PIT = 2000;
  CK(cudaSetDevice(2));
  long long* pout = (long long*)nvl_alloc(2, 1024);
  CK(cudaSetDevice(2));

  auto run_probe = [&]() -> double {
    k_probe<<<1, 1, 0, st[1]>>>(probe_tgt, pout, PIT);
    CK(cudaStreamSynchronize(st[1]));
    long long h[2];
    CK(cudaMemcpy(h, pout, sizeof h, cudaMemcpyDeviceToHost));
    return (double)h[0];
  };

  // 空闲基线
  sync_all();
  CK(cudaSetDevice(2));
  double cyc_idle = run_probe();
  // 再取一次稳定值
  cyc_idle = std::min(cyc_idle, run_probe());

  // GPU1 全力打 GPU0 的同时测
  double cyc_busy = 0;
  {
    sync_all();
    CK(cudaSetDevice(1));
    // 打足够长的时间覆盖探针
    k_wr<<<grid, blk, 0, st[0]>>>(dst[0], N, REP * 4);
    CK(cudaSetDevice(2));
    cyc_busy = run_probe();
    sync_all();
  }
  CK(cudaSetDevice(0));

  printf("| 场景 | 往返延迟 cycles | ns | vs 空闲 |\n|---|---|---|---|\n");
  printf("| GPU2->GPU0 空闲 | %.0f | %.0f | 1.00x |\n", cyc_idle,
         cyc2ns(cyc_idle, env.clkGHz));
  printf("| GPU2->GPU0 (GPU1 满载打 GPU0) | %.0f | %.0f | %.2fx |\n", cyc_busy,
         cyc2ns(cyc_busy, env.clkGHz), cyc_busy / cyc_idle);
  printf("\n(探针 = 单线程 st.release.sys + ld.acquire.sys 往返, %d 次取平均)\n",
         PIT);

  printf("\n[done]\n");
  return 0;
}
