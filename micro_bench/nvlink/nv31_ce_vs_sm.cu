// ============================================================================
// nv31_ce_vs_sm —— Copy Engine vs SM 路径完整对比
//
// (a) size 扫描 4KB->512MB: cudaMemcpyPeerAsync(CE) vs SM kernel 拷贝
// (b) 多 CE 并发: 2/4/8 stream 同时 memcpyPeer, 对照 asyncEngineCount
// (c) CE 拷贝时 SM 是否空闲: CE + 计算 kernel, 测计算性能损失
// (d) CE + SM 同打一条链路, 测总带宽和份额
// (e) memcpyPeer 启动开销 (0/小字节的固定延迟)
// ============================================================================
#include "nvl_common.cuh"
#include <algorithm>
#include <vector>

__global__ void k_copy(uint4* __restrict__ dst, const uint4* __restrict__ src,
                       size_t n, int rep) {
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += s) st128(dst + i, ld128(src + i));
}

// 纯计算 kernel (FMA 密集, 不碰内存) —— 用来测 CE 是否偷 SM
__global__ void k_compute(float* out, int iters) {
  float a = threadIdx.x * 1e-3f, b = 1.0000001f, c = 0.9999999f;
  for (int i = 0; i < iters; ++i) { a = fmaf(a, b, c); a = fmaf(a, c, b); }
  if (a == 12345.678f) out[blockIdx.x] = a;  // sink, 永不成立
}

int main() {
  NvlEnv env = nvl_init(2);
  printf("# nv31_ce_vs_sm — %s x%d, %d SM, %.3f GHz\n", env.name, env.ndev,
         env.sm, env.clkGHz);
  nvl_enable_peers(env.ndev);

  int ae0 = 0, ae1 = 0;
  cudaDeviceGetAttribute(&ae0, cudaDevAttrAsyncEngineCount, 0);
  cudaDeviceGetAttribute(&ae1, cudaDevAttrAsyncEngineCount, 1);
  printf("cudaDevAttrAsyncEngineCount: GPU0=%d GPU1=%d\n", ae0, ae1);

  const size_t MAXB = 512ull << 20;
  void* src0 = nvl_alloc(0, MAXB, 0x11);
  void* dst1 = nvl_alloc(1, MAXB);
  CK(cudaSetDevice(0));
  float* csink = (float*)nvl_alloc(0, 4096);
  CK(cudaSetDevice(0));

  int blk = 256, FG = env.sm * 4;

  // ============================================================ (e) 启动开销
  hdr("e) cudaMemcpyPeerAsync 启动开销 (小字节固定延迟)");
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  printf("| 字节 | 平均单次 us | 等效 GB/s |\n|---|---|---|\n");
  for (size_t b : {(size_t)0, (size_t)4, (size_t)256, (size_t)4096,
                   (size_t)65536, (size_t)(1 << 20)}) {
    const int R = 2000;
    // warmup
    for (int i = 0; i < 100; ++i) cudaMemcpyPeerAsync(dst1, 1, src0, 0, b, s);
    CK(cudaStreamSynchronize(s));
    Timer t; t.start(s);
    for (int i = 0; i < R; ++i) cudaMemcpyPeerAsync(dst1, 1, src0, 0, b, s);
    double ms = t.stop_ms(s);
    double us = ms * 1000.0 / R;
    printf("| %zu | %.2f | %.1f |\n", b, us, b ? b / 1e9 / (us / 1e6) : 0.0);
  }

  // ============================================================ (a) size 扫描
  hdr("a) size 扫描: CE (memcpyPeer) vs SM kernel 拷贝");
  printf("| 大小 | CE GB/s | CE us | SM GB/s | SM us | 胜者 |\n");
  printf("|---|---|---|---|---|---|\n");
  size_t szs[] = {4ull<<10, 16ull<<10, 64ull<<10, 256ull<<10, 1ull<<20,
                  4ull<<20, 16ull<<20, 64ull<<20, 256ull<<20, 512ull<<20};
  for (size_t B : szs) {
    size_t n = B / sizeof(uint4);
    // 小 size 多跑几次
    int R = B <= (1<<20) ? 2000 : (B <= (16u<<20) ? 200 : 20);
    // CE
    for (int i = 0; i < 20; ++i) cudaMemcpyPeerAsync(dst1, 1, src0, 0, B, s);
    CK(cudaStreamSynchronize(s));
    double ce_ms = 1e30;
    for (int t2 = 0; t2 < 3; ++t2) {
      Timer t; t.start(s);
      for (int i = 0; i < R; ++i) cudaMemcpyPeerAsync(dst1, 1, src0, 0, B, s);
      double ms = t.stop_ms(s);
      ce_ms = std::min(ce_ms, ms / R);
    }
    // SM
    int g = (int)std::min((size_t)FG, std::max((size_t)1, n / (size_t)blk));
    for (int i = 0; i < 20; ++i)
      k_copy<<<g, blk, 0, s>>>((uint4*)dst1, (const uint4*)src0, n, 1);
    CK(cudaStreamSynchronize(s));
    double sm_ms = 1e30;
    for (int t2 = 0; t2 < 3; ++t2) {
      Timer t; t.start(s);
      for (int i = 0; i < R; ++i)
        k_copy<<<g, blk, 0, s>>>((uint4*)dst1, (const uint4*)src0, n, 1);
      double ms = t.stop_ms(s);
      sm_ms = std::min(sm_ms, ms / R);
    }
    double ceb = gbps(B, ce_ms), smb = gbps(B, sm_ms);
    printf("| %zuKB | %.1f | %.2f | %.1f | %.2f | %s |\n", B >> 10, ceb,
           ce_ms * 1000, smb, sm_ms * 1000, ceb > smb ? "CE" : "SM");
  }

  // ============================================================ (b) 多 CE 并发
  hdr("b) 多 stream 并发 memcpyPeer (推断可用 copy engine 数)");
  const size_t CB = 64ull << 20;
  printf("| stream 数 | 总 GB/s | vs 1 stream |\n|---|---|---|\n");
  double base1 = 0;
  for (int ns : {1, 2, 4, 8}) {
    std::vector<cudaStream_t> ss(ns);
    for (int i = 0; i < ns; ++i) CK(cudaStreamCreate(&ss[i]));
    const int R = 20;
    // warmup
    for (int i = 0; i < ns; ++i)
      cudaMemcpyPeerAsync((char*)dst1 + i * CB, 1, (char*)src0 + i * CB, 0, CB,
                          ss[i]);
    CK(cudaDeviceSynchronize());
    double best = 0;
    for (int t2 = 0; t2 < 3; ++t2) {
      CK(cudaDeviceSynchronize());
      Timer t; t.start(0);
      for (int r = 0; r < R; ++r)
        for (int i = 0; i < ns; ++i)
          cudaMemcpyPeerAsync((char*)dst1 + i * CB, 1, (char*)src0 + i * CB, 0,
                              CB, ss[i]);
      double ms = t.stop_ms(0);
      CK(cudaDeviceSynchronize());
      best = std::max(best, gbps((double)CB * ns * R, ms));
    }
    if (ns == 1) base1 = best;
    printf("| %d | %.1f | %.2fx |\n", ns, best, best / base1);
    for (int i = 0; i < ns; ++i) CK(cudaStreamDestroy(ss[i]));
  }

  // ============================================================ (c) CE 偷 SM?
  hdr("c) CE 拷贝时计算 kernel 的性能损失 (量化 CE 的零 SM 占用)");
  const int CIT = 20000;
  cudaStream_t sc, sd;
  CK(cudaStreamCreate(&sc));
  CK(cudaStreamCreate(&sd));
  // 计算 kernel 单独
  k_compute<<<FG, blk, 0, sc>>>(csink, CIT);
  CK(cudaStreamSynchronize(sc));
  double comp_solo = 1e30;
  for (int t2 = 0; t2 < 3; ++t2) {
    Timer t; t.start(sc);
    k_compute<<<FG, blk, 0, sc>>>(csink, CIT);
    comp_solo = std::min(comp_solo, (double)t.stop_ms(sc));
  }
  // 计算 + CE 并发
  double comp_ce = 1e30, ce_bw_c = 0;
  for (int t2 = 0; t2 < 3; ++t2) {
    CK(cudaDeviceSynchronize());
    cudaEvent_t xa, xb; CK(cudaEventCreate(&xa)); CK(cudaEventCreate(&xb));
    CK(cudaEventRecord(xa, sc));
    k_compute<<<FG, blk, 0, sc>>>(csink, CIT);
    CK(cudaEventRecord(xb, sc));
    // CE 持续拷贝
    Timer tce; tce.start(sd);
    int nrep = 16;
    for (int i = 0; i < nrep; ++i)
      cudaMemcpyPeerAsync(dst1, 1, src0, 0, 64ull << 20, sd);
    double cems = tce.stop_ms(sd);
    CK(cudaDeviceSynchronize());
    float mc; CK(cudaEventElapsedTime(&mc, xa, xb));
    comp_ce = std::min(comp_ce, (double)mc);
    ce_bw_c = std::max(ce_bw_c, gbps((double)(64ull << 20) * nrep, cems));
    cudaEventDestroy(xa); cudaEventDestroy(xb);
  }
  // 计算 + SM 拷贝并发 (对照组)
  double comp_sm = 1e30;
  {
    size_t n = (64ull << 20) / sizeof(uint4);
    for (int t2 = 0; t2 < 3; ++t2) {
      CK(cudaDeviceSynchronize());
      cudaEvent_t xa, xb; CK(cudaEventCreate(&xa)); CK(cudaEventCreate(&xb));
      CK(cudaEventRecord(xa, sc));
      k_compute<<<FG, blk, 0, sc>>>(csink, CIT);
      CK(cudaEventRecord(xb, sc));
      for (int i = 0; i < 16; ++i)
        k_copy<<<FG, blk, 0, sd>>>((uint4*)dst1, (const uint4*)src0, n, 1);
      CK(cudaDeviceSynchronize());
      float mc; CK(cudaEventElapsedTime(&mc, xa, xb));
      comp_sm = std::min(comp_sm, (double)mc);
      cudaEventDestroy(xa); cudaEventDestroy(xb);
    }
  }
  printf("| 场景 | 计算 kernel ms | 相对单独 | 并发拷贝 GB/s |\n|---|---|---|---|\n");
  printf("| 计算单独跑 | %.2f | 1.00x | - |\n", comp_solo);
  printf("| 计算 + CE 拷贝 | %.2f | %.3fx | %.1f |\n", comp_ce,
         comp_ce / comp_solo, ce_bw_c);
  printf("| 计算 + SM 拷贝 | %.2f | %.3fx | - |\n", comp_sm,
         comp_sm / comp_solo);
  printf("\n(CE 行接近 1.00x => CE 真的零 SM 占用; SM 行应明显 >1)\n");

  // ============================================================ (d) CE + SM 同链路
  hdr("d) CE + SM 同时打一条链路");
  {
    const size_t HB = 256ull << 20;
    size_t n = HB / sizeof(uint4);
    // 各自单独
    double ce_solo = 0, sm_solo = 0;
    {
      for (int i = 0; i < 5; ++i) cudaMemcpyPeerAsync(dst1, 1, src0, 0, HB, sc);
      CK(cudaStreamSynchronize(sc));
      Timer t; t.start(sc);
      for (int i = 0; i < 10; ++i) cudaMemcpyPeerAsync(dst1, 1, src0, 0, HB, sc);
      ce_solo = gbps((double)HB * 10, t.stop_ms(sc));

      k_copy<<<FG, blk, 0, sc>>>((uint4*)dst1, (const uint4*)src0, n, 1);
      CK(cudaStreamSynchronize(sc));
      Timer t2_; t2_.start(sc);
      for (int i = 0; i < 10; ++i)
        k_copy<<<FG, blk, 0, sc>>>((uint4*)dst1, (const uint4*)src0, n, 1);
      sm_solo = gbps((double)HB * 10, t2_.stop_ms(sc));
    }
    // 并发
    cudaEvent_t ca, cb, sa, sb;
    CK(cudaEventCreate(&ca)); CK(cudaEventCreate(&cb));
    CK(cudaEventCreate(&sa)); CK(cudaEventCreate(&sb));
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(ca, sc));
    for (int i = 0; i < 10; ++i) cudaMemcpyPeerAsync(dst1, 1, src0, 0, HB, sc);
    CK(cudaEventRecord(cb, sc));
    CK(cudaEventRecord(sa, sd));
    for (int i = 0; i < 10; ++i)
      k_copy<<<FG, blk, 0, sd>>>((uint4*)dst1, (const uint4*)src0, n, 1);
    CK(cudaEventRecord(sb, sd));
    CK(cudaDeviceSynchronize());
    float mc, msm;
    CK(cudaEventElapsedTime(&mc, ca, cb));
    CK(cudaEventElapsedTime(&msm, sa, sb));
    double cbw = gbps((double)HB * 10, mc), sbw = gbps((double)HB * 10, msm);
    double agg = gbps((double)HB * 20, std::max(mc, msm));
    printf("| 场景 | CE GB/s | SM GB/s | 合计 GB/s |\n|---|---|---|---|\n");
    printf("| CE 单独 | %.1f | - | %.1f |\n", ce_solo, ce_solo);
    printf("| SM 单独 | - | %.1f | %.1f |\n", sm_solo, sm_solo);
    printf("| CE + SM 并发 | %.1f | %.1f | %.1f |\n", cbw, sbw, agg);
    printf("\nCE 份额 %.0f%%, SM 份额 %.0f%%\n", 100 * cbw / (cbw + sbw),
           100 * sbw / (cbw + sbw));
  }

  printf("\n[done]\n");
  return 0;
}
