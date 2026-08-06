// ============================================================================
// nv12_lat_vs_load —— 负载下的延迟曲线 (排队论视角, 找 knee point)
//
// 设计:
//   stream_bg   : 背景负载 kernel, grid 可调 -> 制造 0%~100% 的 NVLink 利用率
//   stream_probe: 探针 kernel, 1 thread 远端 pointer-chase 测单次往返延迟
//   两个 stream 并发跑在 GPU0 上, 探针在背景负载"稳态中间"采样。
//
// 同步保证探针落在负载稳态中:
//   背景 kernel 跑一个足够长的固定时长(靠大 N + 循环), 探针 kernel 在它启动后
//   由主机延迟一小段再 launch, 并且探针自己先 spin 等一个 device flag。
//   这里用更稳的办法: 背景 kernel 先置 flag=1 再进主循环; 探针 spin 等 flag。
//
// 输出: 负载强度(grid) -> 实测背景带宽 + 探针延迟。
//   knee = 延迟开始超线性上升的那一点, 记录其带宽利用率 %。
//   knee 之前延迟近似常数 => 队列有余量;
//   knee 之后陡升        => 队列开始堆积, 请求要排队等 credit。
// ============================================================================
#include "nvl_common.cuh"

#define CHASE 256

__global__ void k_build_ring(unsigned long long* ring, size_t nslot,
                             size_t stride_slot) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < nslot; i += s) ring[i] = (unsigned long long)(ring + (i * stride_slot + 1) % nslot);
}

// 背景负载: 反复扫 buffer LOOPS 遍, 保证持续足够久
__global__ __launch_bounds__(256) void k_bg(uint4* __restrict__ dst, size_t n,
                                            int loops,
                                            volatile unsigned* flag) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *flag = 1;
  size_t base = blockIdx.x * 256ull + threadIdx.x;
  size_t s = (size_t)gridDim.x * 256ull;
  uint4 v = make_uint4((unsigned)base, 2, 3, 4);
  for (int l = 0; l < loops; ++l)
    for (size_t i = base; i < n; i += s) st128(dst + i, v);
}

// 探针: spin 等 flag, 然后 pointer-chase
__global__ void k_probe(unsigned long long* ring, long long* out,
                        volatile unsigned* flag, int wait_flag) {
  if (threadIdx.x || blockIdx.x) return;
  if (wait_flag) {
    long long guard = 0;
    while (*flag == 0 && ++guard < 200000000LL) {}
  }
  // 预热一小段, 让探针自己的 TLB/路径热起来
  unsigned long long p = (unsigned long long)ring;
#pragma unroll 1
  for (int i = 0; i < 64; ++i)
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(p) : "l"(p) : "memory");

  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < CHASE; ++i)
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(p) : "l"(p) : "memory");
  long long t1 = clk();
  out[0] = t1 - t0;
  out[1] = (long long)(p & 1);
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv12_lat_vs_load — %s, %d SM, %.3f GHz\n", env.name, env.sm,
         env.clkGHz);

  const size_t BYTES = 128ull << 20;
  const size_t N = BYTES / sizeof(uint4);
  const size_t RINGB = 64ull << 20;
  const size_t NSLOT = RINGB / 8;

  uint4* remote = (uint4*)nvl_alloc(1, BYTES);
  unsigned long long* ring = (unsigned long long*)nvl_alloc(1, RINGB);
  CK(cudaSetDevice(1));
  k_build_ring<<<256, 256>>>(ring, NSLOT, 262147);
  CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(0));
  long long* out = (long long*)nvl_alloc(0, 4096);
  unsigned* flag;
  CK(cudaSetDevice(0));
  CK(cudaMalloc(&flag, 256));
  cudaStream_t s_bg, s_pr;
  CK(cudaStreamCreate(&s_bg));
  CK(cudaStreamCreate(&s_pr));

  long long h[2];

  // ------------------------------------------------ 空载基线
  hdr("0) 空载探针延迟基线 (背景 grid=0)");
  double idle = 1e30;
  for (int r = 0; r < 6; ++r) {
    CK(cudaMemset(out, 0, 32));
    k_probe<<<1, 32, 0, s_pr>>>(ring, out, flag, 0);
    CK(cudaStreamSynchronize(s_pr));
    CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost));
    double c = (double)h[0] / CHASE;
    if (r && c < idle) idle = c;
  }
  printf("空载延迟 = %.1f cyc = %.1f ns\n", idle, cyc2ns(idle, env.clkGHz));

  // ------------------------------------------------ 先标定各 grid 的带宽
  hdr("1) 标定: 各背景 grid 对应的独占远端写带宽 (= 负载强度)");
  int grids[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32, 48, 78, 156, 312};
  int NG = sizeof(grids) / sizeof(int);
  double bwv[64];
  printf("%6s %12s %10s\n", "grid", "BW GB/s", "util%");
  for (int i = 0; i < NG; ++i) {
    int g = grids[i];
    if (g == 0) { bwv[i] = 0; printf("%6d %12.2f %9.1f%%\n", 0, 0.0, 0.0); continue; }
    double ms = bench_ms([&] { k_bg<<<g, 256, 0, 0>>>(remote, N, 1, flag); }, 3);
    bwv[i] = gbps(BYTES, ms);
    printf("%6d %12.2f %9.1f%%\n", g, bwv[i], 100.0 * bwv[i] / 370.0);
  }

  // ------------------------------------------------ 并发测延迟
  hdr("2) 背景负载并发时的探针延迟 (排队论 knee)");
  printf("%6s %12s %9s %11s %11s %11s\n", "bg_grid", "BW GB/s", "util%",
         "lat cyc", "lat ns", "vs idle");
  double prev_lat = idle, prev_util = 0;
  int knee = -1; double knee_util = 0;
  for (int i = 0; i < NG; ++i) {
    int g = grids[i];
    double best = 1e30;
    for (int r = 0; r < 4; ++r) {
      CK(cudaMemset(out, 0, 32));
      CK(cudaMemset(flag, 0, 4));
      CK(cudaDeviceSynchronize());
      if (g > 0) {
        // loops 让背景跑得比探针久
        k_bg<<<g, 256, 0, s_bg>>>(remote, N, 6, flag);
        k_probe<<<1, 32, 0, s_pr>>>(ring, out, flag, 1);
      } else {
        k_probe<<<1, 32, 0, s_pr>>>(ring, out, flag, 0);
      }
      CK(cudaStreamSynchronize(s_pr));
      CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost));
      CK(cudaStreamSynchronize(s_bg));
      CKLAST();
      if (h[0] > 0) { double c = (double)h[0] / CHASE; if (c < best) best = c; }
    }
    double util = 100.0 * bwv[i] / 370.0;
    printf("%6d %12.2f %8.1f%% %11.1f %11.1f %10.2fx\n", g, bwv[i], util, best,
           cyc2ns(best, env.clkGHz), best / idle);
    // knee: 延迟首次超过空载 1.5 倍
    if (knee < 0 && best > idle * 1.5) { knee = g; knee_util = util; }
    prev_lat = best; prev_util = util;
  }
  (void)prev_lat; (void)prev_util;
  if (knee > 0)
    printf("\nknee point: bg_grid=%d, 此时背景带宽利用率 = %.1f%%, 延迟已达空载 1.5x\n",
           knee, knee_util);
  else
    printf("\n未检测到 1.5x knee: 延迟在全负载范围内都比较平 (队列一直有余量)\n");

  printf("\n[done]\n");
  return 0;
}
