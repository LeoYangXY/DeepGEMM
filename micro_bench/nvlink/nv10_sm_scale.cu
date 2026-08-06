// ============================================================================
// nv10_sm_scale —— SM 数细粒度扫描到饱和 / 过饱和
//
// 目标: 画出 远端写带宽 vs 活跃 SM 数 的完整曲线, 回答三个问题
//   Q1 饱和 NVLink 需要几个 SM?
//   Q2 饱和点(knee)在哪, 对应带宽占理论 478 / 实测 370 GB/s 的多少?
//   Q3 超过饱和点后带宽是平的还是掉下来(拥塞崩溃)?
//
// 控制变量:
//   - block=256, __launch_bounds__(256,1) 提示每 SM 只驻留 1 个 block,
//     这样 grid 数 ≈ 活跃 SM 数 (grid<=78 时)。
//   - grid 从 1 扫到 312 (=78*4), 覆盖 1..4 个 wave。
//   - 固定总字节 256MB, 用 cudaEvent 计时。
//   - 同时用 NVLink 硬件字节计数器交叉验证 (确认流量真的走了 link)。
//
// 另做两组对照:
//   - 本地 HBM 写: 看 SM 侧本身的 scaling, 排除 "SM 数不够" 的解释
//   - 远端写 + 4 个 block/SM (block=1024): 看每 SM 更多 warp 能否补偿少 SM
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"

__global__ __launch_bounds__(256, 1) void k_wr256(uint4* __restrict__ dst,
                                                  size_t n) {
  size_t i = blockIdx.x * 256ull + threadIdx.x;
  size_t s = (size_t)gridDim.x * 256ull;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (; i < n; i += s) st128(dst + i, v);
}
__global__ __launch_bounds__(256, 1) void k_rd256(const uint4* __restrict__ src,
                                                  uint4* sink, size_t n) {
  size_t i = blockIdx.x * 256ull + threadIdx.x;
  size_t s = (size_t)gridDim.x * 256ull;
  uint4 a = make_uint4(0, 0, 0, 0);
  for (; i < n; i += s) {
    uint4 v = ld128(src + i);
    a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
  }
  if (a.x == 0xdeadbeefu && a.y == 0xfeedu) *sink = a;
}
// 1024 thr/block => 32 warp/SM 满载
__global__ __launch_bounds__(1024, 1) void k_wr1024(uint4* __restrict__ dst,
                                                    size_t n) {
  size_t i = blockIdx.x * 1024ull + threadIdx.x;
  size_t s = (size_t)gridDim.x * 1024ull;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (; i < n; i += s) st128(dst + i, v);
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv10_sm_scale — %s, %d SM, %.3f GHz\n", env.name, env.sm,
         env.clkGHz);

  const size_t BYTES = 256ull << 20;
  const size_t N = BYTES / sizeof(uint4);
  uint4* local  = (uint4*)nvl_alloc(0, BYTES);
  uint4* remote = (uint4*)nvl_alloc(1, BYTES);
  CK(cudaSetDevice(0));
  uint4* sink = (uint4*)nvl_alloc(0, 4096);
  CK(cudaSetDevice(0));

  // occupancy 检查
  int nb256 = 0, nb1024 = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb256, (void*)k_wr256, 256, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb1024, (void*)k_wr1024, 1024, 0);
  printf("# occupancy: k_wr256 -> %d block/SM, k_wr1024 -> %d block/SM\n", nb256,
         nb1024);
  printf("# 总流量 %zu MB, best-of-3\n", BYTES >> 20);

  hdr("SM 数扫描: 远端写 / 远端读 / 本地写 (block=256, 1 block/SM)");
  printf("%6s %8s %13s %13s %13s %11s %11s\n", "grid", "SM(eff)", "remWR GB/s",
         "remRD GB/s", "locWR GB/s", "remWR/SM", "%peak370");
  int grids[] = {1,2,3,4,6,8,10,12,16,20,24,28,32,39,46,52,60,68,78,
                 90,104,117,130,156,195,234,273,312};
  int NG = sizeof(grids)/sizeof(int);
  double peakw = 0; int peakg = 0;
  for (int i = 0; i < NG; ++i) {
    int g = grids[i];
    double tw = bench_ms([&] { k_wr256<<<g, 256>>>(remote, N); }, 3);
    double tr = bench_ms([&] { k_rd256<<<g, 256>>>(remote, sink, N); }, 3);
    double tl = bench_ms([&] { k_wr256<<<g, 256>>>(local, N); }, 3);
    double bw = gbps(BYTES, tw);
    int eff = g < env.sm ? g : env.sm;
    if (bw > peakw) { peakw = bw; peakg = g; }
    printf("%6d %8d %13.2f %13.2f %13.2f %11.2f %10.1f%%\n", g, eff, bw,
           gbps(BYTES, tr), gbps(BYTES, tl), bw / eff, 100.0 * bw / 370.0);
  }
  printf("\n峰值 remWR = %.2f GB/s @ grid=%d\n", peakw, peakg);

  hdr("对照: block=1024 (32 warp/SM), 少 SM 多 warp 能否补偿?");
  printf("%6s %8s %13s %16s\n", "grid", "warps", "remWR GB/s", "vs blk256同SM");
  int g2[] = {1, 2, 4, 8, 16, 24, 32, 39, 52, 78};
  for (int i = 0; i < (int)(sizeof(g2)/sizeof(int)); ++i) {
    int g = g2[i];
    double t1 = bench_ms([&] { k_wr1024<<<g, 1024>>>(remote, N); }, 3);
    double t2 = bench_ms([&] { k_wr256<<<g, 256>>>(remote, N); }, 3);
    printf("%6d %8d %13.2f %15.2fx\n", g, g * 32, gbps(BYTES, t1),
           gbps(BYTES, t1) / gbps(BYTES, t2));
  }

  // ------------------------------------------------- NVLink 计数器交叉验证
  hdr("NVLink 硬件字节计数器交叉验证 (grid=78, block=256, 远端写 20 次)");
  {
    NvlCounters a = nvl_read_counters(0);
    for (int r = 0; r < 20; ++r) k_wr256<<<78, 256>>>(remote, N);
    CK(cudaDeviceSynchronize());
    NvlCounters b = nvl_read_counters(0);
    NvlCounters d = nvl_diff(a, b);
    double expMB = 20.0 * BYTES / 1e6;
    nvl_report("remWR grid=78", 20.0 * BYTES, d);
    printf("dataTx/payload = %.3f  (payload = %.1f MB)\n",
           (double)d.dataTx / 1e6 / expMB, expMB);
  }

  printf("\n[done]\n");
  return 0;
}
