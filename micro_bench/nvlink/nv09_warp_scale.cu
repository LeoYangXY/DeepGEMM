// ============================================================================
// nv09_warp_scale —— 并发 warp 数扫描: 单 SM 能推多少 NVLink 带宽?
//
// 两段扫描:
//   A) grid=1 (锁死在 1 个 SM), block 内 warp 数 = 1,2,4,8,16,32
//      -> 单 SM 的远端访存流水需要多少 warp 才填满? 单 SM 带宽上限多少?
//   B) block 固定 8 warp (256 thr), grid = 1..156
//      -> 多 SM 叠加是否线性? 什么时候撞上 link 侧上限?
//
// 每个线程做 grid-stride 的 st128 / ld128, 固定总字节数, 用 cudaEvent 计时。
// 固定总字节 => 不同配置下每线程工作量不同, 但总流量相同, GB/s 可直接比较。
// ============================================================================
#include "nvl_common.cuh"

__global__ void k_wr(uint4* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (; i < n; i += s) st128(dst + i, v);
}
__global__ void k_rd(const uint4* __restrict__ src, uint4* sink, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 a = make_uint4(0, 0, 0, 0);
  for (; i < n; i += s) {
    uint4 v = ld128(src + i);
    a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
  }
  if (a.x == 0xdeadbeefu && a.y == 0xfeedu) *sink = a;
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv09_warp_scale — %s, %d SM, %.3f GHz\n", env.name, env.sm,
         env.clkGHz);

  // A 段用小一点的量, 否则 1 warp 跑不完
  const size_t SMALL = 32ull << 20;    // 32 MB
  const size_t BIG   = 256ull << 20;   // 256 MB
  const size_t BUF   = BIG;

  uint4* local  = (uint4*)nvl_alloc(0, BUF);
  uint4* remote = (uint4*)nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));
  uint4* sink = (uint4*)nvl_alloc(0, 4096);
  CK(cudaSetDevice(0));

  // ------------------------------------------------------------------ A
  hdr("A) grid=1 (锁 1 个 SM), 扫 block 内 warp 数 —— 单 SM 的 NVLink 上限");
  printf("总流量 %zu MB, 固定 grid=1\n", SMALL >> 20);
  printf("%6s %8s %14s %14s %14s %14s\n", "warps", "thr", "rem_WR GB/s",
         "rem_RD GB/s", "loc_WR GB/s", "loc_RD GB/s");
  const size_t nS = SMALL / sizeof(uint4);
  double prev_w = 0;
  for (int w = 1; w <= 32; w *= 2) {
    int blk = w * 32;
    double tw = bench_ms([&] { k_wr<<<1, blk>>>(remote, nS); }, 3);
    double tr = bench_ms([&] { k_rd<<<1, blk>>>(remote, sink, nS); }, 3);
    double lw = bench_ms([&] { k_wr<<<1, blk>>>(local, nS); }, 3);
    double lr = bench_ms([&] { k_rd<<<1, blk>>>(local, sink, nS); }, 3);
    double g = gbps(SMALL, tw);
    printf("%6d %8d %14.2f %14.2f %14.2f %14.2f", w, blk, g, gbps(SMALL, tr),
           gbps(SMALL, lw), gbps(SMALL, lr));
    if (prev_w > 0) printf("   (rem_WR x%.2f)", g / prev_w);
    printf("\n");
    prev_w = g;
  }

  // ------------------------------------------------------------------ B
  hdr("B) block=256 (8 warp) 固定, 扫 grid (SM 数) —— 多 SM 叠加线性度");
  printf("总流量 %zu MB\n", BIG >> 20);
  printf("%6s %14s %14s %16s %16s\n", "grid", "rem_WR GB/s", "rem_RD GB/s",
         "WR/SM GB/s", "占 370GB/s%");
  const size_t nB = BIG / sizeof(uint4);
  int grids[] = {1, 2, 4, 8, 16, 24, 32, 39, 48, 56, 64, 72, 78, 96, 117, 156, 234, 312};
  for (int gi = 0; gi < (int)(sizeof(grids) / sizeof(int)); ++gi) {
    int g = grids[gi];
    double tw = bench_ms([&] { k_wr<<<g, 256>>>(remote, nB); }, 3);
    double tr = bench_ms([&] { k_rd<<<g, 256>>>(remote, sink, nB); }, 3);
    double bw = gbps(BIG, tw);
    printf("%6d %14.2f %14.2f %16.2f %15.1f%%\n", g, bw, gbps(BIG, tr),
           bw / (g < env.sm ? g : env.sm), 100.0 * bw / 370.0);
  }

  printf("\n[读法] A 段: rem_WR 停止随 warp 数增长的那一行 = 单 SM 远端写流水填满点。\n");
  printf("       B 段: rem_WR 停止随 grid 增长的那一行 = 饱和所需 SM 数。\n");
  printf("\n[done]\n");
  return 0;
}
