// ============================================================================
// nv04_coalesce —— warp 内合并度
//
// 假设: 一个 warp(32 线程) 每次访存产生的「请求数」由 LSU 合并逻辑决定,
//       粒度是 128B line(或 32B sector)。固定每 warp 搬 128B 有用数据
//       (32 线程 * 4B), 但把这 128B 散布到 L 条不同的 128B line 上:
//         L=1  -> 32 线程都在同一条 line 内, 完全合并 -> 1 个请求
//         L=32 -> 每线程各自一条 line -> 32 个请求, 每个只用 4B
//       如果远端带宽随 L 线性下降, 说明合并发生在 SM LSU 侧,
//       NVLink 侧不会二次合并(即 SM 发多少包, 线上就是多少包)。
//
// 构造: warp 内 lane l 的地址 = warpBase + (l % L)*LINE + (l / L)*4
//       -> 恰好 L 条 line, 每条 line 上放 32/L 个连续 4B = 128/L 字节
//       -> 总有用字节 = 128B, 与 L 无关。控制变量成立。
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"

#define LINE 128

template <int L>
__global__ void k_wr_spread(char* __restrict__ base, size_t spanMask, int iter) {
  int lane = threadIdx.x & 31;
  size_t warpId = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
  size_t nwarp = ((size_t)gridDim.x * blockDim.x) >> 5;
  int inLine = lane % L;        // 落在第几条 line
  int slot = lane / L;          // line 内第几个 4B
  for (int k = 0; k < iter; ++k) {
    size_t w = warpId + (size_t)k * nwarp;
    // 每个 warp 占据 L*LINE 的连续区域(所以不同 warp 不冲突)
    size_t off = (w * (size_t)L * LINE + (size_t)inLine * LINE + slot * 4) & spanMask;
    st32(base + off, 0x5A5A5A5Au);
  }
}

template <int L>
__global__ void k_rd_spread(const char* __restrict__ base, unsigned* sink,
                            size_t spanMask, int iter) {
  int lane = threadIdx.x & 31;
  size_t warpId = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
  size_t nwarp = ((size_t)gridDim.x * blockDim.x) >> 5;
  int inLine = lane % L, slot = lane / L;
  unsigned acc = 0;
  for (int k = 0; k < iter; ++k) {
    size_t w = warpId + (size_t)k * nwarp;
    size_t off = (w * (size_t)L * LINE + (size_t)inLine * LINE + slot * 4) & spanMask;
    acc ^= ld32(base + off);
  }
  if (acc == 0xdeadbeefu) *sink = acc;
}

static int G, B;
static size_t SPANMASK;
static const int ITER = 256;

template <int L>
static void one(char* rem, char* loc, unsigned* sink, double payload,
                double base_wr, double base_rd, bool print_ratio) {
  double t;
  t = bench_ms([&] { k_wr_spread<L><<<G, B>>>(rem, SPANMASK, ITER); }, 3);
  double wr = gbps(payload, t);
  t = bench_ms([&] { k_rd_spread<L><<<G, B>>>(rem, sink, SPANMASK, ITER); }, 3);
  double rd = gbps(payload, t);
  t = bench_ms([&] { k_wr_spread<L><<<G, B>>>(loc, SPANMASK, ITER); }, 3);
  double wl = gbps(payload, t);
  t = bench_ms([&] { k_rd_spread<L><<<G, B>>>(loc, sink, SPANMASK, ITER); }, 3);
  double rl = gbps(payload, t);
  printf("| %d | %dB | %.1f | %.1f | %.1f | %.1f | %.2fx | %.2fx |\n", L,
         128 / L, wr, rd, wl, rl,
         print_ratio ? base_wr / wr : 1.0, print_ratio ? base_rd / rd : 1.0);
  fflush(stdout);
}

// counters 版本
template <int L>
static void one_ctr(char* rem, double payload_per_kernel) {
  double t1 = bench_ms([&] { k_wr_spread<L><<<G, B>>>(rem, SPANMASK, ITER); }, 2);
  int rep = (int)(2000.0 / t1) + 1;
  if (rep > 40000) rep = 40000;
  CK(cudaDeviceSynchronize());
  NvlCounters a = nvl_read_counters(0);
  for (int r = 0; r < rep; ++r) k_wr_spread<L><<<G, B>>>(rem, SPANMASK, ITER);
  CK(cudaDeviceSynchronize());
  NvlCounters b = nvl_read_counters(0);
  NvlCounters d = nvl_diff(a, b);
  double pl = payload_per_kernel * rep;
  printf("| %d | 写 | %.0f | %.0f | %.0f | %.0f | %.0f | %.3f |\n", L, pl / 1e6,
         d.dataTx / 1e6, d.dataRx / 1e6, d.rawTx / 1e6, d.rawRx / 1e6,
         (double)(d.rawTx + d.rawRx) / pl);
  fflush(stdout);
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv04_coalesce — warp 内合并度 (%s, %d SM)\n", env.name, env.sm);

  const size_t BUF = 512ull << 20;
  SPANMASK = BUF - 1;
  CK(cudaSetDevice(0));
  char* loc = (char*)nvl_alloc(0, BUF);
  unsigned* sink = (unsigned*)nvl_alloc(0, 1024);
  char* rem = (char*)nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));

  G = env.sm * 4; B = 256;
  size_t threads = (size_t)G * B;
  double payload = (double)threads * ITER * 4;

  hdr("A) 每 warp 128B 有用数据散布到 L 条 128B line 上");
  printf("payload/kernel = %.1f MB (与 L 无关)\n", payload / 1e6);
  printf("| L(line数) | 每line有用 | 远端写 GB/s | 远端读 GB/s | 本地写 GB/s | 本地读 GB/s | 远端写下降 | 远端读下降 |\n");
  printf("|---|---|---|---|---|---|---|---|\n");

  // 先跑 L=1 拿基线
  double t = bench_ms([&] { k_wr_spread<1><<<G, B>>>(rem, SPANMASK, ITER); }, 3);
  double b_wr = gbps(payload, t);
  t = bench_ms([&] { k_rd_spread<1><<<G, B>>>(rem, sink, SPANMASK, ITER); }, 3);
  double b_rd = gbps(payload, t);

  one<1>(rem, loc, sink, payload, b_wr, b_rd, true);
  one<2>(rem, loc, sink, payload, b_wr, b_rd, true);
  one<4>(rem, loc, sink, payload, b_wr, b_rd, true);
  one<8>(rem, loc, sink, payload, b_wr, b_rd, true);
  one<16>(rem, loc, sink, payload, b_wr, b_rd, true);
  one<32>(rem, loc, sink, payload, b_wr, b_rd, true);

  hdr("B) 硬件计数器: 请求放大 (远端写)");
  printf("| L | 方向 | payload MB | dataTx MB | dataRx MB | rawTx MB | rawRx MB | wire/payload |\n");
  printf("|---|---|---|---|---|---|---|---|\n");
  one_ctr<1>(rem, payload);
  one_ctr<4>(rem, payload);
  one_ctr<8>(rem, payload);
  one_ctr<32>(rem, payload);

  printf("\n[done]\n");
  return 0;
}
