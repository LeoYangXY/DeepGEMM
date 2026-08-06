// ============================================================================
// nv07_align —— 对齐敏感度
//
// 假设: 若远端访问的传输粒度是 G 字节且必须自然对齐, 那么一个跨越
//       G 边界的访问会被拆成 2 个请求 -> 带宽腰斩 / 请求数翻倍。
//
// 实验 A: 整体基址偏移。warp 内地址仍然连续(完全合并), 但整个 buffer
//         起点偏移 OFF = 0/1/2/4/8/16/32/64 字节。
//         - 16B(st128) 要求自然对齐, 非 16 对齐的地址硬件会出错, 所以
//           16B 只测 OFF ∈ {0,16,32,64}(仍是 16 对齐但相对 128B line 偏移)。
//         - 4B(st32) 测 OFF ∈ {0,4,8,16,32,64}(保持 4 对齐)。
//         观察: OFF 非 128 的倍数时, 每个 warp 的 128B 访问会横跨 2 条 line。
//
// 实验 B: 显式跨边界。每线程写一个 16B, 让它 **故意** 骑在 128B / 256B
//         边界上: 地址 = blockBase + BOUND - 8 (即前 8B 在上一条 line,
//         后 8B 在下一条)。与「不跨边界」(地址 = blockBase + BOUND) 对照,
//         其余完全相同。带宽比值直接给出拆分代价。
//         注意 16B 跨 128B 边界时仍需 16B 自然对齐 -> 用 2 条 st64 拼,
//         对照组也用 2 条 st64, 保证指令数相同(控制变量)。
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"

// ------------------------------------------------------ A: 基址偏移, 连续访问
template <int W>
__global__ void k_off(char* __restrict__ base, size_t mask, int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (int k = 0; k < iter; ++k) {
    size_t off = ((g + (size_t)k * s) * W) & mask;
    char* p = base + off;
    if (W == 16) st128(p, v);
    else st32(p, 0x77777777u);
  }
}
template <int W>
__global__ void k_off_rd(const char* __restrict__ base, unsigned* sink,
                         size_t mask, int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  unsigned a = 0;
  for (int k = 0; k < iter; ++k) {
    size_t off = ((g + (size_t)k * s) * W) & mask;
    const char* p = base + off;
    if (W == 16) { uint4 v = ld128(p); a ^= v.x ^ v.y ^ v.z ^ v.w; }
    else a ^= ld32(p);
  }
  if (a == 0xdeadbeefu) *sink = a;
}

// ------------------------------------------------------ B: 显式跨边界
// 关键: warp 内 lane 地址必须保持**连续**, 否则测的是"离散"而不是"跨界"。
// 每线程搬 8B (2 条 st32, 4B 对齐 -> 任意 4 的倍数 delta 都合法)。
// 线程 t 的地址 = (t*8 + delta) , 即整条 warp 连续覆盖 256B。
//   delta = 0  -> 每个 8B 都在 BOUND 内部, 不跨界
//   delta = 4  -> 每隔 BOUND/8 个线程就有一个 8B 骑在 BOUND 边界上
// 两种情况指令数/合并度完全相同, 唯一差别是是否跨 BOUND 边界。
__global__ void k_cross(char* __restrict__ base, size_t mask, int delta,
                        int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (int k = 0; k < iter; ++k) {
    size_t off = (((g + (size_t)k * s) * 8) & mask) + delta;
    char* p = base + off;
    st32(p, 0xE1E1E1E1u);
    st32(p + 4, 0xE2E2E2E2u);
  }
}
__global__ void k_cross_rd(const char* __restrict__ base, unsigned* sink,
                           size_t mask, int delta, int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  unsigned a = 0;
  for (int k = 0; k < iter; ++k) {
    size_t off = (((g + (size_t)k * s) * 8) & mask) + delta;
    const char* p = base + off;
    a ^= ld32(p) ^ ld32(p + 4);
  }
  if (a == 0xdeadbeefu) *sink = a;
}

static int G, Bk;
static const int ITER = 256;

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv07_align — 对齐敏感度 (%s, %d SM)\n", env.name, env.sm);

  const size_t BUF = 512ull << 20;
  const size_t USE = 256ull << 20;   // 只用前 256MB, 留出偏移余量
  const size_t MASK = USE - 1;
  CK(cudaSetDevice(0));
  unsigned* sink = (unsigned*)nvl_alloc(0, 1024);
  char* loc = (char*)nvl_alloc(0, BUF);
  char* rem = (char*)nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));

  G = env.sm * 4; Bk = 256;
  size_t threads = (size_t)G * Bk;

  // ------------------------------------------------ A-1: 16B 访问, 基址偏移
  hdr("A-1) 16B 连续访问, 整体基址偏移 (保持 16B 自然对齐)");
  printf("payload/kernel = %.1f MB\n", threads * (double)ITER * 16 / 1e6);
  printf("| 基址偏移 | 相对128B边界 | 远端写 GB/s | 远端读 GB/s | 本地写 GB/s | 写 vs off0 |\n");
  printf("|---|---|---|---|---|---|");
  printf("\n");
  double pl16 = (double)threads * ITER * 16;
  const int offs16[] = {0, 16, 32, 64, 112};
  double base_wr16 = 0;
  for (int i = 0; i < 5; ++i) {
    int o = offs16[i];
    double t = bench_ms([&] { k_off<16><<<G, Bk>>>(rem + o, MASK, ITER); }, 3);
    double wr = gbps(pl16, t);
    t = bench_ms([&] { k_off_rd<16><<<G, Bk>>>(rem + o, sink, MASK, ITER); }, 3);
    double rd = gbps(pl16, t);
    t = bench_ms([&] { k_off<16><<<G, Bk>>>(loc + o, MASK, ITER); }, 3);
    double wl = gbps(pl16, t);
    if (i == 0) base_wr16 = wr;
    printf("| %dB | %dB | %.1f | %.1f | %.1f | %.3f |\n", o, o % 128, wr, rd,
           wl, wr / base_wr16);
    fflush(stdout);
  }

  // ------------------------------------------------ A-2: 4B 访问, 基址偏移
  hdr("A-2) 4B 连续访问, 整体基址偏移 (保持 4B 自然对齐)");
  double pl4 = (double)threads * ITER * 4;
  printf("payload/kernel = %.1f MB\n", pl4 / 1e6);
  printf("| 基址偏移 | 相对128B边界 | 远端写 GB/s | 远端读 GB/s | 本地写 GB/s | 写 vs off0 |\n");
  printf("|---|---|---|---|---|---|\n");
  const int offs4[] = {0, 4, 8, 16, 32, 64, 124};
  double base_wr4 = 0;
  for (int i = 0; i < 7; ++i) {
    int o = offs4[i];
    double t = bench_ms([&] { k_off<4><<<G, Bk>>>(rem + o, MASK, ITER); }, 3);
    double wr = gbps(pl4, t);
    t = bench_ms([&] { k_off_rd<4><<<G, Bk>>>(rem + o, sink, MASK, ITER); }, 3);
    double rd = gbps(pl4, t);
    t = bench_ms([&] { k_off<4><<<G, Bk>>>(loc + o, MASK, ITER); }, 3);
    double wl = gbps(pl4, t);
    if (i == 0) base_wr4 = wr;
    printf("| %dB | %dB | %.1f | %.1f | %.1f | %.3f |\n", o, o % 128, wr, rd,
           wl, wr / base_wr4);
    fflush(stdout);
  }

  // ------------------------------------------------ B: 显式跨边界
  hdr("B) 显式跨边界代价 (warp 内地址连续, 每线程 8B = 2x st32)");
  printf("delta=0 时每个 8B 都在 8B 自然对齐位置(不跨任何边界);\n");
  printf("delta=4 时每 16 个线程有 1 个 8B 骑在 128B 边界上。\n");
  double plB = (double)threads * ITER * 8;
  printf("payload/kernel = %.1f MB\n", plB / 1e6);
  printf("| delta | 跨界情况 | 远端写 GB/s | 远端读 GB/s | 本地写 GB/s | 远端写相对 |\n");
  printf("|---|---|---|---|---|---|\n");

  double bB = 0;
  auto doB = [&](int delta, const char* what) {
    double t = bench_ms([&] { k_cross<<<G, Bk>>>(rem, MASK, delta, ITER); }, 3);
    double wr = gbps(plB, t);
    t = bench_ms([&] { k_cross_rd<<<G, Bk>>>(rem, sink, MASK, delta, ITER); }, 3);
    double rd = gbps(plB, t);
    t = bench_ms([&] { k_cross<<<G, Bk>>>(loc, MASK, delta, ITER); }, 3);
    double wl = gbps(plB, t);
    if (bB == 0) bB = wr;
    printf("| %d | %s | %.1f | %.1f | %.1f | %.3f |\n", delta, what, wr, rd, wl,
           wr / bB);
    fflush(stdout);
  };
  doB(0, "全部 8B 对齐, 不跨界");
  doB(4, "1/16 线程跨 128B 边界");

  // ------------------------------------------------ C: 计数器验证跨界放大
  hdr("C) 计数器验证: 跨 128B 边界是否真的多发字节");
  printf("| 位置 | payload MB | dataTx MB | dataTx/payload | rawTx MB | rawTx/payload |\n");
  printf("|---|---|---|---|---|---|\n");
  printf("(dataTx 才是净荷计数; rawTx 含 idle flit, 见 nv06 说明, 仅供参考)\n");
  auto ctr = [&](int delta, const char* what) {
    double t = bench_ms([&] { k_cross<<<G, Bk>>>(rem, MASK, delta, ITER); }, 2);
    int rep = (int)(2000.0 / t) + 1;
    if (rep > 40000) rep = 40000;
    CK(cudaDeviceSynchronize());
    NvlCounters a = nvl_read_counters(0);
    for (int r = 0; r < rep; ++r) k_cross<<<G, Bk>>>(rem, MASK, delta, ITER);
    CK(cudaDeviceSynchronize());
    NvlCounters b = nvl_read_counters(0);
    NvlCounters d = nvl_diff(a, b);
    double P = plB * rep;
    printf("| %s | %.0f | %.0f | %.3f | %.0f | %.3f |\n", what, P / 1e6,
           d.dataTx / 1e6, d.dataTx / P, d.rawTx / 1e6, d.rawTx / P);
    fflush(stdout);
  };
  ctr(0, "不跨界");
  ctr(4, "1/16 跨 128B 边界");

  printf("\n[done]\n");
  return 0;
}
