// ============================================================================
// nv21_atomic_tput —— 远端原子吞吐 vs 竞争度
//
// 核心问题:
//   1. 大量线程打远端原子, 每秒能做多少次 (Mops/s)?
//   2. 竞争度(不同目标地址数 NSLOT)从 1 (全撞同一地址) 扫到 1M (几乎无竞争),
//      吞吐曲线长什么样?
//   3. 满竞争(NSLOT=1)时的吞吐倒数 == 什么? 若 ≈ 一次 NVLink 往返(861ns)
//      -> 每次原子都要跨卡串行; 若远高于 -> 远端 L2 内部流水化执行。
//   4. 无竞争(NSLOT=1M)时能不能打满 NVLink (nv01 基线 370 GB/s)?
//
// 方法:
//   grid = 78*8 blocks, block=256 -> 159744 线程。每线程做 ITER 次原子。
//   线程 t 第 i 次的目标 slot = (t*K + i*P) % NSLOT, slot 之间隔 SLOT_STRIDE
//   字节以避免落在同一 cache line (除非我们故意想要同一 line)。
//   这里 SLOT_STRIDE=4B(紧凑, 探究 line 内合并) 和 128B(每 slot 独占 line)
//   两种布局都测, 因为 GPU L2 原子是按 sector 仲裁的, 布局会影响结论。
//
//   用 cudaEvent 计时整个 kernel, Mops/s = 总原子数 / 时间。
//   对照: 本地原子 (同样 kernel 打本地显存)。
//   另外测 red (无返回值) 版本 —— 若 red 吞吐远高于 atom, 说明返回值通道
//   是瓶颈而非原子单元本身。
// ============================================================================
#include "nvl_common.cuh"

#define ITER 64

__device__ __forceinline__ unsigned at_add(unsigned* p, unsigned v) {
  unsigned r;
  asm volatile("atom.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(r) : "l"(p), "r"(v) : "memory");
  return r;
}
__device__ __forceinline__ void rd_add(unsigned* p, unsigned v) {
  asm volatile("red.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned at_add_sys(unsigned* p, unsigned v) {
  unsigned r;
  asm volatile("atom.sys.global.add.u32 %0, [%1], %2;"
               : "=r"(r) : "l"(p), "r"(v) : "memory");
  return r;
}

// RET=1 用 atom(有返回), RET=0 用 red(无返回)
// SYS=1 用 .sys scope
// STRIDE_W = slot 之间的 uint 步长 (1 = 4B 紧凑, 32 = 128B 独占 line)
// !!! 关键: 必须打散 warp 内 lane 的目标 slot。
// 如果一个 warp 的 32 个 lane 都打同一个 slot, SM 会在 warp 内先做
// **原子聚合**(intra-warp aggregation), 32 次原子合并成 1 次发出去,
// 测出来的就不是原子单元速率而是指令发射速率(本地/远端会完全一样)。
// 做法: slot 索引 = (lane * LANE_SPREAD + 其它维) % nslot, 保证 nslot>=32
// 时 warp 内 32 个 lane 打到 32 个不同 slot; nslot<32 时最多只能打散到
// nslot 个(物理上限, 这是真实的满竞争情形, 但仍会有 32/nslot 倍的聚合,
// 需在报告里说明)。
template <int RET, int SYS>
__global__ __launch_bounds__(256) void k_tput(unsigned* base, unsigned nslot,
                                              int stride_w, unsigned* sink) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned lane = tid & 31u;
  unsigned warp = tid >> 5;
  unsigned acc = 0;
  // lane 走最低位 -> warp 内 32 lane 落到 32 个不同 slot (若 nslot>=32)
  // warp 号走高位并乘一个奇数, 让不同 warp 也错开
  unsigned s = (lane + warp * 32u) % nslot;
  unsigned step = 1u;  // 每次迭代整体平移 1 个 slot, 保持 lane 间仍互不相同
#pragma unroll 1
  for (int i = 0; i < ITER; ++i) {
    unsigned* p = base + (size_t)s * stride_w;
    if (RET) {
      unsigned r = SYS ? at_add_sys(p, 1) : at_add(p, 1);
      acc ^= r;
    } else {
      rd_add(p, 1);
    }
    s += step;
    if (s >= nslot) s -= nslot;
  }
  if (acc == 0xdeadbeefu) *sink = acc;
}

// ---------------------------------------------------------------- driver
static int G_GRID, G_BLK;
static unsigned* g_sink;

template <int RET, int SYS>
static double run(unsigned* base, unsigned nslot, int stride_w) {
  double totalOps = (double)G_GRID * G_BLK * ITER;
  double ms = bench_ms(
      [&] { k_tput<RET, SYS><<<G_GRID, G_BLK>>>(base, nslot, stride_w, g_sink); },
      3, 3);
  return totalOps / (ms * 1e-3) / 1e6;  // Mops/s
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv21_atomic_tput — %s, %d SM, %.3f GHz\n", env.name, env.sm, env.clkGHz);

  G_GRID = env.sm * 8;   // 624 blocks
  G_BLK = 256;
  double nthr = (double)G_GRID * G_BLK;
  printf("# grid=%d block=%d -> %.0f 线程, 每线程 %d 次原子, 共 %.1f M ops/kernel\n",
         G_GRID, G_BLK, nthr, ITER, nthr * ITER / 1e6);

  // slot 数最大 1M, stride 最大 32 uint (128B) -> 1M*128B = 128 MB
  const size_t BUF = 192ull << 20;
  unsigned* loc = (unsigned*)nvl_alloc(0, BUF);
  unsigned* rem = (unsigned*)nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));
  g_sink = (unsigned*)nvl_alloc(0, 256);
  CK(cudaSetDevice(0));

  const unsigned SLOTS[] = {1, 2, 4, 8, 32, 128, 1024, 65536, 1048576};
  const int NS = sizeof(SLOTS) / sizeof(SLOTS[0]);

  for (int layout = 0; layout < 2; ++layout) {
    int sw = layout == 0 ? 1 : 32;  // 4B 紧凑 vs 128B 独占 line
    char t[128];
    snprintf(t, sizeof t, "布局: slot 间隔 %dB (%s)", sw * 4,
             sw == 1 ? "同 line 内多 slot" : "每 slot 独占 128B line");
    hdr(t);
    printf("%10s %5s %13s %13s %8s | %13s %13s %8s\n", "NSLOT", "warp",
           "本地atom", "远端atom", "本/远", "本地red", "远端red", "本/远");
    printf("%10s %5s %13s %13s %8s | %13s %13s %8s\n", "", "聚合", "Mops/s",
           "Mops/s", "x", "Mops/s", "Mops/s", "x");
    for (int i = 0; i < NS; ++i) {
      unsigned n = SLOTS[i];
      // slot 覆盖字节不能超过 BUF
      if ((size_t)n * sw * 4 > BUF) { printf("%10u  (skip, 超出 buffer)\n", n); continue; }
      // warp 内 32 lane 落到 min(32,nslot) 个不同 slot -> 聚合倍数
      int agg = n >= 32 ? 1 : (int)(32 / n);
      double la = run<1, 0>(loc, n, sw);
      double ra = run<1, 0>(rem, n, sw);
      double lr = run<0, 0>(loc, n, sw);
      double rr = run<0, 0>(rem, n, sw);
      printf("%10u %5dx %13.1f %13.1f %8.2f | %13.1f %13.1f %8.2f\n", n, agg,
             la, ra, la / ra, lr, rr, lr / rr);
    }
    printf("[warp聚合] NSLOT<32 时 warp 内多个 lane 撞同一 slot, SM 会先做\n");
    printf("           intra-warp 原子聚合, 实际发到 L2 的原子数 = 名义数/聚合倍数。\n");
  }

  // ------------------------------------------------- .sys scope 对照
  hdr(".sys scope 对照 (slot 独占 128B line, atom 有返回)");
  printf("%10s %14s %14s\n", "NSLOT", "远端.gpu", "远端.sys");
  printf("%10s %14s %14s\n", "", "Mops/s", "Mops/s");
  for (int i = 0; i < NS; ++i) {
    unsigned n = SLOTS[i];
    if ((size_t)n * 32 * 4 > BUF) continue;
    double g = run<1, 0>(rem, n, 32);
    double s = run<1, 1>(rem, n, 32);
    printf("%10u %14.1f %14.1f\n", n, g, s);
  }

  // ------------------------------------------------- 换算成带宽 / 往返
  hdr("关键换算");
  {
    double ra1 = run<1, 0>(rem, 1, 32);
    double rr1 = run<0, 0>(rem, 1, 32);
    double raM = run<1, 0>(rem, 1048576, 32);
    double rrM = run<0, 0>(rem, 1048576, 32);
    double la1 = run<1, 0>(loc, 1, 32);
    double laM = run<1, 0>(loc, 1048576, 32);
    // NSLOT=1: warp 内 32 lane 全撞同一 slot -> 硬件先做 32:1 intra-warp 聚合,
    // 真正到达 L2 原子单元的请求数 = 名义 ops / 32。
    printf("满竞争 NSLOT=1 (注意: 有 32:1 intra-warp 聚合, L2 侧真实请求数为名义值/32):\n");
    printf("  远端 atom  %8.1f Mops/s(名义) -> L2侧 %8.1f Mops/s -> 每次L2原子 %7.2f ns\n",
           ra1, ra1 / 32, 32000.0 / ra1);
    printf("  远端 red   %8.1f Mops/s(名义) -> L2侧 %8.1f Mops/s -> 每次L2原子 %7.2f ns\n",
           rr1, rr1 / 32, 32000.0 / rr1);
    printf("  本地 atom  %8.1f Mops/s(名义) -> L2侧 %8.1f Mops/s -> 每次L2原子 %7.2f ns\n",
           la1, la1 / 32, 32000.0 / la1);
    printf("  NSLOT=32 (无 warp 聚合, 每 lane 独占 slot) 才是干净的满竞争对照, 见上表。\n");
    printf("  参考: 一次 NVLink 往返(nv20 单线程 atom 远端) = 861 ns\n");
    printf("  -> 若 满竞争每op间隔 << 861ns, 说明原子在远端就地流水执行,\n");
    printf("     而不是每次把 line 拉回本地做。\n");
    printf("无竞争 NSLOT=1M:\n");
    printf("  远端 atom  %8.1f Mops/s = %7.1f GB/s(按4B请求算) \n", raM, raM * 4 / 1e3);
    printf("  远端 red   %8.1f Mops/s = %7.1f GB/s\n", rrM, rrM * 4 / 1e3);
    printf("  本地 atom  %8.1f Mops/s = %7.1f GB/s\n", laM, laM * 4 / 1e3);
    printf("  参考: nv01 NVLink 远端写基线 370 GB/s (16B/线程 store)\n");
  }

  printf("\n[done]\n");
  return 0;
}
