// ============================================================================
// nv13_fifo_burst —— burst / idle / burst 测缓冲深度与排空速率
//
// 思路 (经典的 FIFO 深度探测):
//   发一个长度 B 的远端 store burst -> 等 S 纳秒 -> 再发一个 burst,
//   **逐条记录第二个 burst 中每条 store 的发射间隔 cycle**。
//
//   store 是 fire-and-forget: 只要下游缓冲(write buffer / NVLink credit FIFO)
//   还有空位, st 指令就能立刻退休 -> 间隔很小(只有指令发射开销)。
//   一旦缓冲填满, 后续 st 必须等一个 credit 回来才能进 -> 间隔跳到"后端服务
//   间隔"(= 1/后端吞吐)。
//   => 前 N 条便宜、第 N+1 条开始变贵 的那个 N 就是 **缓冲深度(以事务计)**。
//
// !! v1 失败教训 (见 raw_logs/nv13_fifo_burst.v1.log):
//   v1 只用 1 个 warp 发 burst。单 warp 发射速率仅 ~15.8 GB/s(实测 64 cyc/512B),
//   而 NVLink 后端排空速率是 370 GB/s —— 慢了 23 倍, 缓冲**永远填不满**,
//   所有 (B,S) 组合都得到恒定 64 cyc, knee = none。这是"探测器比被测对象弱"
//   的典型错误。
//   v2 修正: 用 GRID*1024 个线程同时 burst, 聚合发射速率远超后端, 才能压出
//   back-pressure。仍只由 block0/lane0 采时间戳。
//
// 实现要点:
//   - 时间戳先存 shared memory, kernel 末尾一次性写回 global。
//   - 全 grid 软 barrier(atomic 计数) 保证所有 block 同步开始 burst。
//   - 地址按 nthr 步进, 每条 st 都是全新的 128B 行, 不复用不命中。
//   - __nanosleep 制造精确空闲窗口。
// ============================================================================
#include "nvl_common.cuh"

#define MAXB 1024
#define BURST_BLK 1024   // 每 block 32 warp

__global__ __launch_bounds__(BURST_BLK) void k_burst(uint4* __restrict__ dst,
                                                     long long* __restrict__ tout,
                                                     int B, unsigned sleep_ns,
                                                     unsigned* bar, int nblk) {
  __shared__ long long ts[MAXB + 1];
  int tid = threadIdx.x;
  int gtid = blockIdx.x * BURST_BLK + tid;
  long long nthr = (long long)nblk * BURST_BLK;
  uint4 v = make_uint4(gtid, 2, 3, 4);
  uint4* p = dst + gtid;

  // ---- 全 grid 软 barrier: 保证所有 block 同时开始 burst ----
  __syncthreads();
  if (tid == 0) {
    atomicAdd(bar, 1u);
    long long g = 0;
    while (atomicAdd(bar, 0u) < (unsigned)nblk && ++g < 50000000LL) {}
  }
  __syncthreads();

  // ---------------- burst #1: 把缓冲填满 ----------------
#pragma unroll 1
  for (int i = 0; i < B; ++i) st128(p + (long long)i * nthr, v);

  // ---------------- 空闲 S ns ----------------
  if (sleep_ns) __nanosleep(sleep_ns);
  __syncthreads();

  // ---------------- burst #2: 逐条记时 ----------------
  uint4* q = dst + (long long)MAXB * nthr + gtid;
  bool rec = (blockIdx.x == 0 && tid == 0);
  long long t = clk();
  if (rec) ts[0] = t;
#pragma unroll 1
  for (int i = 0; i < B; ++i) {
    st128(q + (long long)i * nthr, v);
    long long c = clk();
    if (rec) ts[i + 1] = c;
  }
  __syncthreads();
  if (rec)
    for (int i = 0; i <= B; ++i) tout[i] = ts[i];
}

// ============================================================================
// v3 追加: 基于 **load** 的 burst 探测
//
// 为什么必须换成 load: st128 是 fire-and-forget, 没有目的寄存器, clock64 与
// store 的"完成"之间不存在任何数据依赖 -> 记录 lane 会跑在真实流量前面,
// 测出的 d[i] 只是指令发射槽间隔, 甚至能算出 >370GB/s 的不可能速率
// (见 v2 E 段: grid=1 算出 146.8 GB/s, 但 nv09 证明单 SM 上限 50.4 GB/s)。
//
// load 版: 每条 ld 的返回值参与下一次 clock64 之前的一个假依赖
//   (用 inline asm 把 acc 喂进一个 0 代价的运算), 强制 clk() 排在 ld 返回之后。
//   这样 d[i] 才真正是"第 i 个请求的完成间隔"。
//   前 N 个请求可以全部并发挂起 -> d 小; 第 N+1 个开始必须等前面回来 -> d 跳高。
// ============================================================================
__global__ __launch_bounds__(BURST_BLK) void k_burst_ld(
    const uint4* __restrict__ src, long long* __restrict__ tout, int B,
    unsigned sleep_ns, unsigned* bar, int nblk, unsigned* sink) {
  __shared__ long long ts[MAXB + 1];
  int tid = threadIdx.x;
  int gtid = blockIdx.x * BURST_BLK + tid;
  long long nthr = (long long)nblk * BURST_BLK;
  const uint4* p = src + gtid;

  __syncthreads();
  if (tid == 0) {
    atomicAdd(bar, 1u);
    long long g = 0;
    while (atomicAdd(bar, 0u) < (unsigned)nblk && ++g < 50000000LL) {}
  }
  __syncthreads();

  unsigned acc = 0;
  // burst #1
#pragma unroll 1
  for (int i = 0; i < B; ++i) { uint4 v = ld128_v(p + (long long)i * nthr); acc ^= v.x; }
  if (sleep_ns) __nanosleep(sleep_ns);
  __syncthreads();

  // burst #2, 逐条记时; clk() 通过 acc 依赖排在 ld 返回之后
  const uint4* q = src + (long long)MAXB * nthr + gtid;
  bool rec = (blockIdx.x == 0 && tid == 0);
  long long t = clk();
  if (rec) ts[0] = t;
#pragma unroll 1
  for (int i = 0; i < B; ++i) {
    uint4 v = ld128_v(q + (long long)i * nthr);
    acc ^= v.x;
    long long c;
    // 把 acc 作为输入喂给 clock64 前的一条假指令, 建立真依赖
    asm volatile("{ .reg .u32 t; and.b32 t, %1, 0; mov.u64 %0, %%clock64; add.u64 %0, %0, 0; }"
                 : "=l"(c) : "r"(acc) : "memory");
    if (rec) ts[i + 1] = c;
  }
  __syncthreads();
  if (rec)
    for (int i = 0; i <= B; ++i) tout[i] = ts[i];
  if (acc == 0xdeadbeefu) *sink = acc;
}

// ---------------------------------------------------------------- helpers
static long long h[MAXB + 8];

// 跑一次 (B,S,grid), 把 best-of-tries 的逐条间隔填进 d[], 返回 burst 总 cycle
static double measure(uint4* dst, long long* tout, unsigned* bar, int B,
                      unsigned S, int grid, double* d) {
  double bestTot = 1e30;
  for (int r = 0; r < 6; ++r) {
    CK(cudaMemset(tout, 0, sizeof(long long) * (B + 2)));
    CK(cudaMemset(bar, 0, 4));
    k_burst<<<grid, BURST_BLK>>>(dst, tout, B, S, bar, grid);
    CK(cudaDeviceSynchronize());
    CKLAST();
    CK(cudaMemcpy(h, tout, sizeof(long long) * (B + 1), cudaMemcpyDeviceToHost));
    double tot = (double)(h[B] - h[0]);
    if (r > 0 && tot > 0 && tot < bestTot) {
      bestTot = tot;
      for (int i = 0; i < B; ++i) d[i] = (double)(h[i + 1] - h[i]);
    }
  }
  return bestTot;
}

static double measure_ld(const uint4* src, long long* tout, unsigned* bar,
                         int B, unsigned S, int grid, unsigned* sink,
                         double* d) {
  double bestTot = 1e30;
  for (int r = 0; r < 6; ++r) {
    CK(cudaMemset(tout, 0, sizeof(long long) * (B + 2)));
    CK(cudaMemset(bar, 0, 4));
    k_burst_ld<<<grid, BURST_BLK>>>(src, tout, B, S, bar, grid, sink);
    CK(cudaDeviceSynchronize());
    CKLAST();
    CK(cudaMemcpy(h, tout, sizeof(long long) * (B + 1), cudaMemcpyDeviceToHost));
    double tot = (double)(h[B] - h[0]);
    if (r > 0 && tot > 0 && tot < bestTot) {
      bestTot = tot;
      for (int i = 0; i < B; ++i) d[i] = (double)(h[i + 1] - h[i]);
    }
  }
  return bestTot;
}

// 找 knee: 间隔首次持续(连续3条)超过前段基线的 thr 倍
static int find_knee(const double* d, int B, double thr, double* baseOut) {
  int nb = B < 4 ? B : 4;
  double base = 0;
  for (int i = 0; i < nb; ++i) base += d[i];
  base /= nb;
  if (baseOut) *baseOut = base;
  for (int i = nb; i + 2 < B; ++i)
    if (d[i] > base * thr && d[i + 1] > base * thr && d[i + 2] > base * thr)
      return i;
  return -1;
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  // GRID 选择: nv11 测出总在途容量 ~770 KB。若每个 burst step 的流量远大于
  // 770KB, knee 会落在 N=1, 分辨率不够。取 GRID=4 -> 每 step = 4096thr*16B
  // = 64 KB, 770/64 ≈ 12 step 的分辨率, 合适。
  // 同时 4 个 SM 的聚合发射速率 ≈ 4*50 = 200 GB/s, 仍低于 370 后端 ->
  // 单靠 4 SM 压不满。故用 GRID=12 (≈ nv10 的饱和点), 每 step 192 KB,
  // 聚合发射 ~365 GB/s ≈ 后端速率, 是最接近 back-pressure 边界的配置。
  const int GRID = 12;
  const long long NTHR = (long long)GRID * BURST_BLK;
  printf("# nv13_fifo_burst — %s, %.3f GHz\n", env.name, env.clkGHz);
  printf("# v2: grid=%d x block=%d = %lld 线程同时 burst (聚合发射速率 >> 后端)\n",
         GRID, BURST_BLK, NTHR);
  printf("# 每个 burst step = %lld 线程 x 16B = %.1f KB 的远端流量\n", NTHR,
         NTHR * 16.0 / 1024.0);
  printf("# 只有 block0/lane0 采 clock64, 存 shared, kernel 末尾一次性写回\n");

  // 缓冲按 E 段最大 grid=32 来分配: 2*MAXB*(32*1024)*16B = 1 GiB
  const long long MAXTHR = 32ll * BURST_BLK;
  const size_t BUF = 2ull * MAXB * MAXTHR * 16ull + (64ull << 20);
  printf("# 缓冲 %.2f GiB/卡\n", BUF / 1073741824.0);

  uint4* remote = (uint4*)nvl_alloc(1, BUF);
  uint4* local = (uint4*)nvl_alloc(0, BUF);
  CK(cudaSetDevice(0));
  long long* tout = (long long*)nvl_alloc(0, sizeof(long long) * (MAXB + 8));
  unsigned* bar = (unsigned*)nvl_alloc(0, 256);
  CK(cudaSetDevice(0));

  static double d[MAXB + 8];
  int Bs[] = {8, 16, 32, 64, 128, 256, 512, 1024};
  unsigned Ss[] = {0, 100, 1000, 10000};
  int NB = sizeof(Bs) / sizeof(int), NS = sizeof(Ss) / sizeof(unsigned);

  // ============================================================ A
  hdr("A) 远端 GPU0->GPU1: 第二个 burst 内每条 store 的发射间隔 (B x S 扫描)");
  printf("每 burst step 搬 %.0f KB。knee N = 间隔首次连续3条 >2x 基线的下标\n",
         NTHR * 16.0 / 1024.0);
  printf("%6s %8s %11s %11s %11s %9s %13s %12s\n", "B", "S(ns)", "base cyc",
         "d[0..7]", "d[last8]", "knee N", "knee 字节(KB)", "总cyc");
  for (int bi = 0; bi < NB; ++bi) {
    for (int si = 0; si < NS; ++si) {
      int B = Bs[bi]; unsigned S = Ss[si];
      double tot = measure(remote, tout, bar, B, S, GRID, d);
      int nh = B < 8 ? B : 8;
      double f = 0, l = 0;
      for (int i = 0; i < nh; ++i) f += d[i];
      for (int i = B - nh; i < B; ++i) l += d[i];
      f /= nh; l /= nh;
      double base; int knee = find_knee(d, B, 2.0, &base);
      char kb[24], kn[16];
      if (knee < 0) { snprintf(kn, 16, "none"); snprintf(kb, 24, "-"); }
      else { snprintf(kn, 16, "%d", knee);
             snprintf(kb, 24, "%.0f", knee * NTHR * 16.0 / 1024.0); }
      printf("%6d %8u %11.1f %11.1f %11.1f %9s %13s %12.0f\n", B, S, base, f, l,
             kn, kb, tot);
    }
  }

  // ============================================================ B
  hdr("B) 逐条明细 B=512: 看间隔序列长什么样 (前 48 条)");
  for (int si = 0; si < NS; ++si) {
    unsigned S = Ss[si];
    int B = 512;
    measure(remote, tout, bar, B, S, GRID, d);
    printf("\nS=%-6uns d[0..47]: ", S);
    for (int i = 0; i < 48; ++i) printf("%.0f ", d[i]);
    double m = 0; int c = 0;
    for (int i = B - 64; i < B; ++i) { m += d[i]; ++c; }
    m /= c;
    double ns = cyc2ns(m, env.clkGHz);
    printf("\n           稳态(最后64条)平均 %.1f cyc = %.2f ns/step; "
           "聚合服务速率 = %.1f GB/s\n", m, ns, NTHR * 16.0 / ns);
  }

  // ============================================================ C
  hdr("C) 本地 HBM 对照 (同 grid/block, 看 knee 是否 NVLink 专属)");
  printf("%6s %8s %11s %11s %11s %9s %12s\n", "B", "S(ns)", "base cyc",
         "d[0..7]", "d[last8]", "knee N", "总cyc");
  for (int bi = 0; bi < NB; ++bi) {
    int B = Bs[bi]; unsigned S = 0;
    double tot = measure(local, tout, bar, B, S, GRID, d);
    int nh = B < 8 ? B : 8;
    double f = 0, l = 0;
    for (int i = 0; i < nh; ++i) f += d[i];
    for (int i = B - nh; i < B; ++i) l += d[i];
    f /= nh; l /= nh;
    double base; int knee = find_knee(d, B, 2.0, &base);
    printf("%6d %8u %11.1f %11.1f %11.1f %9d %12.0f\n", B, S, base, f, l, knee, tot);
  }

  // ============================================================ D
  hdr("D) 排空实验: 固定 B=512, 细扫空闲时间 S, 看 knee/首条间隔如何恢复");
  printf("若 S 越大 -> 第一条间隔越小 / knee 越靠后, 说明积压在空闲期被排空\n");
  printf("%10s %11s %11s %11s %9s %12s\n", "S(ns)", "d[0]", "d[0..7]",
         "d[last8]", "knee N", "总cyc");
  unsigned Sd[] = {0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000};
  for (int i = 0; i < (int)(sizeof(Sd) / sizeof(unsigned)); ++i) {
    int B = 512;
    double tot = measure(remote, tout, bar, B, Sd[i], GRID, d);
    double f = 0, l = 0;
    for (int j = 0; j < 8; ++j) f += d[j];
    for (int j = B - 8; j < B; ++j) l += d[j];
    f /= 8; l /= 8;
    double base; int knee = find_knee(d, B, 2.0, &base);
    printf("%10u %11.1f %11.1f %11.1f %9d %12.0f\n", Sd[i], d[0], f, l, knee, tot);
  }

  // ============================================================ E
  hdr("E) 关键: 扫 GRID —— knee 的字节位置是否与 grid 无关(=真实硬件容量)?");
  printf("若 knee_N * step_bytes 在不同 grid 下收敛到同一个字节数, 那就是缓冲容量\n");
  printf("%6s %12s %11s %11s %9s %14s %12s\n", "grid", "step KB", "base cyc",
         "稳态 cyc", "knee N", "knee 累计KB", "发射GB/s");
  int gs[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
  for (int i = 0; i < (int)(sizeof(gs) / sizeof(int)); ++i) {
    int g = gs[i], B = 512;
    long long nt = (long long)g * BURST_BLK;
    double stepKB = nt * 16.0 / 1024.0;
    measure(remote, tout, bar, B, 0, g, d);
    double base; int knee = find_knee(d, B, 2.0, &base);
    double m = 0; for (int j = B - 64; j < B; ++j) m += d[j];
    m /= 64;
    double ns = cyc2ns(m, env.clkGHz);
    printf("%6d %12.1f %11.1f %11.1f %9d %14.1f %12.1f\n", g, stepKB, base, m,
           knee, knee < 0 ? -1.0 : knee * stepKB, nt * 16.0 / ns);
  }

  // ============================================================ F (v3 主结果)
  hdr("F) LOAD 版 burst (有真依赖, 可信): 单 warp grid=1 block=1024 逐条完成间隔");
  printf("store 版无法测(fire-and-forget, clk 与完成无依赖, 见 E 段矛盾数据)。\n");
  printf("load 版 clk() 通过返回值建立真依赖 -> d[i] 是真实完成间隔。\n");
  printf("%6s %8s %12s %11s %11s %9s %14s\n", "grid", "S(ns)", "step KB",
         "base cyc", "稳态 cyc", "knee N", "knee 累计KB");
  int gsl[] = {1, 2, 4, 8, 12};
  unsigned Sl[] = {0, 10000};
  for (int gi = 0; gi < (int)(sizeof(gsl) / sizeof(int)); ++gi) {
    for (int si = 0; si < 2; ++si) {
      int g = gsl[gi], B = 256;
      long long nt = (long long)g * BURST_BLK;
      double stepKB = nt * 16.0 / 1024.0;
      measure_ld(remote, tout, bar, B, Sl[si], g, (unsigned*)bar + 8, d);
      double base; int knee = find_knee(d, B, 2.0, &base);
      double m = 0; for (int j = B - 32; j < B; ++j) m += d[j];
      m /= 32;
      printf("%6d %8u %12.1f %11.1f %11.1f %9d %14.1f\n", g, Sl[si], stepKB,
             base, m, knee, knee < 0 ? -1.0 : knee * stepKB);
    }
  }
  hdr("F2) LOAD 版逐条明细 grid=1 (16KB/step), S=0 vs 10us, 前 48 条");
  for (int si = 0; si < 2; ++si) {
    int B = 256;
    measure_ld(remote, tout, bar, B, Sl[si], 1, (unsigned*)bar + 8, d);
    printf("\nS=%-6uns d[0..47]: ", Sl[si]);
    for (int i = 0; i < 48; ++i) printf("%.0f ", d[i]);
    double m = 0; for (int j = B - 32; j < B; ++j) m += d[j];
    m /= 32;
    printf("\n           稳态平均 %.1f cyc = %.1f ns/16KB -> %.1f GB/s (对照 nv09 单SM读 %s)\n",
           m, cyc2ns(m, env.clkGHz), 16384.0 / cyc2ns(m, env.clkGHz), "50.5GB/s");
  }

  printf("\n[读法] knee N = 缓冲填满前能免费塞进去的 burst step 数,\n");
  printf("       N * step_bytes = 缓冲字节容量; E/F 段跨 grid 的收敛值最可信。\n");
  printf("       S 增大后 knee N 变大/d[0] 变小 => 空闲期缓冲被排空了。\n");
  printf("[警告] A-E 段用 store, 已被 E 段自证不可信(算出 >370GB/s 的发射速率);\n");
  printf("       结论请以 F 段 load 版为准。\n");
  printf("\n[done]\n");
  return 0;
}
