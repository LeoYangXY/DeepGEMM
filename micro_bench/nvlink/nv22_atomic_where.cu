// ============================================================================
// nv22_atomic_where —— 远端原子在哪一侧执行?
//
// 待判定的两个假说:
//   H_local  : "拉回来做" —— 远端 atomicAdd 把 cache line 拉到本地 L2,
//              在本地做完 RMW 再写回。特征: (a) 满竞争时可以在本地 L2 里
//              连续做很多次才写回 -> 吞吐极高; (b) wire 流量 << N*4B。
//   H_remote : "送过去做" —— 请求包发到远端 L2, 在远端 ROP/L2 原子单元
//              就地 RMW, 结果(或 ack)回来。特征: (a) 每次都过线;
//              (b) wire Tx ≈ N*包大小, Rx ≈ N*响应包; (c) red 无返回值时
//              Rx 应显著小于 atom。
//
// 三条独立证据:
//   (A) 满竞争吞吐 vs NVLink 往返倒数。
//       nv20 实测远端单次 atom 往返 = 1705 cyc = 861 ns。
//       若 H_local 成立(拉回本地做), 同一 line 被本地 L2 独占后, 后续原子
//       都是本地 L2 延迟(~270ns 串行, 但流水后可到几 ns/op) -> 满竞争吞吐
//       会远高于 1/861ns = 1.16 Mops/s。
//       若 H_remote 成立, 每次原子都要过线, 但远端 L2 可以流水化处理来自
//       同一 line 的连续请求 -> 吞吐 = 远端 L2 原子单元的流水速率(远高于
//       1.16 Mops/s), 而**延迟**仍是 861ns。所以 (A) 单独不能区分,
//       必须配合 (B)。
//
//   (B) wire byte 计数器 (决定性证据)。
//       跑 N 次远端 atomicAdd(4B) 到**同一个地址**。
//       H_local: line 被拉到本地后 N 次原子都在本地做, wire 流量 ~ 常数
//                (只有最初的 line fetch + 最后的 writeback), 与 N 无关。
//       H_remote: wire Tx ∝ N, Rx ∝ N。
//       -> 扫 N, 看 wire bytes 是否线性于 N。这是二选一的判决。
//
//   (C) red (无返回) vs atom (有返回) 的 Rx 流量比。
//       若在远端执行: atom 需要把旧值送回 -> Rx 有 N 个响应包;
//       red 不需要 -> Rx 应明显更小。
//       若在本地执行: 两者 wire 行为应几乎一样(都只是 line 搬运)。
//
// 注意: NVLink 计数器是全卡累积, 且 nvidia-smi 采样有延迟, 所以
//   - 每次测量前后各 sleep 一下让计数器刷新
//   - 用足够大的 N (>=1e8 ops) 让信号远超背景噪声
//   - 同时读 GPU0(发起方) 和 GPU1(目标方) 的计数器交叉验证
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"
#include <unistd.h>

__device__ __forceinline__ unsigned at_add(unsigned* p, unsigned v) {
  unsigned r;
  asm volatile("atom.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(r) : "l"(p), "r"(v) : "memory");
  return r;
}
__device__ __forceinline__ void rd_add(unsigned* p, unsigned v) {
  asm volatile("red.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// 所有线程都打 base[0] (满竞争, 单地址)  —— 用于 (B)(C)
// RET=1 atom, RET=0 red
template <int RET>
__global__ __launch_bounds__(256) void k_same(unsigned* base, int iter,
                                              unsigned* sink) {
  unsigned acc = 0;
#pragma unroll 1
  for (int i = 0; i < iter; ++i) {
    if (RET) acc ^= at_add(base, 1);
    else rd_add(base, 1);
  }
  if (acc == 0xdeadbeefu) *sink = acc;
}

// 每线程打自己独占的 128B line (无竞争) —— 对照, 用于确认计数器灵敏度
template <int RET>
__global__ __launch_bounds__(256) void k_uniq(unsigned* base, int iter,
                                              unsigned* sink) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned* p = base + (size_t)tid * 32;  // 128B 间隔
  unsigned acc = 0;
#pragma unroll 1
  for (int i = 0; i < iter; ++i) {
    if (RET) acc ^= at_add(p, 1);
    else rd_add(p, 1);
  }
  if (acc == 0xdeadbeefu) *sink = acc;
}

// 纯 store 对照: N 次 4B store 到同一远端地址
__global__ __launch_bounds__(256) void k_st_same(unsigned* base, int iter) {
  for (int i = 0; i < iter; ++i) st32(base, i);
}

// ---------------------------------------------------------------- driver
static unsigned* g_sink;
static int G_GRID = 0, G_BLK = 256;

struct WireRes { double ops; double ms; double txMB, rxMB; double peerTxMB, peerRxMB; };

// 跑一个 kernel, 同时抓 GPU0 和 GPU1 的 NVLink 计数器
template <typename F>
static WireRes wire_run(F&& launch, double ops) {
  CK(cudaDeviceSynchronize());
  usleep(400000);  // 让计数器稳定
  NvlCounters a0 = nvl_read_counters(0), a1 = nvl_read_counters(1);
  Timer t; t.start();
  launch();
  double ms = t.stop_ms();
  CK(cudaDeviceSynchronize());
  CKLAST();
  usleep(400000);
  NvlCounters b0 = nvl_read_counters(0), b1 = nvl_read_counters(1);
  NvlCounters d0 = nvl_diff(a0, b0), d1 = nvl_diff(a1, b1);
  WireRes r;
  r.ops = ops; r.ms = ms;
  r.txMB = d0.dataTx / 1e6; r.rxMB = d0.dataRx / 1e6;
  r.peerTxMB = d1.dataTx / 1e6; r.peerRxMB = d1.dataRx / 1e6;
  return r;
}

static void wrow(const char* tag, WireRes r) {
  double bpo_tx = r.ops ? r.txMB * 1e6 / r.ops : 0;
  double bpo_rx = r.ops ? r.rxMB * 1e6 / r.ops : 0;
  printf("%-26s %10.1f %9.2f | %9.1f %9.1f | %8.2f %8.2f | %9.1f %9.1f\n", tag,
         r.ops / 1e6, r.ms, r.txMB, r.rxMB, bpo_tx, bpo_rx, r.peerTxMB, r.peerRxMB);
}
static void whdr() {
  printf("%-26s %10s %9s | %9s %9s | %8s %8s | %9s %9s\n", "case", "Mops", "ms",
         "G0 dTx MB", "G0 dRx MB", "B/op Tx", "B/op Rx", "G1 dTx MB", "G1 dRx MB");
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  G_GRID = env.sm * 4;
  printf("# nv22_atomic_where — %s, %d SM, %.3f GHz\n", env.name, env.sm, env.clkGHz);
  printf("# GPU0 发起, 目标在 GPU1。计数器 Data bytes = 协议净荷。\n");
  printf("# 注意: 计数器是全卡累积值且有 ~100ms 级采样粒度, 已用 400ms sleep 隔离。\n");

  const size_t BUF = 512ull << 20;  // uniq 模式: 78*4*256 线程 *128B = 10MB, 够
  unsigned* loc = (unsigned*)nvl_alloc(0, BUF);
  unsigned* rem = (unsigned*)nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));
  g_sink = (unsigned*)nvl_alloc(0, 256);
  CK(cudaSetDevice(0));

  double nthr = (double)G_GRID * G_BLK;
  printf("# grid=%d block=%d -> %.0f 线程\n", G_GRID, G_BLK, nthr);

  // ========================================================= 证据 A
  hdr("证据 A) 满竞争(单地址)吞吐 vs 单次往返延迟");
  {
    const int IT = 2000;
    double ops = nthr * IT;
    // 远端 atom 满竞争
    double ms_ra = bench_ms([&] { k_same<1><<<G_GRID, G_BLK>>>(rem, IT, g_sink); }, 2, 3);
    double ms_rr = bench_ms([&] { k_same<0><<<G_GRID, G_BLK>>>(rem, IT, g_sink); }, 2, 3);
    double ms_la = bench_ms([&] { k_same<1><<<G_GRID, G_BLK>>>(loc, IT, g_sink); }, 2, 3);
    double ms_lr = bench_ms([&] { k_same<0><<<G_GRID, G_BLK>>>(loc, IT, g_sink); }, 2, 3);
    printf("%-30s %12s %14s %14s\n", "case", "Mops/s", "ns/op(串行化)", "vs 861ns往返");
    struct { const char* n; double ms; } rows[] = {
        {"远端 atom 满竞争", ms_ra}, {"远端 red  满竞争", ms_rr},
        {"本地 atom 满竞争", ms_la}, {"本地 red  满竞争", ms_lr}};
    for (auto& x : rows) {
      double mops = ops / (x.ms * 1e-3) / 1e6;
      double nsop = 1000.0 / mops;
      printf("%-30s %12.1f %14.4f %14.1fx 更快\n", x.n, mops, nsop, 861.0 / nsop);
    }
    printf("\n[读法] 若 ns/op 远小于 861ns, 说明来自不同线程的原子请求被\n");
    printf("       **流水化**了(不是每次都等一个完整往返)。但这不能区分\n");
    printf("       流水发生在本地还是远端 —— 需要证据 B。\n");
  }

  // ========================================================= 证据 B
  hdr("证据 B) wire byte 计数 —— N 次单地址远端原子, 流量是否 ∝ N ?");
  printf("[判决] H_local(拉回本地做): wire 流量与 N 无关(常数);\n");
  printf("       H_remote(远端就地做): wire Tx/Rx 都 ∝ N, B/op 为常数。\n\n");
  whdr();
  {
    // 用不同 iter 让总 ops 变化 4 倍, 看流量是否同比例变化
    int iters[] = {500, 1000, 2000, 4000};
    for (int it : iters) {
      double ops = nthr * it;
      char tag[64];
      snprintf(tag, sizeof tag, "远端 atom 单地址 it=%d", it);
      wrow(tag, wire_run([&] { k_same<1><<<G_GRID, G_BLK>>>(rem, it, g_sink); }, ops));
    }
    printf("\n");
    for (int it : iters) {
      double ops = nthr * it;
      char tag[64];
      snprintf(tag, sizeof tag, "远端 red  单地址 it=%d", it);
      wrow(tag, wire_run([&] { k_same<0><<<G_GRID, G_BLK>>>(rem, it, g_sink); }, ops));
    }
  }

  // ========================================================= 证据 C
  hdr("证据 C) atom(有返回) vs red(无返回) 的 Rx 流量对比 + 无竞争对照");
  printf("[判决] 若远端执行: atom 的 G0-Rx (响应包) 应显著 > red 的 G0-Rx。\n\n");
  whdr();
  {
    const int IT = 2000;
    double ops = nthr * IT;
    wrow("远端 atom 单地址", wire_run([&] { k_same<1><<<G_GRID, G_BLK>>>(rem, IT, g_sink); }, ops));
    wrow("远端 red  单地址", wire_run([&] { k_same<0><<<G_GRID, G_BLK>>>(rem, IT, g_sink); }, ops));
    wrow("远端 st32 单地址", wire_run([&] { k_st_same<<<G_GRID, G_BLK>>>(rem, IT); }, ops));
    const int IT2 = 500;
    double ops2 = nthr * IT2;
    wrow("远端 atom 无竞争(独占line)", wire_run([&] { k_uniq<1><<<G_GRID, G_BLK>>>(rem, IT2, g_sink); }, ops2));
    wrow("远端 red  无竞争(独占line)", wire_run([&] { k_uniq<0><<<G_GRID, G_BLK>>>(rem, IT2, g_sink); }, ops2));
    printf("\n本地对照(不应产生 NVLink 流量, 用于确认计数器背景噪声):\n");
    wrow("本地 atom 单地址", wire_run([&] { k_same<1><<<G_GRID, G_BLK>>>(loc, IT, g_sink); }, ops));
  }

  printf("\n[done]\n");
  return 0;
}
