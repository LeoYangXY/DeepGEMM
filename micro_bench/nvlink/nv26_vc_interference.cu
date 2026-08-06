// ============================================================================
// nv26_vc_interference —— NVLink 流量类型干扰矩阵 / 虚拟通道(VC)推断
//
// 三类远端流量:  STORE(写) / LOAD(读) / ATOMIC(red.global.add, 无返回)
//
// 实验骨架:
//   Phase A: 每类单独跑, 记 T_solo
//   Phase B: 两两并发(两个 host 线程各自 device/stream, 同为 GPU0 发起, 目标
//            GPU1 的两块不相交内存), 记 T_pair
//   Phase C: 【核心】节流版。把每类流量强度压到 ~30% 带宽(靠插入 delay loop),
//            再做两两并发。此时链路远未打满:
//              - 若仍互相拖累 -> 队列/VC 共享 (结构性冲突)
//              - 若各自保住 ~100% -> 只是带宽竞争
//
// 计时姿势: 两个 stream 都 launch 出去 -> 各自 event 测单流时间;
//           墙钟测聚合。kernel 用固定 REP 轮保证 >=50ms。
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------- 节流器
// 用「活跃 warp 数」调节注入强度, 而不是 spin 延迟循环:
//   nwarp = blockDim.x/32 时全速; nwarp 越小, 并发访存请求越少 -> 注入率线性下降。
// 好处: 不增加 SM 上的额外指令压力, 也不改变 block 驻留情况,
//       测到的纯粹是「链路上负载低了」这一个变量。
// 非活跃 warp 直接 return, 不发任何访存。
__device__ __forceinline__ bool warp_active(int nwarp) {
  return (threadIdx.x >> 5) < nwarp;
}

// 远端写: grid-stride, REP 轮。thr = 活跃 warp 数 (0 表示全部活跃)
__global__ void k_store(uint4* __restrict__ dst, size_t n, int rep, int thr,
                        long long* sink) {
  if (thr > 0 && !warp_active(thr)) return;
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += stride) st128(dst + i, v);
}

// 远端读
__global__ void k_load(const uint4* __restrict__ src, size_t n, int rep, int thr,
                       long long* sink) {
  if (thr > 0 && !warp_active(thr)) return;
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += stride) {
      uint4 v = ld128_v(src + i);
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  if (acc.x == 0xdeadbeefu && acc.y == 0x12345678u) *sink = acc.z;
}

// 远端 atomic: red.global.add.u32 (fire-and-forget, 无返回值)
// 每线程打自己的 slot, 避免争同一 cacheline -> 测的是链路 atomic 吞吐不是冲突
__global__ void k_atomic(unsigned* __restrict__ dst, size_t n, int rep, int thr,
                         long long* sink) {
  if (thr > 0 && !warp_active(thr)) return;
  size_t i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int r = 0; r < rep; ++r)
    for (size_t i = i0; i < n; i += stride)
      asm volatile("red.global.add.u32 [%0], 1;" ::"l"(dst + i) : "memory");
}

enum Kind { K_ST = 0, K_LD = 1, K_AT = 2 };
static const char* kname[3] = {"STORE", "LOAD", "ATOMIC"};

struct Cfg {
  int grid, blk, rep, thr;
  size_t nelem;       // 元素数(uint4 或 uint32)
  size_t bytes_pass;  // 一轮逻辑字节
};

static Cfg g_cfg[3];

// 在指定 stream 上发一个 kind 的 kernel
static void launch(int kind, void* buf, long long* sink, cudaStream_t s) {
  const Cfg& c = g_cfg[kind];
  if (kind == K_ST)
    k_store<<<c.grid, c.blk, 0, s>>>((uint4*)buf, c.nelem, c.rep, c.thr, sink);
  else if (kind == K_LD)
    k_load<<<c.grid, c.blk, 0, s>>>((const uint4*)buf, c.nelem, c.rep, c.thr, sink);
  else
    k_atomic<<<c.grid, c.blk, 0, s>>>((unsigned*)buf, c.nelem, c.rep, c.thr, sink);
}

// 节流时非活跃 warp 不发访存, 实际搬运字节按活跃 warp 比例缩放
static double total_bytes(int kind) {
  const Cfg& c = g_cfg[kind];
  double frac = 1.0;
  int wpb = c.blk / 32;
  if (c.thr > 0 && c.thr < wpb) frac = (double)c.thr / wpb;
  return (double)c.bytes_pass * c.rep * frac;
}

int main() {
  NvlEnv env = nvl_init(2);
  printf("# nv26_vc_interference — %s x%d, %d SM, %.3f GHz\n", env.name,
         env.ndev, env.sm, env.clkGHz);
  nvl_enable_peers(env.ndev);

  const size_t BUF = 128ull << 20;  // 每个流量类各自一块 128MB 远端 buffer

  // GPU1 上开 2 块不相交 buffer(给并发的两条流各用一块) x 3 类 = 复用即可
  CK(cudaSetDevice(1));
  void* rbuf[2];
  for (int k = 0; k < 2; ++k) rbuf[k] = nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));
  long long* sink = (long long*)nvl_alloc(0, 4096);
  CK(cudaSetDevice(0));

  // ---- 配置: STORE/LOAD 用 uint4, ATOMIC 用 uint32(4B/次)
  // 【关键】grid 只用 SM 数的一半(39*2=78 block, 每 SM 1 block),
  // 这样两条并发流各占一半 SM, 都能真正同时驻留, 不会互相排队等 SM 槽位。
  // 否则测到的是 SM 调度竞争而不是链路竞争。
  int blk = 256, grid = env.sm;      // 78 block, 每 SM 恰好 1 个
  size_t nV = BUF / sizeof(uint4);
  size_t nU = BUF / sizeof(unsigned);

  // rep 调大保证单次 >= 50ms (78 block x 256 thr 仍能打满链路)
  g_cfg[K_ST] = {grid, blk, 48, 0, nV, BUF};
  g_cfg[K_LD] = {grid, blk, 48, 0, nV, BUF};
  // atomic 慢很多, 只扫 buffer 的一小段, rep 调大保证时长
  size_t nAt = nU / 32;  // 4MB 的 u32 slot = 1M 个
  g_cfg[K_AT] = {grid, blk, 400, 0, nAt, nAt * 4};

  cudaStream_t s0, s1;
  CK(cudaStreamCreate(&s0));
  CK(cudaStreamCreate(&s1));
  cudaEvent_t e0a, e0b, e1a, e1b;
  CK(cudaEventCreate(&e0a)); CK(cudaEventCreate(&e0b));
  CK(cudaEventCreate(&e1a)); CK(cudaEventCreate(&e1b));

  // ---------------------------------------------------------------- solo
  auto solo = [&](int kind, int thr) -> double {
    g_cfg[kind].thr = thr;
    launch(kind, rbuf[0], sink, s0);  // warmup
    CK(cudaStreamSynchronize(s0));
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
      CK(cudaEventRecord(e0a, s0));
      launch(kind, rbuf[0], sink, s0);
      CK(cudaEventRecord(e0b, s0));
      CK(cudaEventSynchronize(e0b));
      float ms; CK(cudaEventElapsedTime(&ms, e0a, e0b));
      best = std::min(best, (double)ms);
    }
    return gbps(total_bytes(kind), best);
  };

  // ---------------------------------------------------------------- pair
  // 返回 (bwA, bwB) —— 用各自 event 算; 同时报告墙钟聚合
  struct PairRes { double a, b, agg, msA, msB, msWall; };
  auto pair_run = [&](int ka, int kb, int thrA, int thrB) -> PairRes {
    g_cfg[ka].thr = thrA;
    g_cfg[kb].thr = thrB;
    // warmup
    launch(ka, rbuf[0], sink, s0);
    launch(kb, rbuf[1], sink, s1);
    CK(cudaDeviceSynchronize());

    PairRes best{0, 0, 0, 1e30, 1e30, 1e30};
    for (int t = 0; t < 3; ++t) {
      Timer wall;
      CK(cudaDeviceSynchronize());
      wall.start(0);
      CK(cudaEventRecord(e0a, s0));
      CK(cudaEventRecord(e1a, s1));
      launch(ka, rbuf[0], sink, s0);
      launch(kb, rbuf[1], sink, s1);
      CK(cudaEventRecord(e0b, s0));
      CK(cudaEventRecord(e1b, s1));
      double msW = wall.stop_ms(0);
      CK(cudaDeviceSynchronize());
      float ma, mb;
      CK(cudaEventElapsedTime(&ma, e0a, e0b));
      CK(cudaEventElapsedTime(&mb, e1a, e1b));
      double aggms = std::max((double)ma, (double)mb);
      double agg = gbps(total_bytes(ka) + total_bytes(kb), aggms);
      if (agg > best.agg) {
        best.a = gbps(total_bytes(ka), ma);
        best.b = gbps(total_bytes(kb), mb);
        best.agg = agg; best.msA = ma; best.msB = mb; best.msWall = msW;
      }
    }
    return best;
  };

  // ================================================================ Phase A
  hdr("A) 单类流量基线 T_solo (全速)");
  double solo_full[3];
  printf("| 流量类型 | 逻辑字节/次(MB) | 时长(ms) | T_solo (GB/s) |\n");
  printf("|---|---|---|---|\n");
  for (int k = 0; k < 3; ++k) {
    g_cfg[k].thr = 0;
    // 先量一次时长
    launch(k, rbuf[0], sink, s0);
    CK(cudaStreamSynchronize(s0));
    CK(cudaEventRecord(e0a, s0));
    launch(k, rbuf[0], sink, s0);
    CK(cudaEventRecord(e0b, s0));
    CK(cudaEventSynchronize(e0b));
    float ms; CK(cudaEventElapsedTime(&ms, e0a, e0b));
    solo_full[k] = solo(k, 0);
    printf("| %s | %.1f | %.1f | %.1f |\n", kname[k],
           total_bytes(k) / 1e6, ms, solo_full[k]);
  }

  // 用硬件计数器验证 atomic 的线上字节
  hdr("A2) NVLink 硬件计数器校验 (GPU0 视角)");
  for (int k = 0; k < 3; ++k) {
    g_cfg[k].thr = 0;
    CK(cudaDeviceSynchronize());
    NvlCounters c0 = nvl_read_counters(0);
    for (int r = 0; r < 4; ++r) launch(k, rbuf[0], sink, s0);
    CK(cudaStreamSynchronize(s0));
    NvlCounters c1 = nvl_read_counters(0);
    NvlCounters d = nvl_diff(c0, c1);
    nvl_report(kname[k], total_bytes(k) * 4, d);
  }

  // ================================================================ Phase B
  hdr("B) 全速并发 3x3 干扰矩阵");
  printf("每格: (T_pair_A/T_solo_A , T_pair_B/T_solo_B) | 聚合 GB/s\n\n");
  printf("| A \\ B | STORE | LOAD | ATOMIC |\n|---|---|---|---|\n");
  double ratioA[3][3], ratioB[3][3];
  double pairA[3][3], pairB[3][3];
  for (int a = 0; a < 3; ++a) {
    printf("| **%s** ", kname[a]);
    for (int b = 0; b < 3; ++b) {
      PairRes r = pair_run(a, b, 0, 0);
      ratioA[a][b] = r.a / solo_full[a];
      ratioB[a][b] = r.b / solo_full[b];
      pairA[a][b] = r.a; pairB[a][b] = r.b;
      printf("| (%.2f, %.2f) %.0f ", ratioA[a][b], ratioB[a][b], r.agg);
    }
    printf("|\n");
  }

  hdr("B2) 全速并发绝对带宽 (GB/s)");
  printf("| A \\ B | STORE | LOAD | ATOMIC |\n|---|---|---|---|\n");
  for (int a = 0; a < 3; ++a) {
    printf("| **%s** ", kname[a]);
    for (int b = 0; b < 3; ++b)
      printf("| A=%.0f B=%.0f ", pairA[a][b], pairB[a][b]);
    printf("|\n");
  }
  printf("\nsolo: STORE=%.1f LOAD=%.1f ATOMIC=%.1f GB/s\n", solo_full[0],
         solo_full[1], solo_full[2]);

  // ================================================================ Phase C
  // 标定节流参数: 找到让每类流量掉到 ~30% solo 的 thr
  hdr("C) 节流标定 —— 把单类流量压到 ~30% 带宽");
  printf("两个节流旋钮:\n");
  printf("  (1) 活跃 warp 数: 减少每 block 的并发访存请求\n");
  printf("  (2) grid 大小: 减少参与的 SM 数\n");
  printf("对 LOAD 有效的是 (1)(读要等返回, 受 outstanding 请求数限制);\n");
  printf("STORE 是 fire-and-forget, 1 warp/SM 就能灌满链路, 只能用 (2)。\n\n");

  printf("注入率曲线 A — 活跃 warp 数 (grid=%d 固定):\n", grid);
  printf("| 类型 | 1w | 2w | 3w | 4w | 6w | 8w(全速) |\n|---|---|---|---|---|---|---|\n");
  const int wsweep[] = {1, 2, 3, 4, 6, 8};
  for (int k = 0; k < 3; ++k) {
    printf("| %s ", kname[k]);
    for (int wi = 0; wi < 6; ++wi) printf("| %.0f ", solo(k, wsweep[wi]));
    printf("|\n");
  }

  // grid 节流: 用更少的 SM
  printf("\n注入率曲线 B — grid 大小 (8 warp/block 全速):\n");
  printf("| 类型 | %dblk | %dblk | %dblk | %dblk | %dblk(全速) |\n|---|---|---|---|---|---|\n",
         grid / 16, grid / 8, grid / 4, grid / 2, grid);
  const int gdiv[] = {16, 8, 4, 2, 1};
  int fullgrid = grid;
  for (int k = 0; k < 3; ++k) {
    printf("| %s ", kname[k]);
    for (int gi = 0; gi < 5; ++gi) {
      g_cfg[k].grid = fullgrid / gdiv[gi];
      printf("| %.0f ", solo(k, 0));
    }
    g_cfg[k].grid = fullgrid;
    printf("|\n");
  }

  // 标定: 对每类流量, 在两个旋钮的联合空间里找最接近 30% 的配置
  int thr30[3], grid30[3];
  double solo30[3];
  for (int k = 0; k < 3; ++k) {
    double best_err = 1e30; int bw_ = 0, bg_ = fullgrid; double bbw = 0;
    for (int gi = 0; gi < 5; ++gi) {
      g_cfg[k].grid = fullgrid / gdiv[gi];
      for (int wi = 0; wi < 6; ++wi) {
        double bw = solo(k, wsweep[wi]);
        double err = fabs(bw / solo_full[k] - 0.30);
        if (err < best_err) {
          best_err = err; bw_ = wsweep[wi]; bg_ = g_cfg[k].grid; bbw = bw;
        }
      }
    }
    g_cfg[k].grid = fullgrid;
    thr30[k] = bw_; grid30[k] = bg_; solo30[k] = bbw;
  }
  printf("\n标定结果 (目标 30%% solo):\n");
  printf("| 类型 | grid | 活跃warp | T_solo(节流) GB/s | 占全速比 |\n|---|---|---|---|---|\n");
  for (int k = 0; k < 3; ++k)
    printf("| %s | %d | %d | %.1f | %.2f |\n", kname[k], grid30[k], thr30[k],
           solo30[k], solo30[k] / solo_full[k]);

  hdr("C2) 【核心】低负载(~30%)并发干扰矩阵");
  printf("两条流各自只用 ~30%% 链路带宽, 合计 ~60%%, 链路远未打满。\n");
  printf("此时若仍互相拖累 => 不是带宽不够, 而是队列/VC 结构性共享。\n\n");
  printf("| A \\ B | STORE | LOAD | ATOMIC |\n|---|---|---|---|\n");
  double lowA[3][3], lowB[3][3], lowAgg[3][3];
  for (int a = 0; a < 3; ++a) {
    printf("| **%s** ", kname[a]);
    for (int b = 0; b < 3; ++b) {
      g_cfg[a].grid = grid30[a];
      g_cfg[b].grid = grid30[b];
      PairRes r = pair_run(a, b, thr30[a], thr30[b]);
      g_cfg[a].grid = fullgrid; g_cfg[b].grid = fullgrid;
      lowA[a][b] = r.a / solo30[a];
      lowB[a][b] = r.b / solo30[b];
      lowAgg[a][b] = r.agg;
      printf("| (%.2f, %.2f) %.0f ", lowA[a][b], lowB[a][b], r.agg);
    }
    printf("|\n");
  }
  printf("\n低负载 solo 基线: STORE=%.1f LOAD=%.1f ATOMIC=%.1f GB/s"
         " (合计若无干扰应达 %.0f/%.0f/%.0f)\n",
         solo30[0], solo30[1], solo30[2], solo30[0] * 2, solo30[1] * 2,
         solo30[2] * 2);

  // ================================================================ 判读
  hdr("D) 判读");
  printf("| 组合 | 全速 (rA,rB) | 低负载 (rA,rB) | 判定 |\n|---|---|---|---|\n");
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      double fs = ratioA[a][b] + ratioB[a][b];
      double ls = lowA[a][b] + lowB[a][b];
      const char* verdict;
      if (ls > 1.85) verdict = "独立通道 (低负载无干扰)";
      else if (ls < 1.5 && fs < 1.5) verdict = "队列/VC 共享 (低负载仍拖累)";
      else if (ls < 1.5) verdict = "低负载仍拖累 -> 结构性冲突";
      else if (fs > 1.7) verdict = "公平分带宽 (纯带宽竞争)";
      else verdict = "带宽竞争 + 部分不公平";
      printf("| %s+%s | (%.2f,%.2f) | (%.2f,%.2f) | %s |\n", kname[a], kname[b],
             ratioA[a][b], ratioB[a][b], lowA[a][b], lowB[a][b], verdict);
    }

  printf("\n[done]\n");
  return 0;
}
