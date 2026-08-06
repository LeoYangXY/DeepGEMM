// ============================================================================
// nv06_wire_overhead —— 协议开销与 packet 结构（本组最重要）
//
// 唯一能直接看到「线上真实字节」的实验。
//   Data bytes = 协议净荷      Raw bytes = 线上总字节(净荷+头+CRC+flit+idle)
//   raw/data = 协议放大比 -> 反推 packet 头开销
//
// 6 种负载, 每种 >= 2s:
//   (a) 大块顺序远端写 (128B/warp 全合并, st.global.v4)
//   (b) 大块顺序远端读
//   (c) 4B 离散远端写 (stride 4096, 保证每个 4B 单独成包)
//   (d) 4B 离散远端读
//   (e) 远端 atomicAdd (32B stride, 各线程独立地址)
//   (f) Copy Engine memcpyPeer
//
// 反推逻辑:
//   写: payload P 字节 -> 若 packet 净荷 = D 字节 + 头 H 字节,
//       rawTx = P*(D+H)/D  => H = D*(rawTx/P - 1)
//   读: 请求包只有头(几乎无净荷)走 Tx, 响应包带数据走 Rx
//       => rawTx 小 (请求), rawRx ~ P*(D+H)/D (响应)  —— 强不对称
//   对比 (a) 与 (c): 同样的 payload, packet 净荷从 256B 掉到 4B,
//       头开销占比爆炸 -> 直接分离出 H。
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"

// ------------------------------------------------------ (a)(b) 大块顺序
__global__ void k_seq_wr(uint4* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (; i < n; i += s) st128(d + i, v);
}
__global__ void k_seq_rd(const uint4* __restrict__ s_, uint4* sink, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 a = make_uint4(0, 0, 0, 0);
  for (; i < n; i += s) {
    uint4 v = ld128(s_ + i);
    a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
  }
  if (a.x == 0xdeadbeefu && a.y == 0xdeadbeefu) *sink = a;
}

// ------------------------------------------------------ (c)(d) 4B 离散
__global__ void k_scat_wr(char* __restrict__ base, size_t mask, int stride,
                          int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (int k = 0; k < iter; ++k) {
    size_t off = ((g + (size_t)k * s) * (size_t)stride) & mask;
    st32(base + off, 0xBEEFBEEFu);
  }
}
__global__ void k_scat_rd(const char* __restrict__ base, unsigned* sink,
                          size_t mask, int stride, int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  unsigned a = 0;
  for (int k = 0; k < iter; ++k) {
    size_t off = ((g + (size_t)k * s) * (size_t)stride) & mask;
    a ^= ld32(base + off);
  }
  if (a == 0xdeadbeefu) *sink = a;
}

// ------------------------------------------------------ (e) 远端 atomicAdd
__global__ void k_atom(unsigned* __restrict__ base, size_t mask, int stride4,
                       int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (int k = 0; k < iter; ++k) {
    size_t idx = ((g + (size_t)k * s) * (size_t)stride4) & mask;
    atomicAdd(base + idx, 1u);
  }
}

struct Out {
  const char* tag;
  double bw, payload, dTx, dRx, rTx, rRx;
};
static Out results[8];
static int nres = 0;

// 通用: 跑 f() 直到累计 >= targetSec, 记录 counters
template <typename F>
static void measure(const char* tag, F&& f, double payload_per_call,
                    double targetSec = 2.0) {
  double t = bench_ms(f, 3);  // ms per call
  double bw = gbps(payload_per_call, t);
  int rep = (int)(targetSec * 1000.0 / t) + 1;
  if (rep > 200000) rep = 200000;

  CK(cudaDeviceSynchronize());
  NvlCounters a = nvl_read_counters(0);
  for (int r = 0; r < rep; ++r) f();
  CK(cudaDeviceSynchronize());
  NvlCounters b = nvl_read_counters(0);
  NvlCounters d = nvl_diff(a, b);

  double P = payload_per_call * rep;
  results[nres++] = Out{tag, bw, P, (double)d.dataTx, (double)d.dataRx,
                        (double)d.rawTx, (double)d.rawRx};
  printf("  [%s] bw=%.1f GB/s rep=%d payload=%.0f MB  dataTx=%.0f dataRx=%.0f "
         "rawTx=%.0f rawRx=%.0f MB\n",
         tag, bw, rep, P / 1e6, d.dataTx / 1e6, d.dataRx / 1e6, d.rawTx / 1e6,
         d.rawRx / 1e6);
  fflush(stdout);
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv06_wire_overhead — 协议开销与 packet 结构 (%s, %d SM)\n", env.name,
         env.sm);
  printf("# 理论单向: 18 link * 26.562 GB/s = %.1f GB/s\n", 18 * 26.562);

  const size_t BUF = 512ull << 20;
  const size_t MASK = BUF - 1;
  CK(cudaSetDevice(0));
  char* loc = (char*)nvl_alloc(0, BUF);
  unsigned* sink = (unsigned*)nvl_alloc(0, 1024);
  char* rem = (char*)nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));

  int G = env.sm * 4, Bk = 256;
  size_t threads = (size_t)G * Bk;

  const size_t SEQ = 256ull << 20;
  const size_t NSEQ = SEQ / 16;
  const int ITER = 256;
  double scat_pl = (double)threads * ITER * 4;

  hdr("原始测量 (每种负载 >= 2s)");
  measure("a_seq_wr", [&] { k_seq_wr<<<G, Bk>>>((uint4*)rem, NSEQ); }, (double)SEQ);
  measure("b_seq_rd", [&] { k_seq_rd<<<G, Bk>>>((const uint4*)rem, (uint4*)sink, NSEQ); }, (double)SEQ);
  measure("c_4B_wr", [&] { k_scat_wr<<<G, Bk>>>(rem, MASK, 4096, ITER); }, scat_pl);
  measure("d_4B_rd", [&] { k_scat_rd<<<G, Bk>>>(rem, sink, MASK, 4096, ITER); }, scat_pl);
  measure("e_atomicAdd", [&] { k_atom<<<G, Bk>>>((unsigned*)rem, (BUF / 4) - 1, 8, ITER); }, scat_pl);
  measure("f_memcpyPeer", [&] { cudaMemcpyPeerAsync(rem, 1, loc, 0, SEQ, 0); }, (double)SEQ);

  // ------------------------------------------------------------- 主表
  hdr("主表: payload / data / raw / 比值");
  printf("| 负载 | 有效BW GB/s | payload MB | dataTx MB | dataRx MB | rawTx MB | rawRx MB | raw/data | wire/payload | rawTx/payload | rawRx/payload |\n");
  printf("|---|---|---|---|---|---|---|---|---|---|---|\n");
  for (int i = 0; i < nres; ++i) {
    Out& o = results[i];
    double data = o.dTx + o.dRx, raw = o.rTx + o.rRx;
    printf("| %s | %.1f | %.0f | %.0f | %.0f | %.0f | %.0f | %.3f | %.3f | %.3f | %.3f |\n",
           o.tag, o.bw, o.payload / 1e6, o.dTx / 1e6, o.dRx / 1e6, o.rTx / 1e6,
           o.rRx / 1e6, data ? raw / data : 0.0,
           o.payload ? raw / o.payload : 0.0,
           o.payload ? o.rTx / o.payload : 0.0,
           o.payload ? o.rRx / o.payload : 0.0);
  }

  // ------------------------------------------------------------- 反推
  hdr("反推: packet 头开销");
  printf("模型: 每个 packet 携带 D 字节净荷 + H 字节协议开销\n");
  printf("      raw = payload * (D+H)/D  =>  H = D * (raw/payload - 1)\n\n");
  printf("| 负载 | 假设净荷 D | 主方向 raw/payload | 推出 H (B/包) |\n");
  printf("|---|---|---|---|\n");
  struct { const char* tag; double D; int useRx; } models[] = {
      {"a_seq_wr", 256, 0}, {"b_seq_rd", 256, 1},
      {"c_4B_wr", 4, 0},    {"d_4B_rd", 4, 1},
      {"e_atomicAdd", 4, 0}, {"f_memcpyPeer", 256, 0}};
  for (auto& m : models) {
    for (int i = 0; i < nres; ++i) {
      if (strcmp(results[i].tag, m.tag)) continue;
      Out& o = results[i];
      double r = (m.useRx ? o.rRx : o.rTx) / o.payload;
      printf("| %s | %.0fB | %.3f | %.1f |\n", m.tag, m.D, r, m.D * (r - 1.0));
    }
  }

  // ------------------------------------------------------------- 效率拆解
  hdr("效率拆解: 为什么只到理论 478 GB/s 的一部分");
  printf("| 负载 | 有效BW | 线上BW=有效BW*rawTx/payload | 线上/理论478 | 协议损失 | 其余损失 |\n");
  printf("|---|---|---|---|---|---|\n");
  for (int i = 0; i < nres; ++i) {
    Out& o = results[i];
    double amp = (o.rTx + o.rRx) / o.payload;
    double wire = o.bw * amp;
    double th = 18 * 26.562;
    printf("| %s | %.1f | %.1f | %.1f%% | %.1f%% | %.1f%% |\n", o.tag, o.bw,
           wire, 100.0 * wire / th, 100.0 * (amp - 1.0) / amp,
           100.0 * (1.0 - wire / th));
  }
  printf("\n注: 「协议损失」= 线上字节里非净荷的占比; 「其余损失」= 线上带宽\n");
  printf("    仍未打满理论峰值的部分(idle flit / flow-control / switch 调度)。\n");

  printf("\n[done]\n");
  return 0;
}
