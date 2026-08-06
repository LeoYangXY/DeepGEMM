// ============================================================================
// nv05_partial_write —— 部分写 / 写放大 / 是否有 RMW
//
// 假设: 远端写如果只覆盖一个传输粒度(sector)的一部分, 有三种可能:
//   (1) write-through 字节使能: 只发被写的字节 + byte-enable 掩码
//       -> rawTx 随 chunk 增大近似线性, rawRx 只有小而恒定的 ack 流量
//   (2) read-modify-write: 远端(或本地)先把整个 sector 读回来再合并
//       -> rawRx 会随写量成比例上升, 且量级接近 rawTx
//   (3) 整 sector 写: 不管写几个字节都搬满 G 字节
//       -> rawTx 恒定不随 chunk 变化(直到 chunk 达到 G)
//
// 构造: 每 128B 块只写头部 C 字节 (C = 1,2,4,8,16,32,64,128), 其余不碰。
//       **块数固定** = NBLK, 所以「有用字节」= NBLK*C 随 C 变,
//       但「触碰的 128B 块数」不变 —— 这是关键控制变量:
//       如果是 (3), rawTx 应当 **完全不随 C 变化**。
//       如果是 (1), rawTx 应随 C 近似线性。
//       rawRx 的走势区分 (1) 和 (2)。
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"

// 每线程负责一个 128B 块, 在块头写 C 字节 (用最少条指令拼出来)
template <int C>
__global__ void k_partial(char* __restrict__ base, size_t nblkMask, int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(0xC1C1C1C1u, 0xC2C2C2C2u, 0xC3C3C3C3u, 0xC4C4C4C4u);
  for (int k = 0; k < iter; ++k) {
    size_t blk = (g + (size_t)k * s) & nblkMask;
    char* p = base + blk * 128;
    if (C == 1)        st8(p, 0xC1);
    else if (C == 2)   st16(p, 0xC1C1);
    else if (C == 4)   st32(p, 0xC1C1C1C1u);
    else if (C == 8)   st64(p, make_uint2(0xC1C1C1C1u, 0xC2C2C2C2u));
    else if (C == 16)  st128(p, v);
    else if (C == 32) { st128(p, v); st128(p + 16, v); }
    else if (C == 64) {
      st128(p, v); st128(p + 16, v); st128(p + 32, v); st128(p + 48, v);
    } else {
      #pragma unroll
      for (int q = 0; q < 8; ++q) st128(p + q * 16, v);
    }
  }
}

static int G, B;
static size_t NBLKMASK;
static const int ITER = 256;

template <int C>
static void run(char* rem, double& out_bw) {
  size_t threads = (size_t)G * B;
  double useful = (double)threads * ITER * C;      // 有用字节
  double touched = (double)threads * ITER * 128;   // 触碰的 128B 块总字节

  double t = bench_ms([&] { k_partial<C><<<G, B>>>(rem, NBLKMASK, ITER); }, 3);
  out_bw = gbps(useful, t);

  int rep = (int)(2000.0 / t) + 1;
  if (rep > 40000) rep = 40000;
  CK(cudaDeviceSynchronize());
  NvlCounters a = nvl_read_counters(0);
  for (int r = 0; r < rep; ++r) k_partial<C><<<G, B>>>(rem, NBLKMASK, ITER);
  CK(cudaDeviceSynchronize());
  NvlCounters b = nvl_read_counters(0);
  NvlCounters d = nvl_diff(a, b);

  double U = useful * rep, T = touched * rep;
  printf("| %d | %.1f | %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %.2f | %.3f | %.4f |\n",
         C, out_bw, U / 1e6, T / 1e6, d.dataTx / 1e6, d.dataRx / 1e6,
         d.rawTx / 1e6, d.rawRx / 1e6,
         U ? d.rawTx / U : 0.0,                       // rawTx / 有用字节
         T ? d.rawTx / T : 0.0,                       // rawTx / 触碰字节
         d.rawTx ? (double)d.rawRx / d.rawTx : 0.0);  // Rx/Tx 比
  fflush(stdout);
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv05_partial_write — 部分写 / 写放大 / RMW 判定 (%s, %d SM)\n",
         env.name, env.sm);

  const size_t BUF = 512ull << 20;
  const size_t NBLK = BUF / 128;
  NBLKMASK = NBLK - 1;
  CK(cudaSetDevice(0));
  unsigned* sink = (unsigned*)nvl_alloc(0, 1024);
  (void)sink;
  char* rem = (char*)nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));

  G = env.sm * 4; B = 256;
  size_t threads = (size_t)G * B;
  printf("每线程负责 1 个 128B 块, 块数固定 = %zu * %d = %.1f M 块/kernel\n",
         threads, ITER, threads * (double)ITER / 1e6);
  printf("=> **触碰的 128B 块数完全不随 C 变化**, 这是控制变量。\n");

  hdr("每 128B 块只写头部 C 字节 (硬件计数器 >= 2s/点)");
  printf("| C(写字节) | 有效BW GB/s | 有用MB | 触碰MB | dataTx MB | dataRx MB | rawTx MB | rawRx MB | rawTx/有用 | rawTx/触碰 | rawRx/rawTx |\n");
  printf("|---|---|---|---|---|---|---|---|---|---|---|\n");

  double bw;
  run<1>(rem, bw);
  run<2>(rem, bw);
  run<4>(rem, bw);
  run<8>(rem, bw);
  run<16>(rem, bw);
  run<32>(rem, bw);
  run<64>(rem, bw);
  run<128>(rem, bw);

  printf("\n判读指南:\n");
  printf("  rawTx/触碰 ~ 恒定 1.0  -> 整 128B 写 (无 byte-enable), 严重写放大\n");
  printf("  rawTx/有用 ~ 恒定      -> byte-enable write-through, 无放大\n");
  printf("  rawRx/rawTx 随 C 上升  -> 有 read-modify-write\n");
  printf("  rawRx/rawTx 小且恒定   -> 只是 write ack / flow control\n");
  printf("\n[done]\n");
  return 0;
}
