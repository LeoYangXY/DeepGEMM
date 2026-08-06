// ============================================================================
// nv14_latency —— 远端延迟精确拆解 + PTX cache modifier 全谱
//
// 方法:
//   单线程 <<<1,1>>> pointer-chase。链表元素是 4B 的 uint32 索引(下一跳 offset)，
//   真串行: 下一次地址 = base + chain[cur]。编译器无法预取/重排。
//   步长固定 STRIDE 字节(> 128B L2 sector, 且不是 2 的整幂对齐冲突)，
//   working set 通过 nodes 数控制:
//     L1 级:  32 KB   (< 256KB L1)
//     L2 级:  8  MB   (> L1, << 60MB L2)
//     HBM级:  512 MB  (>> 60MB L2)
//   远端级: 同上三档, buffer 在 GPU1。
//
//   每档再扫 cache modifier: ca / cg / cs / lu / cv / nc(__ldg) / volatile。
//
// 输出: 平均每跳 cycles (clk() 采样 ITER 跳后除以 ITER) 和 ns。
// ============================================================================
#include "nvl_common.cuh"
#include <vector>
#include <algorithm>
#include <random>

// ---------------------------------------------------------------- PTX 变体
// chain 存 uint32 = 下一跳的"元素下标"(以 uint32 为单位)
#define DEF_LD(NAME, INSN)                                                    \
  __device__ __forceinline__ unsigned NAME(const unsigned* p) {               \
    unsigned v;                                                               \
    asm volatile(INSN " %0, [%1];" : "=r"(v) : "l"(p) : "memory");            \
    return v;                                                                 \
  }

DEF_LD(ld_ca, "ld.global.ca.u32")
DEF_LD(ld_cg, "ld.global.cg.u32")
DEF_LD(ld_cs, "ld.global.cs.u32")
DEF_LD(ld_lu, "ld.global.lu.u32")
DEF_LD(ld_cv, "ld.global.cv.u32")
DEF_LD(ld_nc, "ld.global.nc.u32")
DEF_LD(ld_vol, "ld.volatile.global.u32")
DEF_LD(ld_plain, "ld.global.u32")

enum Mod { M_PLAIN=0, M_CA, M_CG, M_CS, M_LU, M_CV, M_NC, M_VOL, M_NMOD };
static const char* kModName[M_NMOD] = {
    "ld.global (default)", "ld.global.ca", "ld.global.cg", "ld.global.cs",
    "ld.global.lu",        "ld.global.cv", "ld.global.nc", "ld.volatile.global"};

template <int MOD>
__device__ __forceinline__ unsigned ld_sel(const unsigned* p) {
  if (MOD == M_PLAIN) return ld_plain(p);
  if (MOD == M_CA)    return ld_ca(p);
  if (MOD == M_CG)    return ld_cg(p);
  if (MOD == M_CS)    return ld_cs(p);
  if (MOD == M_LU)    return ld_lu(p);
  if (MOD == M_CV)    return ld_cv(p);
  if (MOD == M_NC)    return ld_nc(p);
  return ld_vol(p);
}

// pointer-chase kernel。out[0] = 总 cycles, out[1] = 防优化 sink
template <int MOD>
__global__ void k_chase(const unsigned* __restrict__ chain, int iters,
                        long long* out) {
  unsigned cur = 0;
  // warmup / 让 TLB & 指令 cache 预热 (不计时)
  for (int i = 0; i < 512; ++i) cur = ld_sel<MOD>(chain + cur);
  __syncthreads();
  long long t0 = clk();
  #pragma unroll 8
  for (int i = 0; i < iters; ++i) cur = ld_sel<MOD>(chain + cur);
  long long t1 = clk();
  out[0] = t1 - t0;
  out[1] = (long long)cur;
}

// 空循环基线: 测量 loop overhead(不含访存)
__global__ void k_empty(int iters, long long* out) {
  unsigned cur = 0;
  long long t0 = clk();
  #pragma unroll 8
  for (int i = 0; i < iters; ++i) cur = cur * 1u + 0u;  // 依赖链但无访存
  long long t1 = clk();
  out[0] = t1 - t0;
  out[1] = cur;
}

// ---------------------------------------------------------------- host 构链
// 在 bytes 大小的区域内，按 STRIDE 步长构造随机置换环，元素单位 uint32。
static std::vector<unsigned> build_chain(size_t bytes, size_t stride_bytes,
                                         unsigned seed) {
  size_t nnode = bytes / stride_bytes;
  if (nnode < 8) nnode = 8;
  size_t nelem = bytes / 4;
  std::vector<unsigned> h(nelem, 0u);
  // 节点 i 的元素下标
  std::vector<unsigned> idx(nnode);
  for (size_t i = 0; i < nnode; ++i) idx[i] = (unsigned)(i * stride_bytes / 4);
  // 随机置换(除首元素)形成单环
  std::vector<unsigned> perm(idx.begin() + 1, idx.end());
  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);
  unsigned cur = idx[0];
  for (size_t i = 0; i < perm.size(); ++i) { h[cur] = perm[i]; cur = perm[i]; }
  h[cur] = idx[0];  // 闭环
  return h;
}

struct Level { const char* name; size_t bytes; int dev; };

static double g_clk = 1.98;

// 跑一个 (level, mod) 组合，返回每跳 cycles
template <int MOD>
static double run_one(const unsigned* d_chain, int iters, long long* d_out,
                      double loop_ovh) {
  long long h[2];
  double best = 1e30;
  for (int k = 0; k < 5; ++k) {
    k_chase<MOD><<<1, 1>>>(d_chain, iters, d_out);
    CKLAST();
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost));
    double c = (double)h[0] / iters - loop_ovh;
    if (c < best) best = c;
  }
  return best;
}

typedef double (*RunFn)(const unsigned*, int, long long*, double);
static RunFn kRun[M_NMOD] = {
    run_one<M_PLAIN>, run_one<M_CA>, run_one<M_CG>, run_one<M_CS>,
    run_one<M_LU>,    run_one<M_CV>, run_one<M_NC>, run_one<M_VOL>};

int main() {
  NvlEnv env = nvl_init(2);
  g_clk = 1.98;  // 实测 SM 时钟
  printf("# nv14_latency — %s x%d, %d SM, boost clock %.3f GHz (延迟换算用 "
         "%.2f GHz 实测频率)\n",
         env.name, env.ndev, env.sm, env.clkGHz, g_clk);
  nvl_enable_peers(env.ndev);

  const size_t STRIDE = 8192;  // 8KB: 远大于 128B L2 line / 32B sector,
                               // 也大于 4KB page -> 每跳都换 TLB entry 之外的
                               // 页(用大页时仍在同 2MB 内, 尽量减少 TLB 噪声)
  const int ITERS = 4096;

  CK(cudaSetDevice(0));
  long long* d_out = nullptr;
  CK(cudaMalloc(&d_out, 64));

  // ------- loop overhead 基线
  double loop_ovh = 0;
  {
    long long h[2];
    double best = 1e30;
    for (int k = 0; k < 5; ++k) {
      k_empty<<<1, 1>>>(ITERS, d_out);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost));
      best = fmin(best, (double)h[0] / ITERS);
    }
    loop_ovh = best;
  }
  printf("\n空循环(依赖链, 无访存) 每次迭代 = %.2f cyc  -> 已从下表所有数字中扣除\n",
         loop_ovh);
  printf("pointer-chase stride = %zu B, iters = %d, <<<1,1>>>\n", STRIDE, ITERS);

  Level lv[] = {
      {"local L1  (32 KB WS)",   32ull << 10,   0},
      {"local L2  (8 MB WS)",    8ull << 20,    0},
      {"local L2  (32 MB WS)",   32ull << 20,   0},
      {"local HBM (512 MB WS)",  512ull << 20,  0},
      {"remote L1?(32 KB WS)",   32ull << 10,   1},
      {"remote    (8 MB WS)",    8ull << 20,    1},
      {"remote    (32 MB WS)",   32ull << 20,   1},
      {"remote HBM(512 MB WS)",  512ull << 20,  1},
  };
  const int NLV = sizeof(lv) / sizeof(lv[0]);

  double res[NLV][M_NMOD];

  for (int L = 0; L < NLV; ++L) {
    size_t bytes = lv[L].bytes;
    // L1 档用小 stride 保证节点数够
    size_t stride = (bytes <= (64ull << 10)) ? 256 : STRIDE;
    std::vector<unsigned> h = build_chain(bytes, stride, 12345 + L);
    unsigned* d = (unsigned*)nvl_alloc(lv[L].dev, bytes);
    CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(0));  // 永远从 GPU0 发起
    for (int m = 0; m < M_NMOD; ++m) res[L][m] = kRun[m](d, ITERS, d_out, loop_ovh);
    CK(cudaFree(d));
    CK(cudaSetDevice(0));
    fprintf(stderr, "[done] %s\n", lv[L].name);
  }

  // ---------------------------------------------------------------- 输出
  hdr("A) 每跳延迟 (cycles), 已扣空循环开销");
  printf("| %-22s |", "level \\ modifier");
  for (int m = 0; m < M_NMOD; ++m) printf(" %-19s |", kModName[m]);
  printf("\n|");
  for (int m = 0; m < M_NMOD + 1; ++m) printf("---|");
  printf("\n");
  for (int L = 0; L < NLV; ++L) {
    printf("| %-22s |", lv[L].name);
    for (int m = 0; m < M_NMOD; ++m) printf(" %19.1f |", res[L][m]);
    printf("\n");
  }

  hdr("B) 每跳延迟 (ns @ 1.98 GHz)");
  printf("| %-22s |", "level \\ modifier");
  for (int m = 0; m < M_NMOD; ++m) printf(" %-19s |", kModName[m]);
  printf("\n|");
  for (int m = 0; m < M_NMOD + 1; ++m) printf("---|");
  printf("\n");
  for (int L = 0; L < NLV; ++L) {
    printf("| %-22s |", lv[L].name);
    for (int m = 0; m < M_NMOD; ++m)
      printf(" %19.1f |", cyc2ns(res[L][m], g_clk));
    printf("\n");
  }

  // ---------------------------------------------------------------- 拆解
  hdr("C) 延迟拆解 (以 ld.global 默认 modifier 为准)");
  double l1  = res[0][M_PLAIN];
  double l2  = res[2][M_PLAIN];
  double hbm = res[3][M_PLAIN];
  double rem = res[7][M_PLAIN];
  double rem_small = res[4][M_PLAIN];
  printf("%-46s %10.1f cyc  %8.1f ns\n", "本地 L1 命中",       l1,  cyc2ns(l1,g_clk));
  printf("%-46s %10.1f cyc  %8.1f ns\n", "本地 L2 命中",       l2,  cyc2ns(l2,g_clk));
  printf("%-46s %10.1f cyc  %8.1f ns\n", "本地 HBM (L2 miss)", hbm, cyc2ns(hbm,g_clk));
  printf("%-46s %10.1f cyc  %8.1f ns\n", "远端 512MB (NVLink)", rem, cyc2ns(rem,g_clk));
  printf("%-46s %10.1f cyc  %8.1f ns\n", "远端 32KB (小 WS)",   rem_small, cyc2ns(rem_small,g_clk));
  printf("\n分段:\n");
  printf("  L1->L2 增量          = %8.1f cyc (%.1f ns)  = 本地 L2 查询+返回\n", l2-l1, cyc2ns(l2-l1,g_clk));
  printf("  L2->HBM 增量         = %8.1f cyc (%.1f ns)  = 本地 HBM 访问\n",   hbm-l2, cyc2ns(hbm-l2,g_clk));
  printf("  HBM->remote 增量     = %8.1f cyc (%.1f ns)  = NVLink+NVSwitch 往返净开销\n", rem-hbm, cyc2ns(rem-hbm,g_clk));
  printf("  remote - L1          = %8.1f cyc (%.1f ns)  = 整条远端路径(不含 L1 命中部分)\n", rem-l1, cyc2ns(rem-l1,g_clk));

  hdr("D) modifier 相对默认的增量 (cycles, 负=更快)");
  printf("| %-22s |", "level");
  for (int m = 1; m < M_NMOD; ++m) printf(" %-19s |", kModName[m]);
  printf("\n|");
  for (int m = 0; m < M_NMOD; ++m) printf("---|");
  printf("\n");
  for (int L = 0; L < NLV; ++L) {
    printf("| %-22s |", lv[L].name);
    for (int m = 1; m < M_NMOD; ++m)
      printf(" %19.1f |", res[L][m] - res[L][M_PLAIN]);
    printf("\n");
  }

  CK(cudaFree(d_out));
  printf("\n[done]\n");
  return 0;
}
