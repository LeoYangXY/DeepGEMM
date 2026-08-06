// ============================================================================
// nv32_linkstripe —— 18 条 link 之间怎么分流量？条带粒度是多少？
//
// 这是"资料完全没写"的一块。GPU0 有 18 条物理 link 接到 4 颗 NVSwitch。
// 一次 GPU0->GPU1 的写，硬件是怎么把请求摊到 18 条 link 上的？
//
// 探测手法：
//   NVML 能读**每条 link 单独的** Data/Raw Tx/Rx 字节计数器。
//   所以只要：snapshot 全部 18 条 -> 跑一个"地址范围受控"的负载 -> 再 snapshot,
//   就能直接看见每条 link 分到了多少字节。
//
//   实验 A: 大范围顺序写 -> 看 18 条是否均匀 (均匀 = 有硬件条带化/负载均衡)
//   实验 B: 把写限制在一个很小的地址窗口 W 内反复写, W 从 256B 扫到 1MB。
//           如果 W 小的时候只有 k 条 link 有流量, W 变大后 k 涨到 18,
//           那么 "让 18 条全用上所需的最小 W" / 18 就是**地址条带粒度**。
//           这是唯一能从外部观测到条带粒度的办法。
//   实验 C: 同一份流量发给不同 peer, 看 link 使用集合是否随目标改变
//           (判断 link 是按目标静态划分, 还是全部聚合共享)
// ============================================================================
#include "nvl_common.cuh"
#include <nvml.h>

#define MAXL 32

struct LinkCnt {
  unsigned long long tx[MAXL], rx[MAXL];
  int n;
};

// 解析 nvidia-smi nvlink -gt d -i dev, 取每条 link 的 Data Tx/Rx (KiB -> Byte)
static LinkCnt read_links(int dev) {
  LinkCnt c{};
  char cmd[128];
  snprintf(cmd, sizeof cmd, "nvidia-smi nvlink -gt d -i %d 2>/dev/null", dev);
  FILE* f = popen(cmd, "r");
  if (!f) return c;
  char line[512];
  while (fgets(line, sizeof line, f)) {
    int l;
    unsigned long long v;
    if (sscanf(line, " Link %d: Data Tx: %llu", &l, &v) == 2) {
      if (l < MAXL) { c.tx[l] = v * 1024ull; if (l + 1 > c.n) c.n = l + 1; }
    } else if (sscanf(line, " Link %d: Data Rx: %llu", &l, &v) == 2) {
      if (l < MAXL) c.rx[l] = v * 1024ull;
    }
  }
  pclose(f);
  return c;
}

// 写满整个 buffer
__global__ void k_wr_full(uint4* dst, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (; i < n; i += s) st128(dst + i, v);
}

// 只在前 window 字节内反复写 iters 轮 —— 用来扫条带粒度
__global__ void k_wr_window(uint4* dst, size_t nwin, int iters) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (int it = 0; it < iters; ++it)
    for (size_t i = tid; i < nwin; i += s) st128(dst + i, v);
}

static void print_dist(const char* tag, const LinkCnt& a, const LinkCnt& b,
                       int nlink, bool rx) {
  double tot = 0, mx = 0, mn = 1e30;
  double d[MAXL];
  int active = 0;
  for (int i = 0; i < nlink; ++i) {
    d[i] = (double)((rx ? b.rx[i] : b.tx[i]) - (rx ? a.rx[i] : a.tx[i]));
    tot += d[i];
    if (d[i] > mx) mx = d[i];
    if (d[i] < mn) mn = d[i];
  }
  for (int i = 0; i < nlink; ++i)
    if (d[i] > 0.02 * mx) ++active;
  double mean = tot / nlink, sd = 0;
  for (int i = 0; i < nlink; ++i) sd += (d[i] - mean) * (d[i] - mean);
  sd = sqrt(sd / nlink);
  printf("%-30s total=%8.1f MB  active_links=%2d/%d  max=%7.2f MB min=%7.2f MB "
         "CV=%.3f\n",
         tag, tot / 1e6, active, nlink, mx / 1e6, mn / 1e6,
         mean > 0 ? sd / mean : 0.0);
  printf("   per-link MB: ");
  for (int i = 0; i < nlink; ++i) printf("%.1f ", d[i] / 1e6);
  printf("\n");
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv32_linkstripe — 18 条 NVLink 之间的流量分布与地址条带粒度\n");
  printf("# %s x%d, %d SM\n", env.name, env.ndev, env.sm);

  const size_t BIG = 512ull << 20;
  uint4* r1 = (uint4*)nvl_alloc(1, BIG);
  uint4* r2 = env.ndev > 2 ? (uint4*)nvl_alloc(2, BIG) : nullptr;
  uint4* r3 = env.ndev > 3 ? (uint4*)nvl_alloc(3, BIG) : nullptr;
  CK(cudaSetDevice(0));
  int grid = env.sm * 4, blk = 256;

  // ---------------------------------------------------------------- A
  hdr("A) 大范围顺序写 GPU0->GPU1: 18 条 link 均匀吗");
  {
    // 跑够久, 让计数器变化显著压过背景噪声
    LinkCnt a = read_links(0);
    for (int r = 0; r < 8; ++r) k_wr_full<<<grid, blk>>>(r1, BIG / 16);
    CK(cudaDeviceSynchronize());
    LinkCnt b = read_links(0);
    printf("payload = %.1f MB\n", 8.0 * BIG / 1e6);
    print_dist("GPU0 Tx (发出)", a, b, b.n, false);
    print_dist("GPU0 Rx (收到,应为ack)", a, b, b.n, true);
  }

  // ---------------------------------------------------------------- B
  hdr("B) 地址窗口扫描: 让 18 条 link 全用上需要多大的地址范围");
  printf("窗口 W 内反复写, 总 payload 固定 ~2GB。若 W 小时只有少数 link 有流量,\n"
         "则 W_full/18 ≈ 地址条带粒度。\n\n");
  printf("%-12s %-10s %-8s %-8s %s\n", "窗口W", "payloadMB", "活跃link", "CV",
         "per-link MB (前18条)");
  for (size_t W = 256; W <= (1ull << 22); W <<= 1) {
    size_t nwin = W / 16;
    if (nwin == 0) continue;
    const double TARGET = 2e9;
    int iters = (int)(TARGET / (double)W);
    if (iters < 1) iters = 1;
    if (iters > 2000000) iters = 2000000;
    // 线程数不能超过窗口元素数, 否则大量线程空转
    size_t need = nwin;
    int g = (int)((need + blk - 1) / blk);
    if (g < 1) g = 1;
    if (g > grid) g = grid;

    k_wr_window<<<g, blk>>>(r1, nwin, 1);
    CK(cudaDeviceSynchronize());
    LinkCnt a = read_links(0);
    k_wr_window<<<g, blk>>>(r1, nwin, iters);
    CK(cudaDeviceSynchronize());
    LinkCnt b = read_links(0);

    double d[MAXL], tot = 0, mx = 0;
    for (int i = 0; i < b.n; ++i) {
      d[i] = (double)(b.tx[i] - a.tx[i]);
      tot += d[i];
      if (d[i] > mx) mx = d[i];
    }
    int active = 0;
    for (int i = 0; i < b.n; ++i)
      if (d[i] > 0.05 * mx) ++active;
    double mean = tot / b.n, sd = 0;
    for (int i = 0; i < b.n; ++i) sd += (d[i] - mean) * (d[i] - mean);
    sd = sqrt(sd / b.n);
    printf("%-12zu %-10.0f %-8d %-8.3f ", W, tot / 1e6, active,
           mean > 0 ? sd / mean : 0.0);
    for (int i = 0; i < b.n && i < 18; ++i) printf("%.0f ", d[i] / 1e6);
    printf("\n");
    fflush(stdout);
  }

  // ---------------------------------------------------------------- C
  if (r2 && r3) {
    hdr("C) 同样流量发给不同 peer: link 使用集合是否随目标改变");
    uint4* tgt[3] = {r1, r2, r3};
    for (int t = 0; t < 3; ++t) {
      LinkCnt a = read_links(0);
      for (int r = 0; r < 4; ++r) k_wr_full<<<grid, blk>>>(tgt[t], BIG / 16);
      CK(cudaDeviceSynchronize());
      LinkCnt b = read_links(0);
      char tag[64];
      snprintf(tag, sizeof tag, "GPU0 -> GPU%d Tx", t + 1);
      print_dist(tag, a, b, b.n, false);
    }
    hdr("C2) 同时发给 3 个 peer");
    {
      cudaStream_t s[3];
      for (int i = 0; i < 3; ++i) CK(cudaStreamCreate(&s[i]));
      LinkCnt a = read_links(0);
      for (int r = 0; r < 4; ++r)
        for (int t = 0; t < 3; ++t)
          k_wr_full<<<grid / 3, blk, 0, s[t]>>>(tgt[t], BIG / 16);
      CK(cudaDeviceSynchronize());
      LinkCnt b = read_links(0);
      print_dist("GPU0 -> {1,2,3} Tx", a, b, b.n, false);
      for (int i = 0; i < 3; ++i) cudaStreamDestroy(s[i]);
    }
  }

  printf("\n[done]\n");
  return 0;
}
