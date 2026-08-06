// ============================================================================
// nv03_stride —— stride 扫描找 sector / line 传输粒度
//
// 假设: 远端访问的最小传输单位是 G 字节。若线程按 stride S 访问 4B 数据:
//         S >= G  -> 每个 4B 有用数据要搬 G 字节上线 -> wire/payload = G/4
//         S <  G  -> 相邻线程落在同一个 G 块里, 被合并 -> 放大 = G/(S*32/32)=G/S
//       所以 wire/payload 随 S 增长会在 S=G 处饱和到 G/4。
//       用 NVLink 硬件字节计数器直接量 wire bytes / payload bytes 即可读出 G。
//
// 控制变量: 每个 kernel 的「有用字节」固定(每线程 4B * iter), 只变 stride。
//           带宽 = 有用字节 / 时间。
//           counters 段单独跑, 保证 >= 1.5s 让计数器变化明显。
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"

// 每线程做 iter 次 4B 访问, 第 k 次地址 = ((gtid + k*nthr) * stride) % span
__global__ void k_wr_stride(char* __restrict__ base, size_t spanMask,
                            int stride, int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (int k = 0; k < iter; ++k) {
    size_t idx = g + (size_t)k * s;
    size_t off = (idx * (size_t)stride) & spanMask;
    st32(base + off, 0xA5A5A5A5u);
  }
}
__global__ void k_rd_stride(const char* __restrict__ base, unsigned* sink,
                            size_t spanMask, int stride, int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (int k = 0; k < iter; ++k) {
    size_t idx = g + (size_t)k * s;
    size_t off = (idx * (size_t)stride) & spanMask;
    acc ^= ld32(base + off);
  }
  if (acc == 0xdeadbeefu) *sink = acc;
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv03_stride — stride 扫描找传输粒度 (%s, %d SM)\n", env.name, env.sm);

  const size_t BUF = 512ull << 20;  // 512MB 缓冲, 保证 stride 4096 也能铺开
  const size_t spanMask = BUF - 1;
  CK(cudaSetDevice(0));
  char* loc = (char*)nvl_alloc(0, BUF);
  unsigned* sink = (unsigned*)nvl_alloc(0, 1024);
  char* rem = (char*)nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));

  int grid = env.sm * 4, blk = 256;
  size_t threads = (size_t)grid * blk;
  const int ITER = 512;                       // 每线程 512 次 4B 访问
  double payload = (double)threads * ITER * 4;  // ~163 MB 有用字节

  const int strides[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  const int NS = sizeof(strides) / sizeof(int);

  // ---------------------------------------------------- (A) 带宽扫描
  hdr("A) stride 扫描: 有效带宽 (每线程 4B 访问, 有用字节固定)");
  printf("有用 payload = %.1f MB / kernel\n", payload / 1e6);
  printf("| stride B | 远端写 GB/s | 远端读 GB/s | 本地写 GB/s | 本地读 GB/s |\n");
  printf("|---|---|---|---|---|\n");
  double wr_rem[16], rd_rem[16];
  for (int i = 0; i < NS; ++i) {
    int S = strides[i];
    double t;
    t = bench_ms([&] { k_wr_stride<<<grid, blk>>>(rem, spanMask, S, ITER); }, 3);
    wr_rem[i] = gbps(payload, t);
    t = bench_ms([&] { k_rd_stride<<<grid, blk>>>(rem, sink, spanMask, S, ITER); }, 3);
    rd_rem[i] = gbps(payload, t);
    double wl = gbps(payload, bench_ms([&] { k_wr_stride<<<grid, blk>>>(loc, spanMask, S, ITER); }, 3));
    double rl = gbps(payload, bench_ms([&] { k_rd_stride<<<grid, blk>>>(loc, sink, spanMask, S, ITER); }, 3));
    printf("| %d | %.1f | %.1f | %.1f | %.1f |\n", S, wr_rem[i], rd_rem[i], wl, rl);
  }

  // ---------------------------------------------------- (B) 硬件计数器
  hdr("B) NVLink 硬件计数器: 真实 wire bytes / payload bytes");
  printf("(每个点跑够 ~2s; rawTx/rawRx 为 GPU0 全部 18 link 累计)\n");
  printf("| stride B | 方向 | payload MB | dataTx MB | dataRx MB | rawTx MB | rawRx MB | wire/payload |\n");
  printf("|---|---|---|---|---|---|---|---|\n");

  const int cs[] = {4, 16, 32, 64, 128, 256, 1024};
  const int NC = sizeof(cs) / sizeof(int);
  for (int i = 0; i < NC; ++i) {
    int S = cs[i];
    for (int dir = 0; dir < 2; ++dir) {
      // 先测一次单 kernel 时间, 算需要 rep 次才够 2s
      double t1 = dir == 0
          ? bench_ms([&] { k_wr_stride<<<grid, blk>>>(rem, spanMask, S, ITER); }, 2)
          : bench_ms([&] { k_rd_stride<<<grid, blk>>>(rem, sink, spanMask, S, ITER); }, 2);
      int rep = (int)(2000.0 / t1) + 1;
      if (rep > 20000) rep = 20000;

      CK(cudaDeviceSynchronize());
      NvlCounters a = nvl_read_counters(0);
      for (int r = 0; r < rep; ++r) {
        if (dir == 0) k_wr_stride<<<grid, blk>>>(rem, spanMask, S, ITER);
        else k_rd_stride<<<grid, blk>>>(rem, sink, spanMask, S, ITER);
      }
      CK(cudaDeviceSynchronize());
      NvlCounters b = nvl_read_counters(0);
      NvlCounters d = nvl_diff(a, b);
      double pl = payload * rep;
      printf("| %d | %s | %.0f | %.0f | %.0f | %.0f | %.0f | %.3f |\n", S,
             dir == 0 ? "写" : "读", pl / 1e6, d.dataTx / 1e6, d.dataRx / 1e6,
             d.rawTx / 1e6, d.rawRx / 1e6, (double)(d.rawTx + d.rawRx) / pl);
      fflush(stdout);
    }
  }

  printf("\n[done]\n");
  return 0;
}
