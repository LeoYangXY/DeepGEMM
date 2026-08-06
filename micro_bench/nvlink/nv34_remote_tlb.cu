// ============================================================================
// nv34_remote_tlb —— 远端访问的地址翻译代价
//
// 一次 GPU0 读 GPU1 的访问, 地址要被翻译两次概念上的映射:
//   (1) GPU0 侧: VA -> "这块在 peer 上" 的判定 + peer aperture 定位
//   (2) 到达 GPU1 后: 定位到具体 HBM 物理页
// 官方资料对 peer 地址翻译资源(有多少 TLB entry、peer aperture 有几个槽)
// 一个字都没有。本实验从外部量它。
//
//   A) TLB reach: 指针追逐, 每一跳跨一个页 (2MB), 页数 N 从 1 扫到 8192。
//      延迟随 N 的台阶就是各级 TLB 的容量。远端 vs 本地对比,
//      看远端是不是多交一份翻译税。
//   B) peer 分配数量: 在 GPU1 上分配 K 块独立 cudaMalloc buffer,
//      GPU0 轮流访问它们。K 从 1 扫到 1024。
//      如果某个 K 之后延迟跳变 -> peer 映射槽位用完了。
//   C) 页大小影响: 用 VMM (cuMemCreate) 显式控制 2MB 粒度 vs cudaMalloc,
//      以及连续大页 vs 打散的多个小映射, 对远端带宽的影响。
// ============================================================================
#include "nvl_common.cuh"
#include <cuda.h>

// 指针追逐: buf 里存的是"下一跳的元素下标"
// MODE=0 普通 ld (可进 L1/L2)  MODE=1 ld.cv (强制绕过缓存, 用来隔离出 TLB 效应)
template <int MODE>
__global__ void k_chase(const unsigned long long* buf, int hops, unsigned long long* out) {
  unsigned long long i = 0;
  for (int k = 0; k < hops; ++k) {
    if (MODE == 0) i = buf[i];
    else asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(i) : "l"(buf + i) : "memory");
  }
  long long t0 = clk();
  for (int k = 0; k < hops; ++k) {
    if (MODE == 0) i = buf[i];
    else asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(i) : "l"(buf + i) : "memory");
  }
  long long t1 = clk();
  out[0] = (t1 - t0) / hops;
  out[1] = i;
}

// 建环: npage 个页, 每页取第一个元素, 环状链接。用伪随机顺序打乱, 防止预取。
__global__ void k_ring(unsigned long long* buf, long long pgElems, int npage,
                       const int* order) {
  if (threadIdx.x || blockIdx.x) return;
  for (int k = 0; k < npage; ++k) {
    long long cur = (long long)order[k] * pgElems;
    long long nxt = (long long)order[(k + 1) % npage] * pgElems;
    buf[cur] = (unsigned long long)nxt;
  }
}

// K 块独立 buffer 轮流访问 —— 必须真串行(下一次地址依赖上一次读到的值),
// 否则编译器会把 K 个独立 load 并行发出去, 测到的是吞吐不是延迟。
// 做法: 每块 buffer 的第 0 个 u64 里存"下一块的编号", 形成跨 buffer 的指针环。
__global__ void k_multibuf_init(unsigned long long** bufs, int K) {
  if (threadIdx.x || blockIdx.x) return;
  for (int k = 0; k < K; ++k) bufs[k][0] = (unsigned long long)((k + 1) % K);
}
__global__ void k_multibuf(unsigned long long** bufs, int K, int iters,
                           unsigned long long* out) {
  unsigned long long idx = 0;
  for (int it = 0; it < iters; ++it)
    for (int k = 0; k < K; ++k) {
      unsigned long long v;
      asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(bufs[idx]) : "memory");
      idx = v;
    }
  long long t0 = clk();
  for (int it = 0; it < iters; ++it)
    for (int k = 0; k < K; ++k) {
      unsigned long long v;
      asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(bufs[idx]) : "memory");
      idx = v;
    }
  long long t1 = clk();
  out[0] = (t1 - t0) / ((long long)iters * K);
  out[1] = idx;
}

__global__ void k_wr(uint4* d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (; i < n; i += s) st128(d + i, v);
}
// 在 K 块不连续的 buffer 之间轮转写。
// 关键: 每个 block 整块负责一段, block 内地址连续 —— 保持完美合并,
// 这样测出来的差异只可能来自"映射被打散"而不是访存模式变差。
__global__ void k_wr_scatter(uint4** bufs, int K, size_t nPerBuf) {
  uint4 v = make_uint4(1, 2, 3, 4);
  for (int b = blockIdx.x % K; b < K; b += gridDim.x > (unsigned)K ? K : gridDim.x) {
    uint4* p = bufs[b];
    int nb = gridDim.x / K; if (nb < 1) nb = 1;
    int sub = (blockIdx.x / K) % nb;
    for (size_t i = (size_t)sub * blockDim.x + threadIdx.x; i < nPerBuf;
         i += (size_t)nb * blockDim.x)
      st128(p + i, v);
    if (gridDim.x > (unsigned)K) break;
  }
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv34_remote_tlb — 远端访问的地址翻译代价\n# %s x%d, SM clk %.3f GHz\n",
         env.name, env.ndev, env.clkGHz);

  const long long PG = 2 * 1024 * 1024;               // 2MB 页
  const long long pgElems = PG / sizeof(unsigned long long);
  const int MAXPG = 8192;                              // 16 GB
  const size_t BUF = (size_t)MAXPG * PG;

  unsigned long long* loc = (unsigned long long*)nvl_alloc(0, BUF);
  unsigned long long* rem = (unsigned long long*)nvl_alloc(1, BUF);
  CK(cudaSetDevice(0));
  unsigned long long* out;
  CK(cudaMalloc(&out, 64));
  int* order;
  CK(cudaMallocManaged(&order, MAXPG * sizeof(int)));

  // ------------------------------------------------------------------ A
  hdr("A) TLB reach: 每跳跨 2MB 页, 页数 N 扫描");
  printf("cached = 普通 ld (可命中 L1/L2)；bypass = ld.global.cv (绕过缓存, 隔离 TLB)\n");
  printf("%-8s %-10s %-11s %-11s %-11s %-11s\n", "页数N", "覆盖",
         "本地cached", "远端cached", "本地bypass", "远端bypass");
  for (int npage = 1; npage <= MAXPG; npage *= 2) {
    // 伪随机置换页序, 破坏顺序预取
    for (int i = 0; i < npage; ++i) order[i] = i;
    unsigned s = 12345u;
    for (int i = npage - 1; i > 0; --i) {
      s = s * 1103515245u + 12345u;
      int j = (int)((s >> 16) % (unsigned)(i + 1));
      int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    CK(cudaSetDevice(0));
    k_ring<<<1, 1>>>(loc, pgElems, npage, order);
    k_ring<<<1, 1>>>(rem, pgElems, npage, order);
    CK(cudaDeviceSynchronize());

    int hops = npage * 4;
    if (hops < 512) hops = 512;
    if (hops > 8192) hops = 8192;
    unsigned long long h[2];
    double v[4];
    k_chase<0><<<1, 1>>>(loc, hops, out); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost)); v[0] = (double)h[0];
    k_chase<0><<<1, 1>>>(rem, hops, out); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost)); v[1] = (double)h[0];
    k_chase<1><<<1, 1>>>(loc, hops, out); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost)); v[2] = (double)h[0];
    k_chase<1><<<1, 1>>>(rem, hops, out); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost)); v[3] = (double)h[0];
    char cov[32];
    double mb = npage * 2.0;
    if (mb < 1024) snprintf(cov, sizeof cov, "%.0f MB", mb);
    else snprintf(cov, sizeof cov, "%.1f GB", mb / 1024);
    printf("%-8d %-10s %-11.0f %-11.0f %-11.0f %-11.0f\n", npage, cov, v[0], v[1],
           v[2], v[3]);
    fflush(stdout);
  }
  printf("(cycles @ %.2f GHz; 1 cyc = %.3f ns)\n", env.clkGHz, 1.0 / env.clkGHz);

  // ------------------------------------------------------------------ B
  hdr("B) peer 独立分配数量 K: 映射槽位会不会用完");
  {
    const int KMAX = 1024;
    unsigned long long** hb = (unsigned long long**)malloc(KMAX * sizeof(void*));
    for (int k = 0; k < KMAX; ++k) {
      CK(cudaSetDevice(1));
      CK(cudaMalloc(&hb[k], 2 * 1024 * 1024));  // 每块 2MB, 独立分配
      CK(cudaMemset(hb[k], 0, 2 * 1024 * 1024));
    }
    CK(cudaSetDevice(0));
    unsigned long long** db;
    CK(cudaMalloc(&db, KMAX * sizeof(void*)));
    CK(cudaMemcpy(db, hb, KMAX * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    printf("串行指针环跨 K 块独立 2MB peer 分配, 每次访问都是一次完整远端往返\n");
    printf("%-8s %-16s %-12s\n", "K", "覆盖(MB)", "延迟 cyc");
    for (int K = 1; K <= KMAX; K *= 2) {
      unsigned long long h[2];
      k_multibuf_init<<<1, 1>>>(db, K);
      CK(cudaDeviceSynchronize());
      int iters = 2048 / K; if (iters < 2) iters = 2;
      k_multibuf<<<1, 1>>>(db, K, iters, out);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost));
      printf("%-8d %-16d %-12llu\n", K, K * 2, h[0]);
      fflush(stdout);
    }
    // ---- C: 打散映射对带宽的影响 ----
    hdr("C) 远端写带宽: 1 块连续 2GB vs K 块打散的 2MB");
    CK(cudaSetDevice(1));
    uint4* cont;
    CK(cudaMalloc(&cont, 1024ull << 20));
    CK(cudaSetDevice(0));
    int grid = env.sm * 4, blk = 256;
    size_t nTot = (1024ull << 20) / 16;
    double t = bench_ms([&] { k_wr<<<grid, blk>>>(cont, nTot); }, 5);
    printf("%-40s %8.1f GB/s\n", "1 块连续 1GB", gbps(1024.0 * 1048576, t));
    uint4** db2 = (uint4**)db;
    for (int K = 8; K <= 512; K *= 4) {
      size_t nPer = (2ull << 20) / 16;
      double tt = bench_ms([&] { k_wr_scatter<<<grid, blk>>>(db2, K, nPer); }, 20);
      char tag[64];
      snprintf(tag, sizeof tag, "%d 块独立 2MB (共 %d MB)", K, K * 2);
      printf("%-40s %8.1f GB/s\n", tag, gbps((double)K * 2 * 1048576, tt));
    }
  }

  printf("\n[done]\n");
  return 0;
}
