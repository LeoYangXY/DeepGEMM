// ============================================================================
// nv11_littles_law —— 用 Little's Law 反推 NVLink 硬件在途容量 (BDP / credit 池)
//
// Little's Law:  L(在途请求数) = λ(吞吐, req/s) × W(平均延迟, s)
// 等价地       : 在途字节数    = 带宽(B/s) × 延迟(s)
//
// 实验设计: 扫并发度(= grid 数, 每 block 256 thr), 每个点同时测两个量
//   1) 吞吐  λ : 整个 kernel 的聚合远端写/读带宽 (cudaEvent 计时)
//   2) 延迟  W : **同一次 kernel 运行中**, 由 block0 的 thread0 充当"延迟探针",
//                做远端 pointer-chase (依赖链 load), 用 clock64 测单次往返。
//                探针只有 1 个线程, 对总流量贡献 <0.1%, 不干扰负载。
//
//   把两者相乘 => 在途字节数。再除以 128B (NVLink 事务粒度) => 在途请求数,
//   再除以 18 条 link => 每条 link 的 credit 数。
//
// 关键判据: 未饱和时 在途字节数 随并发度线性增长(队列没满);
//           饱和后 在途字节数 收敛到常数 => 那个常数就是硬件容量上限。
//
// 探针实现: 远端缓冲预先填成一条随机置换的指针环 (stride > 2MB 防 TLB/预取),
//           thread0 循环 CHASE 次 ld.volatile, 每次地址依赖上一次结果。
// ============================================================================
#include "nvl_common.cuh"

#define CHASE 512      // 探针 pointer-chase 步数
#define LOADK 4        // 负载线程每轮独立访存数 (nv08 测出 K*=4)

// 在远端 buffer 上建一条指针环: idx -> next idx, 步长很大且带扰动
__global__ void k_build_ring(unsigned long long* ring, size_t nslot,
                             size_t stride_slot) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < nslot; i += s) {
    size_t nxt = (i * stride_slot + 1) % nslot;  // stride_slot 与 nslot 互质
    ring[i] = (unsigned long long)(ring + nxt);
  }
}

// 负载 + 探针 合体 kernel
//   block0.thread0 -> 探针 (pointer chase, 不参与写)
//   其余所有线程   -> 背景负载 (远端 st128, 每轮 LOADK 个独立请求)
__global__ __launch_bounds__(256) void k_load_probe(
    uint4* __restrict__ dst, size_t n, unsigned long long* __restrict__ ring,
    long long* __restrict__ lat_out, int do_probe) {
  int gtid = blockIdx.x * 256 + threadIdx.x;

  if (do_probe && gtid == 0) {
    // ---- 延迟探针 ----
    unsigned long long p = (unsigned long long)ring;
    long long t0 = clk();
#pragma unroll 1
    for (int i = 0; i < CHASE; ++i) {
      asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(p) : "l"(p) : "memory");
    }
    long long t1 = clk();
    lat_out[0] = t1 - t0;
    lat_out[1] = (long long)(p & 1);  // sink, 防优化
    return;
  }

  // ---- 背景负载 ----
  size_t i = (size_t)gtid;
  size_t s = (size_t)gridDim.x * 256ull;
  uint4 v = make_uint4(gtid, 2, 3, 4);
  for (; i + 3 * s < n; i += 4 * s) {
    st128(dst + i, v);
    st128(dst + i + s, v);
    st128(dst + i + 2 * s, v);
    st128(dst + i + 3 * s, v);
  }
  for (; i < n; i += s) st128(dst + i, v);
}

// 纯探针 (空载延迟基线)
__global__ void k_probe_only(unsigned long long* ring, long long* lat_out) {
  if (threadIdx.x || blockIdx.x) return;
  unsigned long long p = (unsigned long long)ring;
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < CHASE; ++i)
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(p) : "l"(p) : "memory");
  long long t1 = clk();
  lat_out[0] = t1 - t0;
  lat_out[1] = (long long)(p & 1);
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv11_littles_law — %s, %d SM, %.3f GHz\n", env.name, env.sm,
         env.clkGHz);

  const size_t BYTES = 256ull << 20;
  const size_t N = BYTES / sizeof(uint4);
  const size_t RINGB = 64ull << 20;
  const size_t NSLOT = RINGB / 8;

  uint4* remote = (uint4*)nvl_alloc(1, BYTES);
  unsigned long long* ring = (unsigned long long*)nvl_alloc(1, RINGB);
  CK(cudaSetDevice(1));
  // stride 选一个与 NSLOT 互质的大素数倍, 保证遍历且跨 2MB page
  size_t stride = 262147;  // 素数, 262147*8B ≈ 2 MB
  k_build_ring<<<256, 256>>>(ring, NSLOT, stride);
  CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(0));
  long long* lat = (long long*)nvl_alloc(0, 4096);
  CK(cudaSetDevice(0));

  // ---------------------------------------------------- 空载延迟基线
  hdr("0) 空载远端延迟基线 (单线程 pointer-chase, CHASE=512)");
  long long h[2];
  double base_cyc = 1e30;
  for (int r = 0; r < 5; ++r) {
    k_probe_only<<<1, 32>>>(ring, lat);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, lat, 16, cudaMemcpyDeviceToHost));
    double c = (double)h[0] / CHASE;
    if (r && c < base_cyc) base_cyc = c;
  }
  double base_ns = cyc2ns(base_cyc, env.clkGHz);
  printf("空载单次远端依赖 load = %.1f cycles = %.1f ns\n", base_cyc, base_ns);

  // ---------------------------------------------------- 主扫描
  hdr("1) 并发度扫描: 同时测吞吐 λ 与探针延迟 W, 求在途字节 = λ×W");
  printf("每 block 256 thr, 每线程 %d 个独立 st128 展开\n", LOADK);
  printf("%6s %9s %12s %11s %11s %13s %13s %12s\n", "grid", "threads",
         "BW GB/s", "lat cyc", "lat ns", "inflight_B", "inflight/128B", "per-link");
  int grids[] = {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 39, 52, 64, 78, 156, 312};
  int NG = sizeof(grids) / sizeof(int);
  double conv_sum = 0; int conv_n = 0;
  for (int i = 0; i < NG; ++i) {
    int g = grids[i];
    // 吞吐: 不带探针跑 (探针那 1 个线程不写, 会让 block0 少 1 线程, 忽略)
    double ms = bench_ms([&] { k_load_probe<<<g, 256>>>(remote, N, ring, lat, 0); }, 3);
    double bw = gbps(BYTES, ms);

    // 延迟: 带探针跑, 取多次最小(最小值 = 最干净的稳态测量)
    double lc = 1e30;
    for (int r = 0; r < 4; ++r) {
      CK(cudaMemset(lat, 0, 32));
      k_load_probe<<<g, 256>>>(remote, N, ring, lat, 1);
      CK(cudaDeviceSynchronize());
      CKLAST();
      CK(cudaMemcpy(h, lat, 16, cudaMemcpyDeviceToHost));
      if (h[0] > 0) { double c = (double)h[0] / CHASE; if (c < lc) lc = c; }
    }
    double lns = cyc2ns(lc, env.clkGHz);
    double inflight = bw * 1e9 * (lns * 1e-9);  // Bytes = B/s * s
    printf("%6d %9d %12.2f %11.1f %11.1f %13.0f %13.1f %12.1f\n", g, g * 256,
           bw, lc, lns, inflight, inflight / 128.0, inflight / 128.0 / 18.0);
    if (bw > 355.0) { conv_sum += inflight; ++conv_n; }  // 饱和区
  }
  if (conv_n) {
    double m = conv_sum / conv_n;
    printf("\n饱和区(BW>355GB/s, %d 个点)平均在途字节 = %.0f B = %.1f KB\n", conv_n,
           m, m / 1024.0);
    printf("  折算 128B 请求 = %.0f 个;  每条 link (18条) = %.1f 个\n", m / 128.0,
           m / 128.0 / 18.0);
    printf("  折算 256B 请求 = %.0f 个;  每条 link (18条) = %.1f 个\n", m / 256.0,
           m / 256.0 / 18.0);
  }

  printf("\n[读法] 若 inflight_B 在低并发时随 grid 增长、饱和后收敛到常数,\n");
  printf("       该常数即硬件最大在途数据量 (NVLink BDP / credit 池容量)。\n");
  printf("\n[done]\n");
  return 0;
}
