// ============================================================================
// nv23_pingpong —— 跨 GPU 往返同步延迟 (集合通信的硬地板)
//
// 核心问题:
//   1. 两个 GPU 用 flag 互相 ping-pong 一次要多久? 这是任何跨 GPU 同步
//      (barrier / allreduce 的一步) 的物理下界。
//   2. flag 放哪最快? GPU0 显存 / GPU1 显存 / host pinned(mapped)。
//      —— 决定 NCCL 之类库该把同步变量放哪。
//   3. "写 payload -> 置 flag -> 对端看到 flag -> 读 payload" 的端到端延迟,
//      payload 从 0 扫到 64KB。看 payload 大小对同步延迟的边际影响
//      (即同步延迟里有多少是固定开销, 多少随数据量走)。
//
// 方法:
//   GPU0 跑 k_ping<<<1,1>>>, GPU1 跑 k_pong<<<1,1>>>, 分别在各自 device 的
//   非阻塞 stream 上。两者共用一对 flag (f0/f1):
//     ping: 写 f0 = seq;  自旋等 f1 == seq;  (一个往返)
//     pong: 自旋等 f0 == seq;  写 f1 = seq;
//   ping 侧用 clock64() 量 N 次往返总时间, /N 得单次 RTT。
//
//   写用 st.release.sys, 读用 ld.acquire.sys —— 必须是 .sys scope,
//   否则跨 GPU 不保证可见 (.gpu scope 只在本卡内有序)。
//
// !!! 死锁防护 (这是本实验最大风险) !!!
//   - 每个自旋循环有 SPIN_MAX 上限, 超过就置 timeout 标志并跳出所有循环。
//   - 两个 kernel 都会写自己的 status, 主机侧检查。
//   - 先用 N=64 小迭代 smoke test, 通过了才跑大 N。
//   - 主机侧 launch 顺序: 先 launch pong(消费者, 会先自旋等), 再 launch ping,
//     然后对两个 device 都 cudaDeviceSynchronize。
//   - 所有 flag buffer 在 kernel 启动前 memset 归零并 sync。
// ============================================================================
#include "nvl_common.cuh"

#define SPIN_MAX 200000000ull  // 单次自旋上限 (~0.1s @2GHz), 超时即退出

// 自旋等 *p == want, 返回 0 正常 / 1 超时
__device__ __forceinline__ int spin_until(volatile unsigned* p, unsigned want) {
  unsigned long long n = 0;
  while (ld_acquire_sys((const unsigned*)p) != want) {
    if (++n > SPIN_MAX) return 1;
  }
  return 0;
}

// ---------------------------------------------------------------- 纯 flag RTT
// ping 侧: 量 N 次往返
__global__ void k_ping(unsigned* f0, unsigned* f1, int N, long long* tout,
                       int* status) {
  *status = 0;
  long long t0 = clk();
  for (int i = 1; i <= N; ++i) {
    st_release_sys(f0, (unsigned)i);
    if (spin_until((volatile unsigned*)f1, (unsigned)i)) { *status = 1; break; }
  }
  long long t1 = clk();
  *tout = t1 - t0;
}
// pong 侧: 镜像
__global__ void k_pong(unsigned* f0, unsigned* f1, int N, int* status) {
  *status = 0;
  for (int i = 1; i <= N; ++i) {
    if (spin_until((volatile unsigned*)f0, (unsigned)i)) { *status = 1; break; }
    st_release_sys(f1, (unsigned)i);
  }
}

// ---------------------------------------------------------------- payload 版
// producer: 写 payload(在 consumer 侧的 buffer) -> release flag
// consumer: acquire flag -> 读 payload -> 回 ack
// 量 producer 侧的端到端往返(含 payload 传输 + 对端确认)
__global__ void k_prod(uint4* payload, int nvec, unsigned* f0, unsigned* f1,
                       int N, long long* tout, int* status) {
  *status = 0;
  uint4 v = make_uint4(1, 2, 3, 4);
  long long t0 = clk();
  for (int i = 1; i <= N; ++i) {
    for (int k = 0; k < nvec; ++k) st128(payload + k, v);
    __threadfence_system();               // payload 必须先于 flag 落地
    st_release_sys(f0, (unsigned)i);
    if (spin_until((volatile unsigned*)f1, (unsigned)i)) { *status = 1; break; }
  }
  long long t1 = clk();
  *tout = t1 - t0;
}
__global__ void k_cons(const uint4* payload, int nvec, unsigned* f0,
                       unsigned* f1, int N, int* status, uint4* sink) {
  *status = 0;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int i = 1; i <= N; ++i) {
    if (spin_until((volatile unsigned*)f0, (unsigned)i)) { *status = 1; break; }
    for (int k = 0; k < nvec; ++k) {      // 真读 payload, 防优化
      uint4 x = ld128_v(payload + k);
      acc.x ^= x.x; acc.y ^= x.y; acc.z ^= x.z; acc.w ^= x.w;
    }
    __threadfence_system();
    st_release_sys(f1, (unsigned)i);
  }
  if (acc.x == 0xdeadbeefu && acc.y == 0xfeedu) *sink = acc;
}

// ---------------------------------------------------------------- driver
static cudaStream_t s0, s1;
static long long* d_tout;   // 在 GPU0
static int *d_st0, *d_st1;  // 各自 device

struct RTT { double cyc; double ns; int to; };

// 跑一次 flag ping-pong。f0/f1 是已经可被两侧访问的指针。
static RTT run_flag(unsigned* f0, unsigned* f1, int N, double clkGHz) {
  // 归零
  CK(cudaSetDevice(0));
  CK(cudaMemsetAsync(f0, 0, 4, s0));
  CK(cudaMemsetAsync(f1, 0, 4, s0));
  CK(cudaStreamSynchronize(s0));
  CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());

  // 先 launch pong (它会先进入自旋等), 再 launch ping
  CK(cudaSetDevice(1));
  k_pong<<<1, 1, 0, s1>>>(f0, f1, N, d_st1);
  CKLAST();
  CK(cudaSetDevice(0));
  k_ping<<<1, 1, 0, s0>>>(f0, f1, N, d_tout, d_st0);
  CKLAST();

  CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(0));

  long long c = 0; int st0 = 0, st1 = 0;
  CK(cudaMemcpy(&c, d_tout, sizeof c, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&st0, d_st0, sizeof st0, cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(1));
  CK(cudaMemcpy(&st1, d_st1, sizeof st1, cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(0));
  RTT r;
  r.to = st0 | st1;
  r.cyc = (double)c / N;
  r.ns = cyc2ns(r.cyc, clkGHz);
  return r;
}

static RTT run_payload(uint4* payload, int bytes, unsigned* f0, unsigned* f1,
                       int N, double clkGHz, uint4* sink) {
  int nvec = bytes / 16;
  CK(cudaSetDevice(0));
  CK(cudaMemsetAsync(f0, 0, 4, s0));
  CK(cudaMemsetAsync(f1, 0, 4, s0));
  CK(cudaStreamSynchronize(s0));
  CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());

  CK(cudaSetDevice(1));
  k_cons<<<1, 1, 0, s1>>>(payload, nvec, f0, f1, N, d_st1, sink);
  CKLAST();
  CK(cudaSetDevice(0));
  k_prod<<<1, 1, 0, s0>>>(payload, nvec, f0, f1, N, d_tout, d_st0);
  CKLAST();

  CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(0));

  long long c = 0; int st0 = 0, st1 = 0;
  CK(cudaMemcpy(&c, d_tout, sizeof c, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&st0, d_st0, sizeof st0, cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(1));
  CK(cudaMemcpy(&st1, d_st1, sizeof st1, cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(0));
  RTT r;
  r.to = st0 | st1;
  r.cyc = (double)c / N;
  r.ns = cyc2ns(r.cyc, clkGHz);
  return r;
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv23_pingpong — %s, SM clock %.3f GHz\n", env.name, env.clkGHz);
  printf("# <<<1,1>>> spin kernel on GPU0 / GPU1, flag 用 st.release.sys /\n");
  printf("# ld.acquire.sys。SPIN_MAX=%llu (超时保护)。\n", SPIN_MAX);

  CK(cudaSetDevice(0));
  CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
  CK(cudaSetDevice(1));
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaSetDevice(0));

  // flag 三种归属
  unsigned* fg0 = (unsigned*)nvl_alloc(0, 4096);   // GPU0 显存
  unsigned* fg1 = (unsigned*)nvl_alloc(1, 4096);   // GPU1 显存
  CK(cudaSetDevice(0));
  d_tout = (long long*)nvl_alloc(0, 256);
  d_st0 = (int*)nvl_alloc(0, 256);
  d_st1 = (int*)nvl_alloc(1, 256);
  CK(cudaSetDevice(0));
  uint4* sink = (uint4*)nvl_alloc(1, 4096);
  // host pinned + mapped
  CK(cudaSetDevice(0));
  unsigned* fh = nullptr;
  CK(cudaHostAlloc((void**)&fh, 4096, cudaHostAllocMapped | cudaHostAllocPortable));
  memset(fh, 0, 4096);
  unsigned* fh_dev = nullptr;
  CK(cudaHostGetDevicePointer((void**)&fh_dev, fh, 0));
  CK(cudaSetDevice(0));

  // ---------------------------------------------------- smoke test
  hdr("0) Smoke test (N=64) —— 先确认不死锁");
  {
    RTT r = run_flag(fg0, fg0 + 64, 64, env.clkGHz);
    printf("flag 都在 GPU0: RTT=%.1f cyc (%.1f ns) timeout=%d\n", r.cyc, r.ns, r.to);
    if (r.to) { printf("!! smoke test 超时, 放弃后续 (可能死锁)\n"); return 1; }
  }

  const int N = 20000;
  printf("\n正式测量 N=%d 次往返\n", N);

  // ---------------------------------------------------- A) flag 归属
  hdr("A) 纯 flag ping-pong RTT vs flag 放置位置");
  printf("%-46s %12s %12s %8s\n", "flag 放置 (f0 / f1)", "RTT cyc", "RTT ns", "超时");
  struct FC { const char* n; unsigned* a; unsigned* b; };
  FC cases[] = {
      {"f0=GPU0, f1=GPU0 (都在发起方本地)", fg0, fg0 + 64},
      {"f0=GPU1, f1=GPU1 (都在对端)", fg1, fg1 + 64},
      {"f0=GPU1, f1=GPU0 (各写对端, 各读本地)", fg1, fg0 + 64},
      {"f0=GPU0, f1=GPU1 (各写本地, 各读对端)", fg0, fg1 + 64},
      {"f0=host pinned, f1=host pinned", fh_dev, fh_dev + 64},
  };
  double best_ns = 1e30;
  for (auto& c : cases) {
    RTT r = run_flag(c.a, c.b, N, env.clkGHz);
    printf("%-46s %12.1f %12.1f %8d\n", c.n, r.cyc, r.ns, r.to);
    if (!r.to && r.ns < best_ns) best_ns = r.ns;
  }
  printf("\n[参考] nv20 远端读延迟 ≈ 826 ns, 远端 atom 往返 ≈ 861 ns。\n");
  printf("       一次 ping-pong = 2 个单向跨卡传播 + 2 次自旋检测开销。\n");

  // ---------------------------------------------------- B) payload 扫描
  hdr("B) payload 大小 vs 端到端同步往返 (写payload -> 置flag -> 对端读payload -> ack)");
  printf("payload 在 GPU1 显存 (GPU0 远端写入, GPU1 本地读)\n");
  printf("flag: f0=GPU1(生产者远端写), f1=GPU0(消费者远端写)\n\n");
  printf("%12s %12s %12s %14s %14s\n", "payload B", "RTT cyc", "RTT ns",
         "增量 ns", "等效 GB/s");
  {
    const size_t PBUF = 1ull << 20;
    uint4* pay = (uint4*)nvl_alloc(1, PBUF);
    CK(cudaSetDevice(0));
    int sizes[] = {0, 64, 256, 1024, 4096, 16384, 65536};
    int Np = 4000;  // payload 大时慢, 减少迭代
    double base = -1;
    for (int b : sizes) {
      RTT r = run_payload(pay, b, fg1, fg0 + 64, Np, env.clkGHz, sink);
      if (base < 0) base = r.ns;
      double inc = r.ns - base;
      double gbs = (b > 0 && inc > 0) ? b / inc : 0;  // B/ns = GB/s
      printf("%12d %12.1f %12.1f %14.1f %14.1f%s\n", b, r.cyc, r.ns, inc, gbs,
             r.to ? "  TIMEOUT" : "");
    }
    printf("\n[读法] 增量 ns = 相对 payload=0 的额外耗时 = 纯数据传输代价。\n");
    printf("       等效 GB/s = payload / 增量, 反映单线程搬运 payload 的速率\n");
    printf("       (单线程只能发有限个 outstanding 请求, 所以远低于 370 GB/s 峰值)。\n");
    CK(cudaFree(pay));
  }

  printf("\n[done]\n");
  return 0;
}
