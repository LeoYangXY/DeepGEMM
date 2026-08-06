// ============================================================================
// nv24_ordering —— 跨 GPU 内存序 litmus test (message passing)
//
// 经典 MP 测试:
//   GPU0 (writer):  data = 42 ;  flag = 1
//   GPU1 (reader):  自旋等 flag==1 ;  读 data
//   若读到 data != 42 -> 观察到了 store-store 重排 (违例)。
//
// 对比四种写法 (写侧 x 读侧):
//   0 relaxed : 两个都是普通 st.global,       读侧 ld.volatile
//   1 fence   : data 用 st.global + threadfence_system + flag st.global
//   2 acqrel  : flag 用 st.release.sys, 读侧 ld.acquire.sys
//   3 volatile: 全 volatile 读写
//
// 每次迭代都换一个全新的 (data, flag) 槽位, 避免上一轮的值残留导致假阴性。
// data 初始化为 0, 期望值 42。读到 0 或其它 -> 违例。
//
// 另外做 store-store 观察: GPU0 连续写 A、B 到远端两个不同 128B 块,
// GPU1 轮询看到 B 时检查 A 是否已到, 统计乱序率。
//
// !!! 死锁防护: 所有自旋有 SPIN_MAX 上限, 超时记为 timeout 并继续下一轮
//     (不 break 整个测试, 因为超时不等于违例)。
// ============================================================================
#include "nvl_common.cuh"

#define SPIN_MAX 2000000ull   // 每轮自旋上限, 超时算 timeout 不算违例
#define MAGIC 42u

// 一个 slot: data 和 flag 分别放在不同的 128B line, 避免同 line 顺带可见
struct Slot { unsigned data; unsigned pad0[31]; unsigned flag; unsigned pad1[31]; };

// -------------------------------------------------------------- writer
// MODE: 0 relaxed, 1 fence, 2 acqrel, 3 volatile
template <int MODE>
__global__ void k_writer(Slot* s, int n, long long* tout) {
  long long t0 = clk();
  for (int i = 0; i < n; ++i) {
    Slot* p = s + i;
    if (MODE == 0) {                      // 无任何序保证
      st32(&p->data, MAGIC);
      st32(&p->flag, 1u);
    } else if (MODE == 1) {               // fence 分隔
      st32(&p->data, MAGIC);
      __threadfence_system();
      st32(&p->flag, 1u);
    } else if (MODE == 2) {               // release 语义
      st32(&p->data, MAGIC);
      st_release_sys(&p->flag, 1u);
    } else {                              // volatile
      ((volatile unsigned*)&p->data)[0] = MAGIC;
      ((volatile unsigned*)&p->flag)[0] = 1u;
    }
  }
  long long t1 = clk();
  *tout = t1 - t0;
}

// -------------------------------------------------------------- reader
// RMODE: 0 relaxed(ld.volatile), 1 acquire
template <int RMODE>
__global__ void k_reader(Slot* s, int n, unsigned* nviol, unsigned* nto,
                         long long* tout) {
  unsigned viol = 0, to = 0;
  long long t0 = clk();
  for (int i = 0; i < n; ++i) {
    Slot* p = s + i;
    unsigned long long spin = 0;
    int timedout = 0;
    if (RMODE == 0) {
      while (ld32_v(&p->flag) != 1u) {
        if (++spin > SPIN_MAX) { timedout = 1; break; }
      }
    } else {
      while (ld_acquire_sys(&p->flag) != 1u) {
        if (++spin > SPIN_MAX) { timedout = 1; break; }
      }
    }
    if (timedout) { ++to; continue; }
    // flag 已见, 现在读 data
    unsigned d = (RMODE == 0) ? ld32_v(&p->data) : ld_acquire_sys(&p->data);
    if (d != MAGIC) ++viol;               // 观察到重排
  }
  long long t1 = clk();
  *tout = t1 - t0;
  *nviol = viol;
  *nto = to;
}

// -------------------------------------------------------------- store-store
// GPU0 连续写 A 再写 B (不同 128B 块), GPU1 看到 B 时检查 A
struct SS { unsigned a; unsigned pada[31]; unsigned b; unsigned padb[31]; };
__global__ void k_ss_writer(SS* s, int n) {
  for (int i = 0; i < n; ++i) { st32(&s[i].a, MAGIC); st32(&s[i].b, 1u); }
}
__global__ void k_ss_reader(SS* s, int n, unsigned* nviol, unsigned* nto) {
  unsigned viol = 0, to = 0;
  for (int i = 0; i < n; ++i) {
    unsigned long long spin = 0; int td = 0;
    while (ld32_v(&s[i].b) != 1u) { if (++spin > SPIN_MAX) { td = 1; break; } }
    if (td) { ++to; continue; }
    if (ld32_v(&s[i].a) != MAGIC) ++viol;
  }
  *nviol = viol; *nto = to;
}

// -------------------------------------------------------------- driver
static cudaStream_t s0, s1;
static long long *d_tw, *d_tr;
static unsigned *d_viol, *d_to;

struct Res { unsigned viol, to; double wcyc, rcyc; };

template <int WMODE, int RMODE>
static Res run(Slot* slots, int n, int rounds) {
  Res acc{0, 0, 0, 0};
  for (int r = 0; r < rounds; ++r) {
    // 每轮把 slot 全部清零 (data=0, flag=0) —— 关键, 否则残留值造成假阴性
    CK(cudaSetDevice(1));
    CK(cudaMemset(slots, 0, (size_t)n * sizeof(Slot)));
    CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());

    // 先 launch reader (它会自旋等), 再 launch writer
    CK(cudaSetDevice(1));
    k_reader<RMODE><<<1, 1, 0, s1>>>(slots, n, d_viol, d_to, d_tr);
    CKLAST();
    CK(cudaSetDevice(0));
    k_writer<WMODE><<<1, 1, 0, s0>>>(slots, n, d_tw);
    CKLAST();
    CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());

    unsigned v = 0, t = 0; long long tw = 0, tr = 0;
    CK(cudaSetDevice(1));
    CK(cudaMemcpy(&v, d_viol, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&t, d_to, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&tr, d_tr, 8, cudaMemcpyDeviceToHost));
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(&tw, d_tw, 8, cudaMemcpyDeviceToHost));
    acc.viol += v; acc.to += t;
    acc.wcyc += (double)tw / n; acc.rcyc += (double)tr / n;
  }
  acc.wcyc /= rounds; acc.rcyc /= rounds;
  return acc;
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv24_ordering — %s, %.3f GHz\n", env.name, env.clkGHz);
  printf("# MP litmus: GPU0 写 data=42 再写 flag=1; GPU1 等 flag 后读 data。\n");
  printf("# data 与 flag 分处不同 128B line。每轮 slot 全清零, 每次迭代用新 slot。\n");
  printf("# SPIN_MAX=%llu (超时不计入违例, 单独统计)。\n", SPIN_MAX);

  const int N = 20000;      // 每轮迭代数 (= slot 数)
  const int ROUNDS = 60;    // 轮数 -> 总迭代 1.2 M
  printf("# N=%d slots/轮, ROUNDS=%d -> 总计 %.2f M 次 MP 观测/组合\n",
         N, ROUNDS, (double)N * ROUNDS / 1e6);

  CK(cudaSetDevice(0));
  CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
  CK(cudaSetDevice(1));
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));

  // slot 放在 GPU1 (reader 本地, writer 远端写) —— 最容易暴露重排的方向
  Slot* slots = (Slot*)nvl_alloc(1, (size_t)N * sizeof(Slot));
  SS* ss = (SS*)nvl_alloc(1, (size_t)N * sizeof(SS));
  CK(cudaSetDevice(1));
  d_viol = (unsigned*)nvl_alloc(1, 256);
  d_to = (unsigned*)nvl_alloc(1, 256);
  d_tr = (long long*)nvl_alloc(1, 256);
  CK(cudaSetDevice(0));
  d_tw = (long long*)nvl_alloc(0, 256);
  CK(cudaSetDevice(0));

  hdr("A) MP litmus —— 四种写侧 x 两种读侧");
  printf("%-34s %12s %12s %12s %11s %11s\n", "写侧 / 读侧", "总迭代",
         "违例数", "超时数", "写 cyc/it", "读 cyc/it");

  double tot = (double)N * ROUNDS;
#define ROW(WM, RM, NAME)                                                      \
  {                                                                            \
    Res r = run<WM, RM>(slots, N, ROUNDS);                                     \
    printf("%-34s %12.0f %12u %12u %11.1f %11.1f\n", NAME, tot, r.viol, r.to,  \
           r.wcyc, r.rcyc);                                                    \
    fflush(stdout);                                                            \
  }
  ROW(0, 0, "relaxed st / relaxed ld")
  ROW(0, 1, "relaxed st / acquire ld")
  ROW(1, 0, "fence.sys / relaxed ld")
  ROW(1, 1, "fence.sys / acquire ld")
  ROW(2, 0, "release.sys / relaxed ld")
  ROW(2, 1, "release.sys / acquire ld")
  ROW(3, 0, "volatile / relaxed ld")
  ROW(3, 1, "volatile / acquire ld")

  hdr("B) store-store 乱序观察 (GPU0 写 A 再写 B, GPU1 见 B 查 A)");
  {
    unsigned tv = 0, tt = 0;
    for (int r = 0; r < ROUNDS; ++r) {
      CK(cudaSetDevice(1));
      CK(cudaMemset(ss, 0, (size_t)N * sizeof(SS)));
      CK(cudaDeviceSynchronize());
      CK(cudaSetDevice(1));
      k_ss_reader<<<1, 1, 0, s1>>>(ss, N, d_viol, d_to);
      CKLAST();
      CK(cudaSetDevice(0));
      k_ss_writer<<<1, 1, 0, s0>>>(ss, N);
      CKLAST();
      CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
      CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());
      unsigned v = 0, t = 0;
      CK(cudaMemcpy(&v, d_viol, 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&t, d_to, 4, cudaMemcpyDeviceToHost));
      tv += v; tt += t;
      CK(cudaSetDevice(0));
    }
    printf("总迭代 %.0f, 乱序(见B时A未到) %u 次, 超时 %u 次, 乱序率 %.3e\n",
           tot, tv, tt, tv / tot);
  }

  printf("\n[done]\n");
  return 0;
}
