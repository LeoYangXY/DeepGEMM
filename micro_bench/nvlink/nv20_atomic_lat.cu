// ============================================================================
// nv20_atomic_lat —— 远端原子操作延迟全谱
//
// 核心问题:
//   1. 一次远端(跨 NVLink) 原子操作要多少 cycle? 相比本地原子贵多少?
//   2. 有返回值(atom) vs 无返回值(red, fire-and-forget) 差多少?
//      -> 差值 = "等返回值的往返代价"。若 red 远快于 atom, 说明写类原子
//         可以 posted (发出去不等确认)。
//   3. scope (.gpu vs .sys) 对延迟的影响 —— .sys 要不要额外 fence/绕过缓存?
//   4. 不同 op (add/exch/cas/min/and) / 不同宽度 (32/64/f32/f64) 的差异。
//
// 方法(保证串行, 测的是真延迟不是吞吐):
//   单线程 <<<1,1>>>, 循环 N 次原子。为了强制串行, 把上一次的返回值
//   拌进下一次的地址偏移(offset = ret & 0) —— 值恒为 0 所以地址不变,
//   但编译器/硬件无法知道, 必须等返回值到达才能发下一条。
//   对 red (无返回值) 没有返回值可依赖 —— 那就直接测"发射速率",
//   这正是我们想要的: red 的 cyc/op = 发射间隔, atom 的 cyc/op = 完整往返。
//   为了让 red 也不被流水线无限堆积掩盖, 额外报告 red + membar 的版本。
//
//   计时用 clock64() (SM 时钟域 1.98GHz), 取 (t1-t0)/N。
// ============================================================================
#include "nvl_common.cuh"

#define NITER 2000

// ---------------------------------------------------------------- PTX 原语
// 有返回值 atom.*
#define DEF_ATOM32(name, ptxop, scope, ty)                                     \
  __device__ __forceinline__ unsigned name(unsigned* p, unsigned v) {          \
    unsigned r;                                                                \
    asm volatile("atom." scope ".global." ptxop "." ty " %0, [%1], %2;"        \
                 : "=r"(r) : "l"(p), "r"(v) : "memory");                       \
    return r;                                                                  \
  }

DEF_ATOM32(a_add_gpu, "add", "gpu", "u32")
DEF_ATOM32(a_exch_gpu, "exch", "gpu", "b32")
DEF_ATOM32(a_min_gpu, "min", "gpu", "u32")
DEF_ATOM32(a_and_gpu, "and", "gpu", "b32")
DEF_ATOM32(a_add_sys, "add", "sys", "u32")
DEF_ATOM32(a_exch_sys, "exch", "sys", "b32")
DEF_ATOM32(a_min_sys, "min", "sys", "u32")
DEF_ATOM32(a_and_sys, "and", "sys", "b32")
DEF_ATOM32(a_add_cta, "add", "cta", "u32")

// CAS 需要三操作数
__device__ __forceinline__ unsigned a_cas_gpu(unsigned* p, unsigned c, unsigned v) {
  unsigned r;
  asm volatile("atom.gpu.global.cas.b32 %0, [%1], %2, %3;"
               : "=r"(r) : "l"(p), "r"(c), "r"(v) : "memory");
  return r;
}
__device__ __forceinline__ unsigned a_cas_sys(unsigned* p, unsigned c, unsigned v) {
  unsigned r;
  asm volatile("atom.sys.global.cas.b32 %0, [%1], %2, %3;"
               : "=r"(r) : "l"(p), "r"(c), "r"(v) : "memory");
  return r;
}

// 64-bit
__device__ __forceinline__ unsigned long long a_add64_gpu(unsigned long long* p,
                                                          unsigned long long v) {
  unsigned long long r;
  asm volatile("atom.gpu.global.add.u64 %0, [%1], %2;"
               : "=l"(r) : "l"(p), "l"(v) : "memory");
  return r;
}
__device__ __forceinline__ unsigned long long a_exch64_gpu(unsigned long long* p,
                                                           unsigned long long v) {
  unsigned long long r;
  asm volatile("atom.gpu.global.exch.b64 %0, [%1], %2;"
               : "=l"(r) : "l"(p), "l"(v) : "memory");
  return r;
}
__device__ __forceinline__ unsigned long long a_add64_sys(unsigned long long* p,
                                                          unsigned long long v) {
  unsigned long long r;
  asm volatile("atom.sys.global.add.u64 %0, [%1], %2;"
               : "=l"(r) : "l"(p), "l"(v) : "memory");
  return r;
}
// float / double
__device__ __forceinline__ float a_addf_gpu(float* p, float v) {
  float r;
  asm volatile("atom.gpu.global.add.f32 %0, [%1], %2;"
               : "=f"(r) : "l"(p), "f"(v) : "memory");
  return r;
}
__device__ __forceinline__ double a_addd_gpu(double* p, double v) {
  double r;
  asm volatile("atom.gpu.global.add.f64 %0, [%1], %2;"
               : "=d"(r) : "l"(p), "d"(v) : "memory");
  return r;
}
__device__ __forceinline__ float a_addf_sys(float* p, float v) {
  float r;
  asm volatile("atom.sys.global.add.f32 %0, [%1], %2;"
               : "=f"(r) : "l"(p), "f"(v) : "memory");
  return r;
}

// 无返回值 red.*  (fire-and-forget)
#define DEF_RED32(name, ptxop, scope, ty)                                      \
  __device__ __forceinline__ void name(unsigned* p, unsigned v) {              \
    asm volatile("red." scope ".global." ptxop "." ty " [%0], %1;" ::"l"(p),   \
                 "r"(v) : "memory");                                           \
  }
DEF_RED32(r_add_gpu, "add", "gpu", "u32")
DEF_RED32(r_min_gpu, "min", "gpu", "u32")
DEF_RED32(r_and_gpu, "and", "gpu", "b32")
DEF_RED32(r_add_sys, "add", "sys", "u32")
DEF_RED32(r_add_cta, "add", "cta", "u32")

__device__ __forceinline__ void r_add64_gpu(unsigned long long* p, unsigned long long v) {
  asm volatile("red.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void r_addf_gpu(float* p, float v) {
  asm volatile("red.gpu.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ void r_addd_gpu(double* p, double v) {
  asm volatile("red.gpu.global.add.f64 [%0], %1;" ::"l"(p), "d"(v) : "memory");
}
// release scope 变体 (带内存序语义)
__device__ __forceinline__ unsigned a_add_rel_sys(unsigned* p, unsigned v) {
  unsigned r;
  asm volatile("atom.add.release.sys.global.u32 %0, [%1], %2;"
               : "=r"(r) : "l"(p), "r"(v) : "memory");
  return r;
}
__device__ __forceinline__ unsigned a_add_acq_sys(unsigned* p, unsigned v) {
  unsigned r;
  asm volatile("atom.add.acquire.sys.global.u32 %0, [%1], %2;"
               : "=r"(r) : "l"(p), "r"(v) : "memory");
  return r;
}

// ---------------------------------------------------------------- 依赖链原语
// 关键: 上一次 atom 的返回值必须真正参与下一条 atom 的地址计算, 否则
// ptxas 会把 (r & 0) 常量折叠成 0, 依赖链断裂 -> 测出来只是发射间隔。
// 用一条不透明的 PTX and.b32 (编译器无法折叠 inline asm 的输出) 把返回值
// 与运行时传入的 mask(=0) 相与, 得到恒 0 但硬件必须等返回值的偏移量。
__device__ __forceinline__ unsigned opaque_zero(unsigned r, unsigned mask) {
  unsigned o;
  asm volatile("and.b32 %0, %1, %2;" : "=r"(o) : "r"(r), "r"(mask));
  return o;
}
// 64-bit 版本: 取低 32 位后同样处理
__device__ __forceinline__ unsigned opaque_zero64(unsigned long long r, unsigned mask) {
  unsigned lo = (unsigned)r;
  unsigned o;
  asm volatile("and.b32 %0, %1, %2;" : "=r"(o) : "r"(lo), "r"(mask));
  return o;
}
// float -> bits -> opaque zero
__device__ __forceinline__ unsigned opaque_zerof(float f, unsigned mask) {
  unsigned b = __float_as_uint(f);
  unsigned o;
  asm volatile("and.b32 %0, %1, %2;" : "=r"(o) : "r"(b), "r"(mask));
  return o;
}

// ---------------------------------------------------------------- kernels
// ===== 32-bit 有返回值, 串行依赖 =====
// op: 0 add, 1 exch, 2 cas, 3 min, 4 and ; sc: 0 gpu, 1 sys, 2 cta
template <int OP, int SC>
__global__ void k_atom32(unsigned* p, long long* tout, unsigned* sink,
                         unsigned mask) {
  unsigned dep = 0;
  // warmup 让指令 cache / TLB 就位
  for (int i = 0; i < 64; ++i) {
    if (SC == 0) dep ^= a_add_gpu(p, 1);
    else if (SC == 1) dep ^= a_add_sys(p, 1);
    else dep ^= a_add_cta(p, 1);
  }
  dep = opaque_zero(dep, mask);
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NITER; ++i) {
    unsigned* q = p + dep;  // 地址依赖上一次返回值 -> 强制串行
    unsigned r;
    if (SC == 0) {
      if (OP == 0) r = a_add_gpu(q, 1);
      else if (OP == 1) r = a_exch_gpu(q, i);
      else if (OP == 2) r = a_cas_gpu(q, 0xffffffffu, i);
      else if (OP == 3) r = a_min_gpu(q, 0xffffffffu);
      else r = a_and_gpu(q, 0xffffffffu);
    } else if (SC == 1) {
      if (OP == 0) r = a_add_sys(q, 1);
      else if (OP == 1) r = a_exch_sys(q, i);
      else if (OP == 2) r = a_cas_sys(q, 0xffffffffu, i);
      else if (OP == 3) r = a_min_sys(q, 0xffffffffu);
      else r = a_and_sys(q, 0xffffffffu);
    } else {
      r = a_add_cta(q, 1);
    }
    dep = opaque_zero(r, mask);  // 恒 0 -> 地址不变; 依赖链保留
  }
  long long t1 = clk();
  *tout = t1 - t0;
  if (dep == 0xdeadbeefu) *sink = dep;
}

// ===== 32-bit release/acquire sys =====
template <int MODE>  // 0 release, 1 acquire
__global__ void k_atom32_ord(unsigned* p, long long* tout, unsigned* sink,
                             unsigned mask) {
  unsigned dep = 0;
  for (int i = 0; i < 64; ++i) dep ^= a_add_sys(p, 1);
  dep = opaque_zero(dep, mask);
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NITER; ++i) {
    unsigned* q = p + dep;
    unsigned r = (MODE == 0) ? a_add_rel_sys(q, 1) : a_add_acq_sys(q, 1);
    dep = opaque_zero(r, mask);
  }
  long long t1 = clk();
  *tout = t1 - t0;
  if (dep == 0xdeadbeefu) *sink = dep;
}

// ===== 64 / float / double 有返回值 =====
// W: 0 u64.add, 1 b64.exch, 2 f32.add, 3 f64.add ; SC: 0 gpu 1 sys
template <int W, int SC>
__global__ void k_atomW(void* p, long long* tout, unsigned* sink, unsigned mask) {
  unsigned dep = 0;
  // warmup
  for (int i = 0; i < 64; ++i) dep ^= (unsigned)a_add64_gpu((unsigned long long*)p, 1);
  dep = opaque_zero(dep, mask);
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NITER; ++i) {
    char* q = (char*)p + dep;
    if (W == 0) {
      unsigned long long r = (SC == 0) ? a_add64_gpu((unsigned long long*)q, 1)
                                       : a_add64_sys((unsigned long long*)q, 1);
      dep = opaque_zero64(r, mask);
    } else if (W == 1) {
      unsigned long long r = a_exch64_gpu((unsigned long long*)q, i);
      dep = opaque_zero64(r, mask);
    } else if (W == 2) {
      float r = (SC == 0) ? a_addf_gpu((float*)q, 1.0f) : a_addf_sys((float*)q, 1.0f);
      dep = opaque_zerof(r, mask);
    } else {
      double r = a_addd_gpu((double*)q, 1.0);
      dep = opaque_zerof((float)r, mask);
    }
  }
  long long t1 = clk();
  *tout = t1 - t0;
  if (dep == 0xdeadbeefu) *sink = dep;
}

// ===== red.* 无返回值 =====
// 没有返回值 -> 无法做依赖链。测的是"发射间隔"。
// FENCE=1 时每次 red 后加 membar.sys, 强制等它真正完成 -> 得到完成延迟上界。
// OP: 0 add32, 1 min32, 2 and32, 3 add64, 4 addf32, 5 addf64
// SC: 0 gpu, 1 sys, 2 cta
template <int OP, int SC, int FENCE>
__global__ void k_red(void* p, long long* tout) {
  // warmup
  for (int i = 0; i < 64; ++i) r_add_gpu((unsigned*)p, 1);
  __threadfence_system();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NITER; ++i) {
    if (OP == 0) {
      if (SC == 0) r_add_gpu((unsigned*)p, 1);
      else if (SC == 1) r_add_sys((unsigned*)p, 1);
      else r_add_cta((unsigned*)p, 1);
    } else if (OP == 1) r_min_gpu((unsigned*)p, 0xffffffffu);
    else if (OP == 2) r_and_gpu((unsigned*)p, 0xffffffffu);
    else if (OP == 3) r_add64_gpu((unsigned long long*)p, 1);
    else if (OP == 4) r_addf_gpu((float*)p, 1.0f);
    else r_addd_gpu((double*)p, 1.0);
    if (FENCE) __threadfence_system();
  }
  long long t1 = clk();
  *tout = t1 - t0;
}

// 对照: 纯 st.global.u32 (非原子写) 的发射间隔 + 带 fence 的完成延迟
template <int FENCE>
__global__ void k_st(unsigned* p, long long* tout) {
  for (int i = 0; i < 64; ++i) st32(p, i);
  __threadfence_system();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NITER; ++i) {
    st32(p, i);
    if (FENCE) __threadfence_system();
  }
  long long t1 = clk();
  *tout = t1 - t0;
}
// 对照: 依赖链 ld.volatile (纯读延迟)
__global__ void k_ld(unsigned* p, long long* tout, unsigned* sink, unsigned mask) {
  unsigned dep = 0;
  for (int i = 0; i < 64; ++i) dep ^= ld32_v(p);
  dep = opaque_zero(dep, mask);
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NITER; ++i) dep = opaque_zero(ld32_v(p + dep), mask);
  long long t1 = clk();
  *tout = t1 - t0;
  if (dep == 0xdeadbeefu) *sink = dep;
}

// ---------------------------------------------------------------- driver
static long long* g_tout;
static unsigned* g_sink;
static double g_clk;

template <typename F>
static double measure(F&& launch) {
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    CK(cudaMemset(g_tout, 0, sizeof(long long)));
    launch();
    CK(cudaDeviceSynchronize());
    CKLAST();
    long long h = 0;
    CK(cudaMemcpy(&h, g_tout, sizeof h, cudaMemcpyDeviceToHost));
    double c = (double)h / NITER;
    if (r > 0 && c < best) best = c;
  }
  return best;
}

struct Pair { double loc, rem; };
static void prow(const char* tag, Pair p) {
  printf("%-34s %10.1f %10.1f %10.3f %10.3f %10.2f\n", tag, p.loc, p.rem,
         cyc2ns(p.loc, g_clk), cyc2ns(p.rem, g_clk), p.rem / (p.loc > 0 ? p.loc : 1));
}
static void phdr() {
  printf("%-34s %10s %10s %10s %10s %10s\n", "操作", "本地cyc", "远端cyc",
         "本地ns", "远端ns", "远/本");
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  g_clk = env.clkGHz;
  printf("# nv20_atomic_lat — %s, SM clock %.3f GHz, NITER=%d\n", env.name,
         env.clkGHz, NITER);
  printf("# 单线程 <<<1,1>>> 串行原子; atom.* 用返回值做地址依赖强制串行,\n");
  printf("# red.* 无返回值故测发射间隔, 另给 red+membar.sys 的完成延迟。\n");

  void* loc = nvl_alloc(0, 4096);
  void* rem = nvl_alloc(1, 4096);
  CK(cudaSetDevice(0));
  g_tout = (long long*)nvl_alloc(0, 256);
  g_sink = (unsigned*)nvl_alloc(0, 256);
  CK(cudaSetDevice(0));

  // ============================================================ A) 32-bit ops
  hdr("A) 32-bit 原子, 有返回值 atom.* (串行依赖链) — scope=.gpu");
  phdr();
#define RUN32(OP, SC, NAME)                                                    \
  {                                                                            \
    Pair p;                                                                    \
    p.loc = measure([&] { k_atom32<OP, SC><<<1, 1>>>((unsigned*)loc, g_tout, g_sink, 0u); }); \
    p.rem = measure([&] { k_atom32<OP, SC><<<1, 1>>>((unsigned*)rem, g_tout, g_sink, 0u); }); \
    prow(NAME, p);                                                             \
  }
  RUN32(0, 0, "atom.gpu.add.u32")
  RUN32(1, 0, "atom.gpu.exch.b32")
  RUN32(2, 0, "atom.gpu.cas.b32")
  RUN32(3, 0, "atom.gpu.min.u32")
  RUN32(4, 0, "atom.gpu.and.b32")

  hdr("B) 同上但 scope=.sys");
  phdr();
  RUN32(0, 1, "atom.sys.add.u32")
  RUN32(1, 1, "atom.sys.exch.b32")
  RUN32(2, 1, "atom.sys.cas.b32")
  RUN32(3, 1, "atom.sys.min.u32")
  RUN32(4, 1, "atom.sys.and.b32")
  {
    Pair p;
    p.loc = measure([&] { k_atom32_ord<0><<<1, 1>>>((unsigned*)loc, g_tout, g_sink, 0u); });
    p.rem = measure([&] { k_atom32_ord<0><<<1, 1>>>((unsigned*)rem, g_tout, g_sink, 0u); });
    prow("atom.add.release.sys.u32", p);
    p.loc = measure([&] { k_atom32_ord<1><<<1, 1>>>((unsigned*)loc, g_tout, g_sink, 0u); });
    p.rem = measure([&] { k_atom32_ord<1><<<1, 1>>>((unsigned*)rem, g_tout, g_sink, 0u); });
    prow("atom.add.acquire.sys.u32", p);
  }
  RUN32(0, 2, "atom.cta.add.u32 (对照)")

  // ============================================================ C) 宽度
  hdr("C) 宽度扫描, 有返回值 atom.*");
  phdr();
#define RUNW(W, SC, NAME)                                                      \
  {                                                                            \
    Pair p;                                                                    \
    p.loc = measure([&] { k_atomW<W, SC><<<1, 1>>>(loc, g_tout, g_sink, 0u); });   \
    p.rem = measure([&] { k_atomW<W, SC><<<1, 1>>>(rem, g_tout, g_sink, 0u); });   \
    prow(NAME, p);                                                             \
  }
  RUNW(0, 0, "atom.gpu.add.u64")
  RUNW(0, 1, "atom.sys.add.u64")
  RUNW(1, 0, "atom.gpu.exch.b64")
  RUNW(2, 0, "atom.gpu.add.f32")
  RUNW(2, 1, "atom.sys.add.f32")
  RUNW(3, 0, "atom.gpu.add.f64")

  // ============================================================ D) red 无返回
  hdr("D) 无返回值 red.* —— 发射间隔 (fire-and-forget, 无 fence)");
  printf("   这一栏测的是 issue rate, 不是完成延迟。远端若≈本地 -> posted。\n");
  phdr();
#define RUNRED(OP, SC, F, NAME)                                                \
  {                                                                            \
    Pair p;                                                                    \
    p.loc = measure([&] { k_red<OP, SC, F><<<1, 1>>>(loc, g_tout); });         \
    p.rem = measure([&] { k_red<OP, SC, F><<<1, 1>>>(rem, g_tout); });         \
    prow(NAME, p);                                                             \
  }
  RUNRED(0, 0, 0, "red.gpu.add.u32")
  RUNRED(0, 1, 0, "red.sys.add.u32")
  RUNRED(0, 2, 0, "red.cta.add.u32")
  RUNRED(1, 0, 0, "red.gpu.min.u32")
  RUNRED(2, 0, 0, "red.gpu.and.b32")
  RUNRED(3, 0, 0, "red.gpu.add.u64")
  RUNRED(4, 0, 0, "red.gpu.add.f32")
  RUNRED(5, 0, 0, "red.gpu.add.f64")
  {
    Pair p;
    p.loc = measure([&] { k_st<0><<<1, 1>>>((unsigned*)loc, g_tout); });
    p.rem = measure([&] { k_st<0><<<1, 1>>>((unsigned*)rem, g_tout); });
    prow("st.global.u32 (非原子对照)", p);
  }

  hdr("E) red.* + membar.sys —— 强制等完成, 得到真实完成延迟");
  phdr();
  RUNRED(0, 0, 1, "red.gpu.add.u32 + fence.sys")
  RUNRED(0, 1, 1, "red.sys.add.u32 + fence.sys")
  RUNRED(3, 0, 1, "red.gpu.add.u64 + fence.sys")
  {
    Pair p;
    p.loc = measure([&] { k_st<1><<<1, 1>>>((unsigned*)loc, g_tout); });
    p.rem = measure([&] { k_st<1><<<1, 1>>>((unsigned*)rem, g_tout); });
    prow("st.global.u32 + fence.sys", p);
    p.loc = measure([&] { k_ld<<<1, 1>>>((unsigned*)loc, g_tout, g_sink, 0u); });
    p.rem = measure([&] { k_ld<<<1, 1>>>((unsigned*)rem, g_tout, g_sink, 0u); });
    prow("ld.volatile.u32 (纯读延迟对照)", p);
  }

  printf("\n[读法] atom(有返回) - red(无返回,无fence) = 等返回值的往返代价。\n");
  printf("       red+fence ≈ 完成延迟; red 裸值 ≈ 发射间隔。\n");
  printf("\n[done]\n");
  return 0;
}
