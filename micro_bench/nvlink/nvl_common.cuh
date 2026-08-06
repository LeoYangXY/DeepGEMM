// ============================================================================
// nvl_common.cuh —— NVLink 微架构探测公共基础设施
//
// 平台假设: 4x NVIDIA H20 (sm_90), 每卡 18 条 NVLink4 link, 经 4 颗 NVSwitch
//           全互联。所有 GPU-GPU 流量都要过 switch (1 hop)。
//
// 设计原则:
//   1. 所有 kernel 用 volatile / inline PTX, 禁止编译器把远端访存优化掉。
//   2. 带宽用 cudaEvent 计时(设备时钟域), 延迟用 clock64() (SM 时钟域)。
//   3. 每个实验都自带 warmup, 并重复多次取稳定值。
//   4. 输出统一为可直接贴进 markdown 的表格。
// ============================================================================
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------- error check
#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "[CUDA ERR] %s:%d %s -> %s\n", __FILE__, __LINE__, #x,   \
              cudaGetErrorString(e_));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CKLAST()                                                               \
  do {                                                                         \
    cudaError_t e_ = cudaGetLastError();                                       \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "[KERNEL ERR] %s:%d -> %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(e_));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------- topology
struct NvlEnv {
  int ndev;
  int sm;          // SM 数 (dev0)
  double clkGHz;   // SM 时钟 GHz (来自 cudaDeviceProp.clockRate)
  char name[64];
};

static inline NvlEnv nvl_init(int nWant = 2) {
  NvlEnv env{};
  CK(cudaGetDeviceCount(&env.ndev));
  if (env.ndev < nWant) {
    fprintf(stderr, "need >=%d GPUs, got %d\n", nWant, env.ndev);
    exit(1);
  }
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  env.sm = p.multiProcessorCount;
  env.clkGHz = p.clockRate / 1e6;  // clockRate 单位 kHz
  snprintf(env.name, sizeof(env.name), "%s", p.name);
  return env;
}

// 在 [0, n) 范围内两两打开 peer access。返回成功打开的对数。
static inline int nvl_enable_peers(int n) {
  int cnt = 0;
  for (int i = 0; i < n; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      int can = 0;
      CK(cudaDeviceCanAccessPeer(&can, i, j));
      if (!can) continue;
      cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
      if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
      else if (e != cudaSuccess) { fprintf(stderr,"peer %d->%d fail: %s\n",i,j,cudaGetErrorString(e)); continue; }
      ++cnt;
    }
  }
  CK(cudaSetDevice(0));
  return cnt;
}

// ---------------------------------------------------------------- 计时
struct Timer {
  cudaEvent_t a, b;
  Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
  ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
  void start(cudaStream_t s = 0) { CK(cudaEventRecord(a, s)); }
  float stop_ms(cudaStream_t s = 0) {
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms;
  }
};

// 跑 f() rep 次, 返回 best-of-tries 的单次毫秒数
template <typename F>
static inline double bench_ms(F&& f, int rep, int tries = 3) {
  f();  // warmup
  CK(cudaDeviceSynchronize());
  double best = 1e30;
  Timer t;
  for (int k = 0; k < tries; ++k) {
    t.start();
    for (int r = 0; r < rep; ++r) f();
    double ms = t.stop_ms();
    if (ms / rep < best) best = ms / rep;
  }
  CKLAST();
  return best;
}

// ---------------------------------------------------------------- 设备侧原语
// 强制不被优化掉的远端访存。全部用 inline PTX 精确控制指令形态。

// --- 128-bit ---
__device__ __forceinline__ void st128(void* p, const uint4& v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint4 ld128(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
// volatile 版本: 绕过 L1/L2 复用, 每次都真的发到 memory subsystem
__device__ __forceinline__ uint4 ld128_v(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
// --- 64-bit ---
__device__ __forceinline__ void st64(void* p, uint2 v) {
  asm volatile("st.global.v2.u32 [%0], {%1,%2};" ::"l"(p), "r"(v.x), "r"(v.y)
               : "memory");
}
__device__ __forceinline__ uint2 ld64(const void* p) {
  uint2 v;
  asm volatile("ld.global.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
// --- 32-bit ---
__device__ __forceinline__ void st32(void* p, unsigned v) {
  asm volatile("st.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld32(const void* p) {
  unsigned v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ld32_v(const void* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// --- 16 / 8 bit (用于部分写实验) ---
__device__ __forceinline__ void st16(void* p, unsigned short v) {
  asm volatile("st.global.u16 [%0], %1;" ::"l"(p), "h"(v) : "memory");
}
__device__ __forceinline__ void st8(void* p, unsigned char v) {
  unsigned short t = v;
  asm volatile("st.global.u8 [%0], %1;" ::"l"(p), "h"(t) : "memory");
}

// --- 系统级 acquire/release (跨 GPU 可见性) ---
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// 精确 clock64 采样(带 memory barrier, 防止指令重排跨越采样点)
__device__ __forceinline__ long long clk() {
  long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)::"memory");
  return t;
}

// ---------------------------------------------------------------- 主机侧输出
static inline void hdr(const char* title) {
  printf("\n================ %s ================\n", title);
}
static inline void row_hdr(const char* a, const char* b) {
  printf("%-22s %s\n", a, b);
}

// GB/s 计算: bytes 总量 / 毫秒
static inline double gbps(double bytes, double ms) {
  return bytes / 1e9 / (ms / 1e3);
}
// cycles -> ns
static inline double cyc2ns(double c, double clkGHz) { return c / clkGHz; }

// ---------------------------------------------------------------- 内存分配
// 在 dev 上分配 bytes, 返回指针 (调用后当前 device 会被设置成 dev)
static inline void* nvl_alloc(int dev, size_t bytes, int fill = 0) {
  CK(cudaSetDevice(dev));
  void* p = nullptr;
  CK(cudaMalloc(&p, bytes));
  CK(cudaMemset(p, fill, bytes));
  CK(cudaDeviceSynchronize());
  return p;
}
