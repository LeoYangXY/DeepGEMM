// Sort — 原生 CUDA (bitonic + LSD radix)
// 编译: nvcc -O3 -std=c++17 -arch=sm_120 -o sort sorting.cu && ./sort
//
// =============================================================================
// 算法讲解
// =============================================================================
//
// 1) Bitonic Sort (N <= 1024, 全部在 shared memory)
//    比较网络: 比较对 (i, i^j) 在编译期就定死，完全 data-independent，GPU 友好。
//    外循环 k = 2,4,...,N  构造长度为 k 的 bitonic 序列
//    内循环 j = k/2, k/4, ..., 1  按距离 j 比较交换
//    方向: (tid & k)==0 升序, 否则降序。最终整体升序。
//    复杂度 O(log² N) 轮，每轮 N/2 次比较，全部并行。
//    非 2 次幂: pad FLT_MAX，排完再切回 N。
//
// 2) LSD Radix Sort (大 N 高性能路径)
//    先把 float 映射成可比较的 uint (符号位翻转):
//      正数: XOR 0x80000000  (把符号位翻成 1，大小关系保持)
//      负数: XOR 0xFFFFFFFF  (按位取反，使得更负的数变成更小的 uint)
//    然后从低 4 bit 到高 4 bit 做 8 趟 (32/4):
//      a. histogram: 每个 block 统计 16 个 bin 的计数
//         布局 hist[bin * num_blocks + block]，bin-major
//      b. exclusive scan 整个 hist → 每个 (bin, block) 的全局起始偏移
//      c. scatter: 块内对每个 thread 的 16-bin 计数做 exclusive scan
//         得到稳定 rank，再写出到 dest
//    必须稳定: LSD 每一趟不稳定，下一趟就会打乱已经排好的低位。
//    所以 scatter 用确定的 thread-major rank，不用 atomic。
//
// =============================================================================

#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

constexpr int kWarp = 32;
constexpr int kBlock = 256;
constexpr int kVec = 4;
constexpr int kTile = kBlock * kVec; // 1024
constexpr int kRadixBits = 4;
constexpr int kRadix = 1 << kRadixBits; // 16

// -------------------- scan primitives (for histogram) --------------------

template <typename T>
__device__ __forceinline__ T warp_inclusive_scan(T val) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int off = 1; off < kWarp; off <<= 1) {
    T other = __shfl_up_sync(0xffffffff, val, off);
    if (lane >= off) val += other;
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T block_exclusive_scan(T val, T *warp_sums,
                                                  T &aggregate) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = blockDim.x >> 5;

  T incl = warp_inclusive_scan(val);
  if (lane == 31) warp_sums[warp] = incl;
  __syncthreads();

  if (warp == 0) {
    T wval = (lane < nwarps) ? warp_sums[lane] : T(0);
    T wincl = warp_inclusive_scan(wval);
    if (lane < nwarps) warp_sums[lane] = wincl;
  }
  __syncthreads();

  aggregate = warp_sums[nwarps - 1];
  T warp_excl = (warp == 0) ? T(0) : warp_sums[warp - 1];
  return warp_excl + incl - val;
}

__device__ __forceinline__ unsigned float_to_ordered(float f) {
  unsigned u = __float_as_uint(f);
  unsigned mask = (u & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return u ^ mask;
}

__device__ __forceinline__ float ordered_to_float(unsigned u) {
  unsigned mask = (u & 0x80000000u) ? 0x80000000u : 0xffffffffu;
  return __uint_as_float(u ^ mask);
}

// -------------------- bitonic --------------------

__global__ void bitonic_sort_shared_kernel(float *data, int n) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int n2 = blockDim.x;
  smem[tid] = (tid < n) ? data[tid] : FLT_MAX;
  __syncthreads();

  for (int k = 2; k <= n2; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int ixj = tid ^ j;
      if (ixj > tid) {
        bool ascending = ((tid & k) == 0);
        if (ascending) {
          if (smem[tid] > smem[ixj]) {
            float t = smem[tid];
            smem[tid] = smem[ixj];
            smem[ixj] = t;
          }
        } else {
          if (smem[tid] < smem[ixj]) {
            float t = smem[tid];
            smem[tid] = smem[ixj];
            smem[ixj] = t;
          }
        }
      }
      __syncthreads();
    }
  }
  if (tid < n) data[tid] = smem[tid];
}

// -------------------- radix encode / decode --------------------

__global__ void encode_kernel(const float *in, unsigned *out, int n,
                              int n_pad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = float_to_ordered(in[i]);
  else if (i < n_pad) out[i] = 0xffffffffu;
}

__global__ void decode_kernel(const unsigned *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = ordered_to_float(in[i]);
}

// -------------------- radix histogram (atomic-free) --------------------

__global__ void radix_hist_kernel(const unsigned *keys, int *hist, int n_pad,
                                  int shift, int num_blocks) {
  __shared__ int warp_hist[kBlock / kWarp][kRadix];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int e = bid * kTile + tid * kVec;

  uint4 kv = *reinterpret_cast<const uint4 *>(keys + e);

  int h[kRadix];
#pragma unroll
  for (int r = 0; r < kRadix; ++r) h[r] = 0;
  h[(kv.x >> shift) & (kRadix - 1)]++;
  h[(kv.y >> shift) & (kRadix - 1)]++;
  h[(kv.z >> shift) & (kRadix - 1)]++;
  h[(kv.w >> shift) & (kRadix - 1)]++;

#pragma unroll
  for (int r = 0; r < kRadix; ++r) {
    int val = h[r];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      val += __shfl_xor_sync(0xffffffff, val, off);
    if (lane == 0) warp_hist[warp][r] = val;
  }
  __syncthreads();

  if (tid < kRadix) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < kBlock / kWarp; ++w) s += warp_hist[w][tid];
    hist[tid * num_blocks + bid] = s;
  }
}

// -------------------- int exclusive scan of histogram --------------------

__global__ void int_scan_tiles_kernel(const int *in, int *out, int *tile_sums,
                                      int n) {
  __shared__ int warp_sums[kBlock / kWarp];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int e = bid * kTile + tid * kVec;

  int4 v = make_int4(0, 0, 0, 0);
  if (e + 3 < n) {
    v = *reinterpret_cast<const int4 *>(in + e);
  } else if (e < n) {
    v.x = in[e];
    if (e + 1 < n) v.y = in[e + 1];
    if (e + 2 < n) v.z = in[e + 2];
  }

  int4 ex;
  ex.x = 0;
  ex.y = v.x;
  ex.z = v.x + v.y;
  ex.w = v.x + v.y + v.z;
  int total = v.x + v.y + v.z + v.w;

  int aggregate;
  int excl = block_exclusive_scan(total, warp_sums, aggregate);
  ex.x += excl;
  ex.y += excl;
  ex.z += excl;
  ex.w += excl;

  if (e + 3 < n) {
    *reinterpret_cast<int4 *>(out + e) = ex;
  } else if (e < n) {
    out[e] = ex.x;
    if (e + 1 < n) out[e + 1] = ex.y;
    if (e + 2 < n) out[e + 2] = ex.z;
  }
  if (tid == 0) tile_sums[bid] = aggregate;
}

__global__ void exclusive_scan_varlen_int(int *data, int n) {
  __shared__ int warp_sums[kBlock / kWarp];
  const int tid = threadIdx.x;
  const int chunk = (n + kBlock - 1) / kBlock;
  const int start = tid * chunk;
  const int end = min(n, start + chunk);

  int my_total = 0;
  for (int i = start; i < end; ++i) my_total += data[i];
  int aggregate;
  int excl = block_exclusive_scan(my_total, warp_sums, aggregate);
  int running = excl;
  for (int i = start; i < end; ++i) {
    int x = data[i];
    data[i] = running;
    running += x;
  }
}

__global__ void int_add_prefix_kernel(int *out, const int *tile_excl, int n) {
  const int bid = blockIdx.x;
  const int addv = tile_excl[bid];
  if (addv == 0) return;
  const int e = bid * kTile + threadIdx.x * kVec;
  if (e + 3 < n) {
    int4 v = *reinterpret_cast<int4 *>(out + e);
    v.x += addv;
    v.y += addv;
    v.z += addv;
    v.w += addv;
    *reinterpret_cast<int4 *>(out + e) = v;
  } else if (e < n) {
    out[e] += addv;
    if (e + 1 < n) out[e + 1] += addv;
    if (e + 2 < n) out[e + 2] += addv;
  }
}

void exclusive_scan_int(int *d_in, int *d_out, int n, int *d_tile_sums) {
  int tiles = CEIL_DIV(n, kTile);
  int_scan_tiles_kernel<<<tiles, kBlock>>>(d_in, d_out, d_tile_sums, n);
  if (tiles > 1) {
    exclusive_scan_varlen_int<<<1, kBlock>>>(d_tile_sums, tiles);
    int_add_prefix_kernel<<<tiles, kBlock>>>(d_out, d_tile_sums, n);
  }
}

// -------------------- radix scatter (stable rank, no atomics) --------------------

__global__ void radix_scatter_kernel(const unsigned *in, unsigned *out,
                                     const int *hist_excl, int shift,
                                     int num_blocks) {
  __shared__ int thist[kRadix * kBlock];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int e = bid * kTile + tid * kVec;

  uint4 kv = *reinterpret_cast<const uint4 *>(in + e);
  int b0 = (kv.x >> shift) & (kRadix - 1);
  int b1 = (kv.y >> shift) & (kRadix - 1);
  int b2 = (kv.z >> shift) & (kRadix - 1);
  int b3 = (kv.w >> shift) & (kRadix - 1);

  int h[kRadix];
#pragma unroll
  for (int r = 0; r < kRadix; ++r) h[r] = 0;
  h[b0]++;
  h[b1]++;
  h[b2]++;
  h[b3]++;

#pragma unroll
  for (int r = 0; r < kRadix; ++r) thist[r * kBlock + tid] = h[r];
  __syncthreads();

  // Hillis-Steele: 每个 bin 沿 tid 做 inclusive scan
  for (int stride = 1; stride < kBlock; stride <<= 1) {
    int t[kRadix];
#pragma unroll
    for (int r = 0; r < kRadix; ++r) {
      t[r] = thist[r * kBlock + tid];
      if (tid >= stride) t[r] += thist[r * kBlock + tid - stride];
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kRadix; ++r) thist[r * kBlock + tid] = t[r];
    __syncthreads();
  }

  // thread exclusive + 本 thread 4 个 key 内部的稳定 rank
  int r0 = 0;
  int r1 = (b1 == b0);
  int r2 = (b2 == b0) + (b2 == b1);
  int r3 = (b3 == b0) + (b3 == b1) + (b3 == b2);

  int dest0 = hist_excl[b0 * num_blocks + bid] +
              (thist[b0 * kBlock + tid] - h[b0]) + r0;
  int dest1 = hist_excl[b1 * num_blocks + bid] +
              (thist[b1 * kBlock + tid] - h[b1]) + r1;
  int dest2 = hist_excl[b2 * num_blocks + bid] +
              (thist[b2 * kBlock + tid] - h[b2]) + r2;
  int dest3 = hist_excl[b3 * num_blocks + bid] +
              (thist[b3 * kBlock + tid] - h[b3]) + r3;

  out[dest0] = kv.x;
  out[dest1] = kv.y;
  out[dest2] = kv.z;
  out[dest3] = kv.w;
}

// -------------------- host sort --------------------

int next_pow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

void launch_bitonic(float *d_data, int n) {
  int n2 = next_pow2(n);
  bitonic_sort_shared_kernel<<<1, n2, n2 * sizeof(float)>>>(d_data, n);
}

void launch_radix(const float *d_in, float *d_out, unsigned *d_a, unsigned *d_b,
                  int *d_hist, int *d_hist_scan, int *d_tile_sums, int n,
                  int n_pad, int num_blocks, int hist_n) {
  int enc_grid = CEIL_DIV(n_pad, kBlock);
  encode_kernel<<<enc_grid, kBlock>>>(d_in, d_a, n, n_pad);

  unsigned *cur = d_a;
  unsigned *nxt = d_b;
  for (int shift = 0; shift < 32; shift += kRadixBits) {
    radix_hist_kernel<<<num_blocks, kBlock>>>(cur, d_hist, n_pad, shift,
                                              num_blocks);
    exclusive_scan_int(d_hist, d_hist_scan, hist_n, d_tile_sums);
    radix_scatter_kernel<<<num_blocks, kBlock>>>(cur, nxt, d_hist_scan, shift,
                                                 num_blocks);
    unsigned *tmp = cur;
    cur = nxt;
    nxt = tmp;
  }
  decode_kernel<<<CEIL_DIV(n, kBlock), kBlock>>>(cur, d_out, n);
}

struct SortBuf {
  unsigned *a, *b;
  int *hist, *hist_scan, *tile_sums;
  int n, n_pad, num_blocks, hist_n;
};

SortBuf alloc_radix_buf(int n) {
  SortBuf b{};
  b.n = n;
  b.n_pad = CEIL_DIV(n, kTile) * kTile;
  b.num_blocks = b.n_pad / kTile;
  b.hist_n = kRadix * b.num_blocks;
  int hist_n_pad = CEIL_DIV(b.hist_n, kTile) * kTile;
  int tiles = CEIL_DIV(hist_n_pad, kTile);
  CUDA_CHECK(cudaMalloc(&b.a, b.n_pad * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&b.b, b.n_pad * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&b.hist, hist_n_pad * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b.hist_scan, hist_n_pad * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b.tile_sums, tiles * sizeof(int)));
  CUDA_CHECK(cudaMemset(b.hist, 0, hist_n_pad * sizeof(int)));
  return b;
}

void free_radix_buf(SortBuf &b) {
  cudaFree(b.a);
  cudaFree(b.b);
  cudaFree(b.hist);
  cudaFree(b.hist_scan);
  cudaFree(b.tile_sums);
}

void launch_sort(const float *d_in, float *d_out, SortBuf *buf, int n) {
  if (n <= 1024) {
    CUDA_CHECK(cudaMemcpy(d_out, d_in, n * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    launch_bitonic(d_out, n);
  } else {
    int hist_n_pad = CEIL_DIV(buf->hist_n, kTile) * kTile;
    CUDA_CHECK(cudaMemset(buf->hist, 0, hist_n_pad * sizeof(int)));
    launch_radix(d_in, d_out, buf->a, buf->b, buf->hist, buf->hist_scan,
                 buf->tile_sums, n, buf->n_pad, buf->num_blocks, buf->hist_n);
  }
}

float bench_ms(void (*fn)(void *), void *ctx, int warmup, int rep) {
  for (int i = 0; i < warmup; ++i) fn(ctx);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < rep; ++i) fn(ctx);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / rep;
}

struct SortCtx {
  const float *d_in;
  float *d_out;
  SortBuf *buf;
  int n;
};

void sort_cb(void *p) {
  auto *c = (SortCtx *)p;
  launch_sort(c->d_in, c->d_out, c->buf, c->n);
}

bool is_sorted_asc(const float *a, int n) {
  for (int i = 1; i < n; ++i) {
    if (a[i] < a[i - 1]) return false;
  }
  return true;
}

bool run_case(int n, const char *tag) {
  printf("===== Sort N=%d (%s) =====\n", n, tag);

  std::vector<float> h_in(n), h_cpu(n), h_gpu(n);
  srand(123 + n);
  for (int i = 0; i < n; ++i)
    h_in[i] = (rand() / (float)RAND_MAX) * 200.f - 100.f;

  // 掺一点重复值
  if (n >= 16) {
    h_in[0] = h_in[1];
    h_in[2] = h_in[3];
  }

  h_cpu = h_in;
  std::sort(h_cpu.begin(), h_cpu.end());

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  SortBuf buf{};
  bool need_radix = n > 1024;
  if (need_radix) buf = alloc_radix_buf(n);

  SortCtx ctx{d_in, d_out, need_radix ? &buf : nullptr, n};

  int warmup = 3, rep = 10;
  if (n >= 1 << 20) {
    warmup = 2;
    rep = 5;
  }
  float ms = bench_ms(sort_cb, &ctx, warmup, rep);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  bool sorted = is_sorted_asc(h_gpu.data(), n);
  float max_diff = 0.f;
  for (int i = 0; i < n; ++i)
    max_diff = fmaxf(max_diff, fabsf(h_gpu[i] - h_cpu[i]));
  bool ok = sorted && max_diff < 1e-5f;
  printf("  [%s] %7.4f ms  sorted=%s  max_diff=%.3e  %s\n\n",
         need_radix ? "radix " : "bitonic", ms, sorted ? "Y" : "N", max_diff,
         ok ? "PASS" : "FAIL");

  if (need_radix) free_radix_buf(buf);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok;
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s  sm_%d%d  mem=%.1f GB\n\n", prop.name, prop.major,
         prop.minor, prop.totalGlobalMem / 1e9);

  bool ok = true;
  ok &= run_case(16, "tiny bitonic");
  ok &= run_case(1024, "smem bitonic");
  ok &= run_case(1025, "radix just over 1 tile");
  ok &= run_case(2048, "radix 2 tiles");
  ok &= run_case(1 << 16, "radix 64K");
  ok &= run_case(1 << 20, "radix 1M");
  ok &= run_case(4 << 20, "radix 4M");

  printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
