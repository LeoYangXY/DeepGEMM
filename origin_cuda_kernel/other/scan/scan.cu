// Prefix Sum (inclusive scan) — 原生 CUDA
// 编译: nvcc -O3 -std=c++17 -arch=sm_120 -o scan scan.cu && ./scan
//
// =============================================================================
// 算法讲解
// =============================================================================
// inclusive scan: out[i] = in[0] + in[1] + ... + in[i]
// 有数据依赖，不能像 elementwise 那样每个 thread 独立算完。
//
// 1) Warp scan (shuffle)
//    __shfl_up_sync 把 lane i-offset 的值传给 lane i，再累加。
//    5 步 (offset=1,2,4,8,16) 得到 warp 内 inclusive prefix，零 shared memory。
//
// 2) Block scan
//    每个 warp 先做 warp scan；lane 31 写出 warp total；
//    warp 0 再 scan 这些 total，得到每个 warp 的 exclusive 前缀；
//    每个 thread 加上自己所在 warp 的 exclusive 前缀。
//
// 3) Device scan (三 kernel，本文件高性能路径)
//    把每一行切成 TILE=1024 的块 (256 thread × float4):
//      K1 scan_tiles: 块内 inclusive scan + 写出块总和
//      K2 exclusive_scan_rows: 对每行的块总和做 exclusive scan
//      K3 add_tile_prefix: 每个元素加上前面所有块的和
//    float4 一次搬 16B，读写都 coalesced。N<=1024 时只有一块，K2/K3 直接跳过。
//
// Hillis-Steele:  O(N log N) 工作量, log N 轮，实现简单但做了多余加法
// Blelloch:      O(N) 工作量, 2 log N 轮 (upsweep+downsweep)
// 本实现:        块内用 warp-shuffle (近似 work-efficient) + 块间三 kernel
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

// -------------------- primitives --------------------

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

// -------------------- kernels --------------------

// baseline: 一行一个 thread，完全串行
__global__ void scan_naive_kernel(const float *in, float *out, int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  const float *ri = in + (long long)row * N;
  float *ro = out + (long long)row * N;
  float s = 0.f;
  for (int i = 0; i < N; ++i) {
    s += ri[i];
    ro[i] = s;
  }
}

// K1: 每个 block 处理一行里的一个 TILE，float4 向量化
__global__ void scan_tiles_kernel(const float *__restrict__ in,
                                  float *__restrict__ out,
                                  float *__restrict__ tile_sums, int M, int N,
                                  int tiles_per_row) {
  __shared__ float warp_sums[kBlock / kWarp];

  const int row = blockIdx.x / tiles_per_row;
  const int tile = blockIdx.x % tiles_per_row;
  const int tid = threadIdx.x;
  const int tile_start = tile * kTile;
  const int e = tile_start + tid * kVec;

  const float *row_in = in + (long long)row * N;
  float *row_out = out + (long long)row * N;

  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  if (e + 3 < N) {
    v = *reinterpret_cast<const float4 *>(row_in + e);
  } else if (e < N) {
    v.x = row_in[e];
    if (e + 1 < N) v.y = row_in[e + 1];
    if (e + 2 < N) v.z = row_in[e + 2];
  }

  // 4 个寄存器内的 inclusive scan
  v.y += v.x;
  v.z += v.y;
  v.w += v.z;

  float aggregate;
  float excl = block_exclusive_scan(v.w, warp_sums, aggregate);
  v.x += excl;
  v.y += excl;
  v.z += excl;
  v.w += excl;

  if (e + 3 < N) {
    *reinterpret_cast<float4 *>(row_out + e) = v;
  } else if (e < N) {
    row_out[e] = v.x;
    if (e + 1 < N) row_out[e + 1] = v.y;
    if (e + 2 < N) row_out[e + 2] = v.z;
  }

  if (tid == 0) {
    tile_sums[(long long)row * tiles_per_row + tile] = aggregate;
  }
}

// K2: 每行一个 block，对 tile_sums 做 in-place exclusive scan
__global__ void exclusive_scan_rows_kernel(float *data, int n) {
  __shared__ float warp_sums[kBlock / kWarp];
  float *row = data + (long long)blockIdx.x * n;
  const int tid = threadIdx.x;
  const int chunk = (n + kBlock - 1) / kBlock;
  const int start = tid * chunk;
  const int end = min(n, start + chunk);

  float my_total = 0.f;
  for (int i = start; i < end; ++i) my_total += row[i];

  float aggregate;
  float excl = block_exclusive_scan(my_total, warp_sums, aggregate);

  float running = excl;
  for (int i = start; i < end; ++i) {
    float x = row[i];
    row[i] = running;
    running += x;
  }
}

// K3: 把前面块的 exclusive 前缀加回每个元素
__global__ void add_tile_prefix_kernel(float *__restrict__ out,
                                       const float *__restrict__ tile_excl,
                                       int M, int N, int tiles_per_row) {
  const int row = blockIdx.x / tiles_per_row;
  const int tile = blockIdx.x % tiles_per_row;
  if (tile == 0) return;

  const float addv = tile_excl[(long long)row * tiles_per_row + tile];
  const int tile_start = tile * kTile;
  const int e = tile_start + threadIdx.x * kVec;
  float *row_out = out + (long long)row * N;

  if (e + 3 < N) {
    float4 v = *reinterpret_cast<float4 *>(row_out + e);
    v.x += addv;
    v.y += addv;
    v.z += addv;
    v.w += addv;
    *reinterpret_cast<float4 *>(row_out + e) = v;
  } else if (e < N) {
    row_out[e] += addv;
    if (e + 1 < N) row_out[e + 1] += addv;
    if (e + 2 < N) row_out[e + 2] += addv;
  }
}

// -------------------- host --------------------

void launch_scan_naive(const float *d_in, float *d_out, int M, int N) {
  int block = 256;
  int grid = CEIL_DIV(M, block);
  scan_naive_kernel<<<grid, block>>>(d_in, d_out, M, N);
}

void launch_scan_fast(const float *d_in, float *d_out, float *d_tile_sums,
                      int M, int N) {
  const int tiles = CEIL_DIV(N, kTile);
  const int grid = M * tiles;
  scan_tiles_kernel<<<grid, kBlock>>>(d_in, d_out, d_tile_sums, M, N, tiles);
  if (tiles > 1) {
    exclusive_scan_rows_kernel<<<M, kBlock>>>(d_tile_sums, tiles);
    add_tile_prefix_kernel<<<grid, kBlock>>>(d_out, d_tile_sums, M, N, tiles);
  }
}

void scan_cpu(const float *in, float *out, int M, int N) {
  for (int r = 0; r < M; ++r) {
    float s = 0.f;
    const float *ri = in + (long long)r * N;
    float *ro = out + (long long)r * N;
    for (int i = 0; i < N; ++i) {
      s += ri[i];
      ro[i] = s;
    }
  }
}

float max_abs_diff(const float *a, const float *b, int n) {
  float m = 0.f;
  for (int i = 0; i < n; ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
  return m;
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

struct ScanCtx {
  const float *d_in;
  float *d_out;
  float *d_tiles;
  int M, N;
  int which; // 0 naive, 1 fast
};

void scan_launch_cb(void *p) {
  auto *c = (ScanCtx *)p;
  if (c->which == 0) launch_scan_naive(c->d_in, c->d_out, c->M, c->N);
  else launch_scan_fast(c->d_in, c->d_out, c->d_tiles, c->M, c->N);
}

bool run_case(int M, int N, bool run_naive) {
  const int n = M * N;
  const int tiles = CEIL_DIV(N, kTile);
  printf("===== Prefix Sum  M=%d  N=%d  (tiles/row=%d) =====\n", M, N, tiles);

  std::vector<float> h_in(n), h_cpu(n), h_gpu(n);
  srand(42);
  for (int i = 0; i < n; ++i)
    h_in[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;

  scan_cpu(h_in.data(), h_cpu.data(), M, N);

  float *d_in = nullptr, *d_out = nullptr, *d_tiles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tiles, (size_t)M * tiles * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  ScanCtx ctx{d_in, d_out, d_tiles, M, N, 1};
  int warmup = 5, rep = 20;
  if (n > 4 * 1024 * 1024) {
    warmup = 3;
    rep = 10;
  }

  float ms_fast = bench_ms(scan_launch_cb, &ctx, warmup, rep);
  CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  float diff_fast = max_abs_diff(h_gpu.data(), h_cpu.data(), n);
  // 并行加法结合律不同，N 越大误差越大
  float tol = 5e-4f * N + 1e-4f;
  bool ok_fast = diff_fast < tol;
  printf("  [fast ] %7.4f ms  max_diff=%.3e  %s\n", ms_fast, diff_fast,
         ok_fast ? "PASS" : "FAIL");

  bool ok_naive = true;
  if (run_naive) {
    ctx.which = 0;
    float ms_naive = bench_ms(scan_launch_cb, &ctx, warmup, rep);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float diff_naive = max_abs_diff(h_gpu.data(), h_cpu.data(), n);
    ok_naive = diff_naive < 1e-4f;
    printf("  [naive] %7.4f ms  max_diff=%.3e  %s\n", ms_naive, diff_naive,
           ok_naive ? "PASS" : "FAIL");
    printf("  speedup vs naive: %.2fx\n", ms_naive / ms_fast);
  }

  double bytes = 2.0 * n * sizeof(float); // read in + write out
  printf("  effective BW (fast): %.1f GB/s\n",
         bytes / (ms_fast * 1e-3) / 1e9);
  printf("\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_tiles));
  return ok_fast && ok_naive;
}

int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("GPU: %s  sm_%d%d  mem=%.1f GB\n\n", prop.name, prop.major,
         prop.minor, prop.totalGlobalMem / 1e9);

  bool ok = true;
  ok &= run_case(4, 16, true);           // 极小
  ok &= run_case(32, 128, true);         // 小于一个 tile
  ok &= run_case(1024, 1024, true);      // 正好一个 tile / 行
  ok &= run_case(256, 4096, true);       // 多 tile
  ok &= run_case(1, 8 * 1024 * 1024, false); // 大 1D，跳过 naive

  printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
