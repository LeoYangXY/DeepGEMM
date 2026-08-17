// Top-K — 原生 CUDA (bitonic + radix select)
// 编译: nvcc -O3 -std=c++17 -arch=sm_120 -o topk topk.cu && ./topk
//
// =============================================================================
// 算法讲解
// =============================================================================
// 对每行 [M, N] 取最大的 K 个值及其下标，结果按降序排列。
//
// 1) Bitonic TopK (N <= 1024)
//    整行 load 进 smem，bitonic 升序排序，pad -FLT_MAX。
//    最大的 K 个在数组尾部，倒序写出即为降序 TopK。
//    比 "K 轮选最大" 好: 一次排序 O(log² N) 轮 vs K 轮全局 reduce。
//
// 2) Radix Select (大 N 高性能路径, 本文件主力)
//    不需要把整行排完。从高 bit 到低 bit 定位第 K 大的阈值 kth:
//      remaining = K
//      for bit = 31 .. 0:
//        count = 有多少元素的 ordered-uint 在当前前缀下 bit=1
//        if count >= remaining:  第 K 大在 bit=1 那一组, kth |= (1<<bit)
//        else:                   bit=1 的全部都比第 K 大更大, remaining -= count
//    32 次遍历，每次一次 block reduce，复杂度 O(32 * N / threads)。
//
//    收集:
//      所有 ordered > kth 的必须进入 TopK
//      再用 ordered == kth 的填满剩下的名额 (处理并列)
//    最后对 K 个结果做插入排序得到降序。
//
//    float → ordered uint 和 radix sort 相同 (符号位翻转)，
//    这样无符号比较就等价于 float 比较。
//
// 3) 为什么不用 "每轮找 max 再 mask 掉" ?
//    那是 O(K * N) 的选择排序。K=50, N=32K 就要 50 次扫全表。
//    Radix select 永远 32 次，和 K 无关。
// =============================================================================

#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
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

constexpr int kBlock = 256;

__device__ __forceinline__ unsigned float_to_ordered(float f) {
  unsigned u = __float_as_uint(f);
  unsigned mask = (u & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return u ^ mask;
}

__device__ __forceinline__ int warp_reduce_sum(int val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, off);
  return val;
}

__device__ __forceinline__ int block_reduce_sum(int val, int *smem) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = blockDim.x >> 5;

  val = warp_reduce_sum(val);
  if (lane == 0) smem[warp] = val;
  __syncthreads();

  val = (tid < nwarps) ? smem[tid] : 0;
  if (warp == 0) val = warp_reduce_sum(val);
  __syncthreads();
  if (tid == 0) smem[0] = val;
  __syncthreads();
  return smem[0];
}

// -------------------- bitonic topk (one row per block, N <= 1024) --------------------

__global__ void topk_bitonic_kernel(const float *in, float *out_val, int *out_idx,
                                    int N, int K) {
  extern __shared__ char smem[];
  float *sval = reinterpret_cast<float *>(smem);
  int *sidx = reinterpret_cast<int *>(sval + blockDim.x);

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int n2 = blockDim.x;
  const float *row_in = in + (long long)row * N;

  sval[tid] = (tid < N) ? row_in[tid] : -FLT_MAX;
  sidx[tid] = (tid < N) ? tid : -1;
  __syncthreads();

  for (int k = 2; k <= n2; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int ixj = tid ^ j;
      if (ixj > tid) {
        bool ascending = ((tid & k) == 0);
        bool need = ascending ? (sval[tid] > sval[ixj]) : (sval[tid] < sval[ixj]);
        if (need) {
          float tv = sval[tid];
          sval[tid] = sval[ixj];
          sval[ixj] = tv;
          int ti = sidx[tid];
          sidx[tid] = sidx[ixj];
          sidx[ixj] = ti;
        }
      }
      __syncthreads();
    }
  }

  // 升序排完，最大 K 个在尾部，倒序写出
  if (tid < K) {
    int src = n2 - 1 - tid;
    out_val[(long long)row * K + tid] = sval[src];
    out_idx[(long long)row * K + tid] = sidx[src];
  }
}

// -------------------- radix select topk --------------------

__global__ void topk_radix_kernel(const float *in, float *out_val, int *out_idx,
                                  int N, int K) {
  __shared__ int red[32];
  __shared__ unsigned s_kth;
  __shared__ int s_remaining;
  __shared__ int gt_cnt, eq_cnt;

  extern __shared__ char extra[];
  float *sval = reinterpret_cast<float *>(extra);
  int *sidx = reinterpret_cast<int *>(sval + K);

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float *row_in = in + (long long)row * N;

  if (tid == 0) {
    s_kth = 0;
    s_remaining = K;
  }
  __syncthreads();

  for (int bit = 31; bit >= 0; --bit) {
    unsigned kth = s_kth;
    unsigned target = (kth >> bit) | 1u;
    int count = 0;
    for (int i = tid; i < N; i += blockDim.x) {
      unsigned v = float_to_ordered(row_in[i]);
      if ((v >> bit) == target) ++count;
    }
    count = block_reduce_sum(count, red);
    if (tid == 0) {
      if (count >= s_remaining) s_kth = kth | (1u << bit);
      else s_remaining = s_remaining - count;
    }
    __syncthreads();
  }

  unsigned kth = s_kth;

  // 收集 ordered > kth
  if (tid == 0) gt_cnt = 0;
  __syncthreads();
  for (int i = tid; i < N; i += blockDim.x) {
    float fv = row_in[i];
    unsigned v = float_to_ordered(fv);
    if (v > kth) {
      int p = atomicAdd(&gt_cnt, 1);
      if (p < K) {
        sval[p] = fv;
        sidx[p] = i;
      }
    }
  }
  __syncthreads();
  int n_gt = gt_cnt;

  // 用 == kth 填满 (处理并列)。syncthreads 必须在 if 外，避免死锁
  if (tid == 0) eq_cnt = 0;
  __syncthreads();
  if (n_gt < K) {
    for (int i = tid; i < N; i += blockDim.x) {
      float fv = row_in[i];
      unsigned v = float_to_ordered(fv);
      if (v == kth) {
        int p = atomicAdd(&eq_cnt, 1);
        if (n_gt + p < K) {
          sval[n_gt + p] = fv;
          sidx[n_gt + p] = i;
        }
      }
    }
  }
  __syncthreads();

  // K 很小，thread 0 插入排序降序即可
  if (tid == 0) {
    for (int i = 1; i < K; ++i) {
      float kv = sval[i];
      int ki = sidx[i];
      int j = i;
      while (j > 0 && sval[j - 1] < kv) {
        sval[j] = sval[j - 1];
        sidx[j] = sidx[j - 1];
        --j;
      }
      sval[j] = kv;
      sidx[j] = ki;
    }
    float *ov = out_val + (long long)row * K;
    int *oi = out_idx + (long long)row * K;
    for (int i = 0; i < K; ++i) {
      ov[i] = sval[i];
      oi[i] = sidx[i];
    }
  }
}

// -------------------- host --------------------

int next_pow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

void launch_topk(const float *d_in, float *d_val, int *d_idx, int M, int N,
                 int K) {
  if (N <= 1024) {
    int n2 = next_pow2(N);
    size_t smem = n2 * (sizeof(float) + sizeof(int));
    topk_bitonic_kernel<<<M, n2, smem>>>(d_in, d_val, d_idx, N, K);
  } else {
    size_t smem = K * (sizeof(float) + sizeof(int));
    topk_radix_kernel<<<M, kBlock, smem>>>(d_in, d_val, d_idx, N, K);
  }
}

void topk_cpu(const float *in, float *out_val, int *out_idx, int M, int N,
              int K) {
  std::vector<std::pair<float, int>> tmp(N);
  for (int r = 0; r < M; ++r) {
    const float *row = in + (long long)r * N;
    for (int i = 0; i < N; ++i) tmp[i] = {row[i], i};
    std::partial_sort(tmp.begin(), tmp.begin() + K, tmp.end(),
                      [](const std::pair<float, int> &a,
                         const std::pair<float, int> &b) {
                        if (a.first != b.first) return a.first > b.first;
                        return a.second < b.second;
                      });
    for (int k = 0; k < K; ++k) {
      out_val[r * K + k] = tmp[k].first;
      out_idx[r * K + k] = tmp[k].second;
    }
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

struct TopkCtx {
  const float *d_in;
  float *d_val;
  int *d_idx;
  int M, N, K;
};

void topk_cb(void *p) {
  auto *c = (TopkCtx *)p;
  launch_topk(c->d_in, c->d_val, c->d_idx, c->M, c->N, c->K);
}

bool run_case(int M, int N, int K, bool unique) {
  printf("===== TopK  M=%d  N=%d  K=%d  (%s) =====\n", M, N, K,
         N <= 1024 ? "bitonic" : "radix-select");

  const int n = M * N;
  std::vector<float> h_in(n);
  std::vector<float> h_val_cpu(M * K), h_val_gpu(M * K);
  std::vector<int> h_idx_cpu(M * K), h_idx_gpu(M * K);

  srand(7 + M + N + K);
  if (unique) {
    for (int r = 0; r < M; ++r)
      for (int i = 0; i < N; ++i)
        h_in[r * N + i] = (float)i + 0.01f * r;
  } else {
    for (int i = 0; i < n; ++i)
      h_in[i] = (rand() / (float)RAND_MAX) * 100.f - 50.f;
  }

  topk_cpu(h_in.data(), h_val_cpu.data(), h_idx_cpu.data(), M, N, K);

  float *d_in = nullptr, *d_val = nullptr;
  int *d_idx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_val, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_idx, M * K * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  TopkCtx ctx{d_in, d_val, d_idx, M, N, K};
  float ms = bench_ms(topk_cb, &ctx, 5, 20);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_val_gpu.data(), d_val, M * K * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_idx_gpu.data(), d_idx, M * K * sizeof(int),
                        cudaMemcpyDeviceToHost));

  float max_diff = 0.f;
  int idx_mismatch = 0;
  for (int i = 0; i < M * K; ++i) {
    max_diff = fmaxf(max_diff, fabsf(h_val_gpu[i] - h_val_cpu[i]));
    if (h_idx_gpu[i] != h_idx_cpu[i]) ++idx_mismatch;
  }
  // 有并列时下标可以不同，但值必须对。unique 时下标也必须对。
  bool ok_val = max_diff < 1e-5f;
  bool ok_idx = unique ? (idx_mismatch == 0) : true;
  bool ok = ok_val && ok_idx;
  printf("  %7.4f ms  val_diff=%.3e  idx_mismatch=%d  %s\n\n", ms, max_diff,
         idx_mismatch, ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_val));
  CUDA_CHECK(cudaFree(d_idx));
  return ok;
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s  sm_%d%d  mem=%.1f GB\n\n", prop.name, prop.major,
         prop.minor, prop.totalGlobalMem / 1e9);

  bool ok = true;
  ok &= run_case(4, 32, 5, true);
  ok &= run_case(128, 1024, 16, true);
  ok &= run_case(128, 1024, 16, false);
  ok &= run_case(64, 4096, 32, true);
  ok &= run_case(64, 4096, 32, false);
  ok &= run_case(8, 1 << 16, 50, true);
  ok &= run_case(32, 32768, 64, false);

  printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
