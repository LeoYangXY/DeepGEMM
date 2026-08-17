#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <nvtx3/nvtx3.hpp>

// ============================================================================
// TMP DEMO: TMA (cuda::memcpy_async) 搬运 与 MMA (Tensor Core) 计算 的 overlap
//   本机: sm_120 (Blackwell RTX 5050), CUDA 13.2
//   看完即删 (tmp_)
//
// 用官方 C++ API (cuda::pipeline / cuda::memcpy_async) 做 software pipeline,
// 实现 "搬第 k+1 块" 与 "算第 k 块" 的 overlap。
//
// 看 overlap 的方法:
//   - nsys : 宏观 (kernel launch 时间线, 本 demo 两个 kernel 顺序, 主要看单 kernel)
//   - ncu  : 微观 (PM Sampling 里 SM/Tensor pipe 与 TMA 曲线同时活跃 = overlap)
// ============================================================================

constexpr int TILE_M = 64, TILE_N = 64, TILE_K = 16, NUM_STAGES = 2;
constexpr int A_TILE_BYTES = TILE_M * TILE_K * 2;
constexpr int B_TILE_BYTES = TILE_K * TILE_N * 2;

// ---- pipeline 版: TMA 与 MMA overlap ----
__global__ void gemm_pipeline(const half* __restrict__ A,
                               const half* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    auto pipeline = cuda::make_pipeline();

    extern __shared__ char smem_raw[];
    half* smem_A[NUM_STAGES];
    half* smem_B[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; ++s) {
        smem_A[s] = reinterpret_cast<half*>(smem_raw + s * (A_TILE_BYTES + B_TILE_BYTES));
        smem_B[s] = reinterpret_cast<half*>(smem_raw + s * (A_TILE_BYTES + B_TILE_BYTES) + A_TILE_BYTES);
    }

    int m_block = blockIdx.y, n_block = blockIdx.x;
    int m_base = m_block * TILE_M, n_base = n_block * TILE_N;
    float acc[4][8];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    const half* A_base = A + m_base * K;
    const half* B_base = B + n_base;

    int num_k = K / TILE_K;

    // prefetch stage 0
    pipeline.producer_acquire();
    cuda::memcpy_async(smem_A[0], A_base, A_TILE_BYTES, pipeline);
    cuda::memcpy_async(smem_B[0], B_base, B_TILE_BYTES, pipeline);
    pipeline.producer_commit();

    for (int kk = 0; kk < num_k; ++kk) {
        int cur = kk % NUM_STAGES;
        int nxt = (kk + 1) % NUM_STAGES;
        int k_next = (kk + 1) * TILE_K;

        // PRODUCER: 异步搬下一块 (与下面 MMA 重叠)
        if (kk + 1 < num_k) {
            pipeline.producer_acquire();
            cuda::memcpy_async(smem_A[nxt], A_base + k_next * K, A_TILE_BYTES, pipeline);
            cuda::memcpy_async(smem_B[nxt], B_base + k_next * N, B_TILE_BYTES, pipeline);
            pipeline.producer_commit();
        }

        // 等当前 stage 完成
        pipeline.consumer_wait();

        // CONSUMER: 计算当前块 (真实 GEMM 这里换成 mma.sync/wgmma Tensor Core)
        // 这里用 FP32 点积代替, 重点演示 "与下面 TMA 搬运重叠", 行为等价
        half* a_ptr = smem_A[cur];
        half* b_ptr = smem_B[cur];
        for (int i = 0; i < 4; ++i)        // 行方向 4 个 16
            for (int j = 0; j < 8; ++j) {  // 列方向 8 个 8
                float sum = 0.0f;
                for (int t = 0; t < 16; ++t) {
                    float av = __half2float(a_ptr[(i*16)*TILE_K + t]);
                    float bv = __half2float(b_ptr[j*8 + t]);
                    sum += av * bv;
                }
                acc[i][j] += sum;
            }
        pipeline.consumer_release();
    }

    float* C_base = C + m_base * N + n_base;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) {
            int row = i * 16 + (threadIdx.x % 16);
            int col = j * 8 + (threadIdx.x / 16);
            if (m_base + row < M && n_base + col < N)
                C_base[row * N + col] = acc[i][j];
        }
}

// ---- naive 版: 无 pipeline, 串行 (逐块 __ldg 后再算) ----
__global__ void gemm_naive(const half* __restrict__ A,
                            const half* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    int m_block = blockIdx.y, n_block = blockIdx.x;
    int m_base = m_block * TILE_M, n_base = n_block * TILE_N;
    float acc[4][8];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    const half* A_base = A + m_base * K;
    const half* B_base = B + n_base;

    half a_reg[4][16], b_reg[8];
    for (int k = 0; k < K; k += TILE_K) {
        const half* a_src = A_base + k * K;
        const half* b_src = B_base + k * N;
        for (int i = 0; i < 4; ++i)
            for (int t = 0; t < 16; ++t)
                a_reg[i][t] = __ldg(&a_src[i * 16 * TILE_K + t]);
        for (int j = 0; j < 8; ++j)
            for (int t = 0; t < 8; ++t)
                b_reg[j] = __ldg(&b_src[j * 8 + t]);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 8; ++j) {
                float sum = 0.0f;
                for (int t = 0; t < 16; ++t) {
                    float av = __half2float(a_reg[i][t]);
                    float bv = __half2float(b_reg[j]);
                    sum += av * bv;
                }
                acc[i][j] += sum;
            }
    }
    float* C_base = C + m_base * N + n_base;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) {
            int row = i * 16 + (threadIdx.x % 16);
            int col = j * 8 + (threadIdx.x / 16);
            if (m_base + row < M && n_base + col < N)
                C_base[row * N + col] = acc[i][j];
        }
}

#define CUDA_CHECK(call) do {                                          \
    cudaError_t e = (call);                                            \
    if (e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(e)); exit(1);                      \
    }                                                                  \
} while(0)

int main() {
    const int M = 512, N = 512, K = 512;
    size_t bytesA = M * K * sizeof(half);
    size_t bytesB = K * N * sizeof(half);
    size_t bytesC = M * N * sizeof(float);

    printf("=== TMP: TMA + MMA overlap demo (sm_120) ===\n");
    printf("M=%d N=%d K=%d, tile=%dx%dx%d, stages=%d\n", M, N, K, TILE_M, TILE_N, TILE_K, NUM_STAGES);

    half *h_A = (half*)malloc(bytesA), *h_B = (half*)malloc(bytesB);
    for (int i = 0; i < M*K; ++i) h_A[i] = __float2half(0.01f * (rand()%100));
    for (int i = 0; i < K*N; ++i) h_B[i] = __float2half(0.01f * (rand()%100));

    half *d_A, *d_B; float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    int smem_size = NUM_STAGES * (A_TILE_BYTES + B_TILE_BYTES);
    printf("smem per block = %d bytes\n", smem_size);

    for (int i = 0; i < 3; ++i) {
        gemm_pipeline<<<grid, block, smem_size>>>(d_A, d_B, d_C, M, N, K);
        gemm_naive<<<grid, block, 0>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    {
        nvtx3::scoped_range r_pipe("gemm_pipeline");
        gemm_pipeline<<<grid, block, smem_size>>>(d_A, d_B, d_C, M, N, K);
    }
    {
        nvtx3::scoped_range r_naive("gemm_naive");
        gemm_naive<<<grid, block, 0>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    printf("Done. (tmp demo, delete after reading)\n");
    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
