#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// ============================================================
// Kernel 1: Memory-Bound — 向量加法 (极低算术强度)
//   算术强度 ≈ 1 FLOP / 12 bytes = 0.083 FLOP/byte
//   预期: DRAM throughput 高, SM throughput 低
// ============================================================
__global__ void kernel_memory_bound(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];   // 1 条 FMA, 3 次 DRAM 访问
    }
}

// ============================================================
// Kernel 2: Compute-Bound — 寄存器内密集计算 (高算术强度)
//   每个线程 load 2 个 float, 然后做 1024 次 FMA, 最后 store 1 个 float
//   算术强度 ≈ 2048 FLOP / 12 bytes ≈ 170 FLOP/byte
//   预期: SM throughput 高, DRAM throughput 低
// ============================================================
__global__ void kernel_compute_bound(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];           // 1 次 load
        float b = B[idx];           // 1 次 load
        float acc = 0.0f;

        // 寄存器内反复做 FMA, 无额外访存
        #pragma unroll
        for (int i = 0; i < 1024; ++i) {
            acc = fmaf(a, b, acc);  // 2 FLOP/iter, 全部在寄存器里
            a   = acc * 0.999f;
        }

        C[idx] = acc;               // 1 次 store
    }
}

// ============================================================
// Kernel 3: Latency-Bound — 跨步访问 (coalescing 差)
//   warp 内线程访问 stride * sizeof(float) 的地址,
//   每次 load 只用到 sector 的一小部分, 大量 bandwidth 浪费
//   预期: DRAM throughput 低 AND SM throughput 低
// ============================================================
__global__ void kernel_latency_bound(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N,
                                      int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// ============================================================
// 工具函数
// ============================================================
#define CUDA_CHECK(call) do {                                   \
    cudaError_t e = (call);                                      \
    if (e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(e));      \
        exit(1);                                                 \
    }                                                           \
} while(0)

void init_array(float* h_data, int N) {
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

// 验证 kernel 2 的数值正确性 (用 CPU 算一遍)
float cpu_compute_bound(float a, float b) {
    float acc = 0.0f;
    for (int i = 0; i < 1024; ++i) {
        acc = fmaf(a, b, acc);
        a   = acc * 0.999f;
    }
    return acc;
}

int main() {
    const int N = 32 * 1024 * 1024;  // 32M 元素 = 128 MB per array
    const int blockSize = 256;
    const int gridSize  = (N + blockSize - 1) / blockSize;

    printf("=== Roofline Demo Kernels ===\n");
    printf("GPU: RTX 5050 (Blackwell, sm_120)\n");
    printf("N = %d elements (%.1f MB per array)\n", N, N * sizeof(float) / 1e6f);
    printf("Block size = %d, Grid size = %d\n\n", blockSize, gridSize);

    // 分配 & 初始化
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    init_array(h_A, N);
    init_array(h_B, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // ---------- Warmup (跑三遍, 避免冷启动干扰 profile) ----------
    printf("Warming up...\n");
    for (int i = 0; i < 3; ++i) {
        kernel_memory_bound<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();

    // ---------- Kernel 1: Memory-Bound ----------
    printf("\n[1/3] Launching kernel_memory_bound...\n");
    kernel_memory_bound<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // ---------- Kernel 2: Compute-Bound ----------
    printf("[2/3] Launching kernel_compute_bound...\n");
    kernel_compute_bound<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // ---------- Kernel 3: Latency-Bound ----------
    printf("[3/3] Launching kernel_latency_bound (stride=16)...\n");
    int stride_gridSize = (N + blockSize * 16 - 1) / (blockSize * 16);
    kernel_latency_bound<<<stride_gridSize, blockSize>>>(d_A, d_B, d_C, N, 16);
    cudaDeviceSynchronize();

    // ---------- 验证 kernel 2 结果 ----------
    float* h_C = (float*)malloc(N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 只验证第一个元素 (kernel 2 最后跑的会覆盖)
    // 重新跑一次 kernel 2 到单独的 buffer 做验证
    float *d_C2;
    CUDA_CHECK(cudaMalloc(&d_C2, N * sizeof(float)));
    kernel_compute_bound<<<gridSize, blockSize>>>(d_A, d_B, d_C2, N);
    CUDA_CHECK(cudaMemcpy(h_C, d_C2, N * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = cpu_compute_bound(h_A[0], h_B[0]);
    printf("\nValidation (kernel_compute_bound, element 0):\n");
    printf("  GPU result = %f\n", h_C[0]);
    printf("  CPU result = %f\n", expected);
    printf("  Diff       = %e\n", fabs(h_C[0] - expected));

    // 清理
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C2);

    printf("\nDone. Profile me with:\n");
    printf("  sudo /usr/local/cuda-13.2/bin/ncu --set full --launch-skip 3 --launch-count 3 -f -o roofline_demo %s\n",
           __FILE__);
    printf("  sudo /usr/local/cuda-13.2/bin/ncu -i roofline_demo.ncu-rep --print-summary per-kernel\n");

    return 0;
}
