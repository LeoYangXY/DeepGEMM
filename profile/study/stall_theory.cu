/**
 * stall_theory.cu - 验证 "scoreboard vs throttle" 理论的针对性实验 (sm_120)
 *
 * 编译: nvcc -O3 -lineinfo -arch=sm_120 -o stall_theory stall_theory.cu
 *
 * 实验设计:
 *  A) SFU 串行依赖链   -> 理论: short_scoreboard (等 SFU 超越函数)
 *  B) add/mul 串行链   -> 理论: 普通计算延迟低, 不应产生 short_scoreboard
 *  C) smem 顺序访问    -> 理论: 无 bank conflict, MIO 压力小
 *  D) smem bank conflict-> 理论: 多端口争用, MIO throttle / 冲突事件飙升
 */

#include <stdio.h>
#include <cuda_runtime.h>

// A) SFU 串行依赖: 用 PTX 强制走 SFU 指令 (sin/rcp/cos/sqrt.approx)
//    注意: 纯 sinf() 在 -O3 会被展开成 FMA 多项式, 不走 SFU; 必须用 PTX 强制
__global__ void kernel_sfu_dep(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    for (int i = 0; i < 200; i++) {
        asm volatile("sin.approx.f32 %0, %0;" : "+f"(v));   // SFU
        asm volatile("add.f32 %0, %0, 1.0;" : "+f"(v));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(v));    // SFU
        asm volatile("cos.approx.f32 %0, %0;" : "+f"(v));    // SFU
        asm volatile("sqrt.approx.f32 %0, %0;" : "+f"(v));   // SFU
        out[idx] = v;                  // 防优化
    }
    out[idx] = v;
}

// B) add/mul 串行依赖: 每步依赖上一步, 但全是非 SFU 的简单 FP 运算
__global__ void kernel_math_dep(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    for (int i = 0; i < 200; i++) {
        v = v * 1.0001f + 0.0007f;    // FMA, 非 SFU
        v = v * v + 0.5f;             // FMA
        v = v * 1.5f - 0.25f;         // FMA
        out[idx] = v;
    }
    out[idx] = v;
}

// C) smem 顺序访问 (无 bank conflict): 每线程访问自己那一行 -> 广播/连续
__global__ void kernel_smem_seq(float *data, float *out, int N) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx < N) smem[tid] = data[idx];
    __syncthreads();
    float acc = 0.0f;
    for (int i = 0; i < 200; i++) {
        acc += smem[tid];             // 每线程只访问自己的 bank -> 无冲突
        acc += smem[(tid+1)%256];
    }
    if (idx < N) out[idx] = acc;
}

// D) smem bank conflict: 所有线程访问同一 bank 的同一偏移 -> 严重冲突
__global__ void kernel_smem_conflict(float *data, float *out, int N) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx < N) smem[tid] = data[idx];
    __syncthreads();
    float acc = 0.0f;
    for (int i = 0; i < 200; i++) {
        // 所有线程都访问 smem[tid % 32] 的同类偏移 -> 同 bank 多端口争用
        acc += smem[tid % 32];        // bank conflict
        acc += smem[(tid % 32) + 32]; // 又一条冲突访问
        acc += smem[(tid % 32) + 64];
        acc += smem[(tid % 32) + 96];
        acc += smem[(tid % 32) + 128];
        acc += smem[(tid % 32) + 160];
        acc += smem[(tid % 32) + 192];
        acc += smem[(tid % 32) + 224];
    }
    if (idx < N) out[idx] = acc;
}

void check(cudaError_t e, const char *m) {
    if (e != cudaSuccess) { fprintf(stderr, "ERR %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

int main() {
    const int N = 1 << 20;
    const int bytes = N * sizeof(float);
    float *d_a, *d_out;
    check(cudaMalloc(&d_a, bytes), "a");
    check(cudaMalloc(&d_out, bytes), "o");
    float *h = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h[i] = (float)(i % 97) * 0.01f + 1.0f;
    check(cudaMemcpy(d_a, h, bytes, cudaMemcpyHostToDevice), "ca");

    int t = 256, b = (N + t - 1) / t;
    printf("=== stall_theory (sm_120) ===\n");
    printf("[A] sfu_dep\n");      kernel_sfu_dep<<<b,t>>>(d_a, d_out, N);      check(cudaDeviceSynchronize(), "A");
    printf("[B] math_dep\n");     kernel_math_dep<<<b,t>>>(d_a, d_out, N);     check(cudaDeviceSynchronize(), "B");
    printf("[C] smem_seq\n");     kernel_smem_seq<<<b,t>>>(d_a, d_out, N);     check(cudaDeviceSynchronize(), "C");
    printf("[D] smem_conflict\n");kernel_smem_conflict<<<b,t>>>(d_a, d_out, N);check(cudaDeviceSynchronize(), "D");
    printf("=== done ===\n");
    cudaFree(d_a); cudaFree(d_out); free(h);
    return 0;
}
