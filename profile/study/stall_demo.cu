/**
 * stall_demo.cu - 演示不同的 warp stall 原因
 *
 * 实验目标：
 * 1. kernel_math_dep    - 数据依赖 stall (指令间依赖, scoreboard等待)
 * 2. kernel_memory_dep  - 内存延迟 stall (long scoreboard, ld/st 不确定延迟)
 * 3. kernel_no_stall    - ILP充分, 几乎无stall
 * 4. kernel_barrier     - barrier同步导致的stall
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel 1: 纯算术依赖链 - 每条指令都依赖前一条的结果
// 预期stall原因: short scoreboard (等待前一条算术指令完成)
__global__ void kernel_math_dep(float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = (float)idx;
    // 长依赖链: 每一步都必须等前一步完成
    for (int i = 0; i < 100; i++) {
        val = val * 1.01f + 0.5f;   // 依赖 val
        val = val * 0.99f - 0.3f;   // 依赖 val
        val = sqrtf(val * val + 1.0f); // 依赖 val
        val = val * 1.02f + 0.1f;   // 依赖 val
    }
    out[idx] = val;
}

// Kernel 2: 随机内存访问 - 大量cache miss, 长延迟load
// 预期stall原因: long scoreboard / memory dependency (等待global memory load)
__global__ void kernel_memory_dep(float *data, int *indices, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float sum = 0.0f;
    int ptr = idx;
    // pointer chasing: 每次load都依赖上一次load的结果
    for (int i = 0; i < 50; i++) {
        ptr = indices[ptr % N];       // load -> 得到下一个地址
        sum += data[ptr % N];         // 依赖上面的load结果
    }
    out[idx] = sum;
}

// Kernel 3: 高ILP, 独立操作多 - 流水线可以隐藏延迟
// 预期: stall很少, 因为有足够的独立指令填满流水线
__global__ void kernel_no_stall(float *a, float *b, float *c, float *d, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // 4条独立的计算链, 互不依赖
    float v0 = a[idx], v1 = b[idx], v2 = c[idx], v3 = d[idx];
    for (int i = 0; i < 100; i++) {
        v0 = v0 * 1.01f + 0.1f;
        v1 = v1 * 0.99f + 0.2f;
        v2 = v2 * 1.02f + 0.3f;
        v3 = v3 * 0.98f + 0.4f;
    }
    out[idx] = v0 + v1 + v2 + v3;
}

// Kernel 4: __syncthreads() barrier导致的stall
// 预期stall原因: barrier (等待同block其他warp到达barrier)
__global__ void kernel_barrier(float *data, float *out, int N) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < N) smem[tid] = data[idx];
    __syncthreads();

    // 多次barrier同步
    for (int i = 0; i < 50; i++) {
        if (idx < N) {
            smem[tid] = smem[tid] * 0.99f + smem[(tid + 1) % 256] * 0.01f;
        }
        __syncthreads();  // 每次迭代都要同步
    }

    if (idx < N) out[idx] = smem[tid];
}

// Kernel 5: 大量独立 FMA 让 FP32 math pipe 饱和
// 预期 stall 原因: math_pipe_throttle (math 管道忙, 无足够独立指令可发射)
__global__ void kernel_math_pipe_throttle(float *a, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = a[idx];
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int i = 0; i < 200; i++) {
        // 多条独立 FMA 持续压 math pipe
        s0 = s0 + x * 1.0001f;
        s1 = s1 + x * 1.0002f;
        s2 = s2 + x * 1.0003f;
        s3 = s3 + x * 1.0004f;
        s0 = s0 + x * 2.0001f;
        s1 = s1 + x * 2.0002f;
        s2 = s2 + x * 2.0003f;
        s3 = s3 + x * 2.0004f;
    }
    out[idx] = s0 + s1 + s2 + s3;
}

// Kernel 6: __threadfence_block() 频繁块内 fence
// 预期 stall 原因: membar (等待 memory barrier 完成)
__global__ void kernel_membar(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    for (int i = 0; i < 100; i++) {
        v = v * 1.01f + 0.1f;
        __threadfence_block();   // 块内内存屏障 -> membar stall
    }
    out[idx] = v;
}

// Kernel 7: 密集分散 global store -> LG (load/store 单元) 队列节流
// 预期 stall 原因: lg_throttle (store 队列满, 等 LG 单元空闲)
__global__ void kernel_lg_throttle(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    for (int i = 0; i < 100; i++) {
        v = v * 1.01f + 0.1f;
        out[(idx * 31 + i * 131) % N] = v;  // 分散 global store, 压 LG 单元
    }
}

// Kernel 8: device-side __nanosleep 睡眠
// 预期 stall 原因: sleeping (warp 主动睡眠等待)
__global__ void kernel_sleep(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    for (int i = 0; i < 100; i++) {
        v = v * 1.01f + 0.1f;
        __nanosleep(1000);   // device 侧睡眠 -> sleeping stall
    }
    out[idx] = v;
}

void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const int bytes = N * sizeof(float);
    const int ibytes = N * sizeof(int);

    // Allocate
    float *d_a, *d_b, *d_c, *d_d, *d_out;
    int *d_indices;
    check_cuda(cudaMalloc(&d_a, bytes), "alloc a");
    check_cuda(cudaMalloc(&d_b, bytes), "alloc b");
    check_cuda(cudaMalloc(&d_c, bytes), "alloc c");
    check_cuda(cudaMalloc(&d_d, bytes), "alloc d");
    check_cuda(cudaMalloc(&d_out, bytes), "alloc out");
    check_cuda(cudaMalloc(&d_indices, ibytes), "alloc indices");

    // 初始化
    float *h_data = (float*)malloc(bytes);
    int *h_indices = (int*)malloc(ibytes);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i % 100) * 0.01f;
        h_indices[i] = (i * 7 + 13) % N;  // 伪随机跳转
    }
    check_cuda(cudaMemcpy(d_a, h_data, bytes, cudaMemcpyHostToDevice), "cpy a");
    check_cuda(cudaMemcpy(d_b, h_data, bytes, cudaMemcpyHostToDevice), "cpy b");
    check_cuda(cudaMemcpy(d_c, h_data, bytes, cudaMemcpyHostToDevice), "cpy c");
    check_cuda(cudaMemcpy(d_d, h_data, bytes, cudaMemcpyHostToDevice), "cpy d");
    check_cuda(cudaMemcpy(d_indices, h_indices, ibytes, cudaMemcpyHostToDevice), "cpy idx");

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("=== Running stall demo kernels ===\n");
    printf("N=%d, blocks=%d, threads=%d\n\n", N, blocks, threads);

    printf("[1] kernel_math_dep - arithmetic dependency chain\n");
    kernel_math_dep<<<blocks, threads>>>(d_out, N);
    check_cuda(cudaDeviceSynchronize(), "sync math_dep");

    printf("[2] kernel_memory_dep - pointer chasing (random global loads)\n");
    kernel_memory_dep<<<blocks, threads>>>(d_a, d_indices, d_out, N);
    check_cuda(cudaDeviceSynchronize(), "sync memory_dep");

    printf("[3] kernel_no_stall - high ILP, independent chains\n");
    kernel_no_stall<<<blocks, threads>>>(d_a, d_b, d_c, d_d, d_out, N);
    check_cuda(cudaDeviceSynchronize(), "sync no_stall");

    printf("[4] kernel_barrier - __syncthreads heavy\n");
    kernel_barrier<<<blocks, threads>>>(d_a, d_out, N);
    check_cuda(cudaDeviceSynchronize(), "sync barrier");

    printf("[5] kernel_math_pipe_throttle - FP32 math pipe saturated\n");
    kernel_math_pipe_throttle<<<blocks, threads>>>(d_a, d_out, N);
    check_cuda(cudaDeviceSynchronize(), "sync math_pipe_throttle");

    printf("[6] kernel_membar - __threadfence_block heavy\n");
    kernel_membar<<<blocks, threads>>>(d_a, d_out, N);
    check_cuda(cudaDeviceSynchronize(), "sync membar");

    printf("[7] kernel_lg_throttle - dense scattered global stores\n");
    kernel_lg_throttle<<<blocks, threads>>>(d_a, d_out, N);
    check_cuda(cudaDeviceSynchronize(), "sync lg_throttle");

    printf("[8] kernel_sleep - __nanosleep device-side\n");
    kernel_sleep<<<blocks, threads>>>(d_a, d_out, N);
    check_cuda(cudaDeviceSynchronize(), "sync sleep");

    printf("\n=== All kernels completed ===\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    cudaFree(d_out); cudaFree(d_indices);
    free(h_data); free(h_indices);
    return 0;
}
