
● 完美的数据！我们可以清楚地看到 kernel_memory_dep 中哪些指令因为 Long Scoreboard（等待内存加载结果）而 stall。
现在我已经收集了足够的数据。让我整理成清晰的讲解。

实验结果与讲解
实验代码在 /home/leo/stall_demo/stall_demo.cu，NCU 报告在 /home/leo/stall_demo/stall_report.ncu-rep（可用 Windows 上的 Nsight Compute GUI 打开查看更直观的图表）。

一、4个 Kernel 的 Warp Stall 对比（单位：cycles/指令）
┌────────────────────┬─────────────────┬───────────────────┬─────────────────┬────────────────┐
 │     Stall 原因     │ kernel_math_dep │ kernel_memory_dep │ kernel_no_stall │ kernel_barrier │
 ├────────────────────┼─────────────────┼───────────────────┼─────────────────┼────────────────┤
 │ Long Scoreboard    │ 0               │ 121.18            │ 1.76            │ 2.21           │
 ├────────────────────┼─────────────────┼───────────────────┼─────────────────┼────────────────┤
 │ LG Throttle        │ 0               │ 46.64             │ 0               │ 0              │
 ├────────────────────┼─────────────────┼───────────────────┼─────────────────┼────────────────┤
 │ Short Scoreboard   │ 0.03            │ 1.64              │ 0.11            │ 9.08           │
 ├────────────────────┼─────────────────┼───────────────────┼─────────────────┼────────────────┤
 │ Math Pipe Throttle │ 1.75            │ 0.01              │ 0.15            │ 0              │
 ├────────────────────┼─────────────────┼───────────────────┼─────────────────┼────────────────┤
 │ MIO Throttle       │ 0.05            │ 1.41              │ 0.06            │ 18.21          │
 ├────────────────────┼─────────────────┼───────────────────┼─────────────────┼────────────────┤
 │ Barrier            │ 0               │ 0                 │ 0               │ 11.29          │
 ├────────────────────┼─────────────────┼───────────────────┼─────────────────┼────────────────┤
 │ Wait               │ 2.92            │ 3.81              │ 0.29            │ 2.52           │
 ├────────────────────┼─────────────────┼───────────────────┼─────────────────┼────────────────┤
 │ Not Selected       │ 5.17            │ 0.20              │ 8.28            │ 2.30           │
 └────────────────────┴─────────────────┴───────────────────┴─────────────────┴────────────────┘
二、各 Stall 原因解释
1.Long Scoreboard — 等待长延迟操作（ld/st/tex）的结果
这就是你问的核心问题。kernel_memory_dep 的主导 stall 原因是 Long Scoreboard = 121 cycles。
机制：GPU 有两个 scoreboard：
●Short Scoreboard：追踪短延迟操作（算术，通常 4-6 cycles 固定延迟）
●Long Scoreboard：追踪长延迟global/不确定延迟操作（global load, shared load, texture fetch）
当一条指令（比如 FADD R6, R9, R14）需要读取寄存器 R14，但 R14 的值还在从 global memory 往回搬（上面的 LDG 还没返回），warp 就会被标记为
 stall_long_scoreboard，直到数据到达。从源码级 stall 采样数据看：
 IABS R8, R12           ← 46515 次被 Long Scoreboard stall
 （R12 = 上一条 LDG 的目的寄存器，数据还没回来）
 FADD R6, R9, R14       ← 28190 次 Long Scoreboard stall
 （R14 = LDG 结果，pointer chasing 要等）
 IMAD.WIDE R12, R15, 0x4, R4  ← 21490 次
2. （R15 来自对 indices[] 的 load）你的问题：ld/st 延迟不确定，scoreboard 怎么处理？
答案：Scoreboard 不需要知道延迟是多少。它的工作方式是：
发射 LDG R14, [addr]  →  在 Long Scoreboard 标记 “R14 pending”
 …
 (warp scheduler 去调度其他 eligible warp)
 …
 当 memory 子系统把数据写回 register file → 清除 R14 的 pending 标记
 下一条需要 R14 的指令现在可以 issue 了关键点：
●Scoreboard 不预测延迟，它只做 “ready/not-ready” 判定
●一条 LDG 可能 20 cycles（L1 hit）或 200-600 cycles（DRAM miss），scoreboard 不关心具体多久
●只有当数据确实到了（memory 系统通知 scoreboard），对应的 pending bit 才被清除
●在等待期间，warp scheduler 不会选这个 warp（改选其他 ready 的 warp）
3.LG Throttle — LSU 端口/队列满
kernel_memory_dep 还有 46.64 cycles 的 LG Throttle。这正是你说的 “port 满了” 的情况：
●GPU 的 Load/Store Unit (LSU) 有有限的 outstanding request 队列
●当 pointer chasing 造成大量 cache miss，这些 load 请求积压在 LSU 队列
●新的 load 指令无法发射 → stall_lg_throttle
4.Math Pipe Throttle — 执行单元满
kernel_math_dep 虽然只有 1.75 cycles 的 math_pipe_throttle，但这是因为 RTX 5050 的 FMA 单元吞吐非常高。有趣的是，它的主导 stall 是 “Not Selected”（5.17），说明 warp
5. 其实是 ready 的（不缺依赖），只是 scheduler 每次只能选一个 warp 发射，轮不到它。Barrier — __syncthreads() 等待
kernel_barrier 的 11.29 cycles barrier stall 很直观：一个 block 里有 256 个线程（8个 warp），到了 __syncthreads() 先到的 warp 必须等后到的。
6.MIO Throttle — shared memory 管道满
kernel_barrier 的主导竟然是 MIO Throttle（18.21），因为它密集使用 shared memory，MIO 管道（处理 shared memory 的通路）饱和了。

三、总结：Scoreboard 工作原理

                  ┌───────────────────────────┐
                  │     Warp Scheduler        │
                  │  (每个 sub-partition 1个)   │
                  └─────────┬─────────────────┘
                            │ 选择一个 eligible warp
                            ▼
            ┌─────────────────────────────────┐
            │  Scoreboard Check               │
            │                                 │
            │  Short SB: 寄存器被短延迟指令占用?  │  ← 算术(FMA等) ~4-6 cycle
            │  Long SB:  寄存器被长延迟指令占用?  │  ← memory ~20-600 cycle
            │                                 │
            │  任何一个 pending → warp NOT eligible │
            └─────────────────────────────────┘
                            │
                  全部 clear → 可以 issue
                            ▼
            ┌──────────────────────────────┐
            │  Dispatch to execution pipe   │
            └──────────────────────────────┘

不确定延迟的处理：
●Scoreboard 是 事件驱动的，不是计时器
●LDG 发射时设置 pending bit
●Memory 系统返回数据时发送 write-back signal 到 scoreboard，清除 bit
●所以无论实际延迟是 20 还是 500 cycles，机制完全一样
●GPU 靠 大量 warp 并行 来隐藏这些不确定延迟（latency hiding）

四、你可以继续做的实验
1.打开 GUI 看更直观的图：
Windows 上打开 Nsight Compute GUI, 加载:
\wsl$\Ubuntu\home\leo\stall_demo\stall_report.ncu-rep
1.GUI 里 “Source” tab 可以把 stall 原因叠加到源码/汇编行上，非常直观。
2.用 --section WarpStateStats 单独看某个 kernel 的 stall 分布（如上实验已做）
3.Nsys vs NCU 的区别：
 - Nsys (Nsight Systems): timeline 级别，看 kernel 何时运行、host-device 同步等宏观行为
 - NCU (Nsight Compute): 单个 kernel 内部微架构级别，看每条指令的 stall 原因（就是我们这次做的）


源码：
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

    printf("\n=== All kernels completed ===\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    cudaFree(d_out); cudaFree(d_indices);
    free(h_data); free(h_indices);
    return 0;
}



