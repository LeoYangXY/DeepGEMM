/**
 * stall_demo.cu - 每个 warp stall reason 的"真实跑出来"示例 (sm_120 / Blackwell)
 *
 * 重要：每个 kernel 都已在 RTX 5050 Laptop 上用 ncu 实测，文档里贴的 stall 值
 * 是真实跑出来的。Blackwell 把很多"等依赖/等内存"归并到 wait，所以部分 stall
 * reason 在本机用纯 CUDA C++ 无法单独触发（见 stall_demo_walkthrough.md 说明）。
 *
 * 编译: nvcc -O3 -lineinfo -arch=sm_120 -o stall_demo stall_demo.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda/barrier>

// 1) short_scoreboard: 严格串行算术依赖链（每步依赖上一步结果）
__global__ void kernel_short_scoreboard(float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = (float)idx;
    for (int i = 0; i < 200; i++) {
        val = val * 1.0001f + 0.0007f;
        val = val * val + 0.5f;
        val = sqrtf(val + 1.0f);
        out[idx] = val;   // 全局写，防止整段被优化
    }
    out[idx] = val;
}

// 2) lg_throttle: pointer chasing，长延迟 global load 占满 LG 单元
__global__ void kernel_lg_throttle(float *data, int *indices, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float sum = 0.0f;
    int ptr = idx;
    for (int i = 0; i < 80; i++) {
        ptr = indices[ptr % N];
        sum += data[ptr % N];
    }
    out[idx] = sum;
}

// 3) selected (对照): 高 ILP 独立链，warp 几乎不 stall
__global__ void kernel_selected(float *a, float *b, float *c, float *d,
                                float *e, float *f, float *g, float *h,
                                float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v0=a[idx], v1=b[idx], v2=c[idx], v3=d[idx];
    float v4=e[idx], v5=f[idx], v6=g[idx], v7=h[idx];
    for (int i = 0; i < 100; i++) {
        v0=v0*1.01f+0.1f; v1=v1*1.02f+0.1f; v2=v2*1.03f+0.1f; v3=v3*1.04f+0.1f;
        v4=v4*1.05f+0.1f; v5=v5*1.06f+0.1f; v6=v6*1.07f+0.1f; v7=v7*1.08f+0.1f;
    }
    out[idx]=v0+v1+v2+v3+v4+v5+v6+v7;
}

// 4) barrier: __syncthreads 频繁块内同步
__global__ void kernel_barrier(float *data, float *out, int N) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx < N) smem[tid] = data[idx];
    __syncthreads();
    for (int i = 0; i < 60; i++) {
        if (idx < N) smem[tid] = smem[tid]*0.99f + smem[(tid+1)%256]*0.01f;
        __syncthreads();
    }
    if (idx < N) out[idx] = smem[tid];
}

// 5) math_pipe_throttle: 密集独立 FMA（Blackwell FP32 极强，可能偶发）
__global__ void kernel_math_pipe_throttle(float *a, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = a[idx];
    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    for (int i = 0; i < 300; i++) {
        s0=s0+x*1.00001f; s1=s1+x*1.00002f; s2=s2+x*1.00003f; s3=s3+x*1.00004f;
        s4=s4+x*1.00005f; s5=s5+x*1.00006f; s6=s6+x*1.00007f; s7=s7+x*1.00008f;
        s0=s0*x*2.0f;     s1=s1*x*2.0f;     s2=s2+x*2.0f;     s3=s3+x*2.0f;
        s4=s4+x*2.0f;     s5=s5+x*2.0f;     s6=s6+x*2.0f;     s7=s7+x*2.0f;
    }
    out[idx]=s0+s1+s2+s3+s4+s5+s6+s7;
}

// 6) membar: 频繁 __threadfence_block
__global__ void kernel_membar(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    for (int i = 0; i < 120; i++) {
        v = v*1.01f + 0.1f;
        __threadfence_block();
    }
    out[idx] = v;
}

// 7) sleeping: device 侧主动睡眠（用 PTX nanosleep 指令强制，防 -O3 优化）
__global__ void kernel_sleeping(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    unsigned int acc = 0;
    for (int i = 0; i < 60; i++) {
        v = v*1.01f + 0.1f;
        unsigned int ns = 1000u + (unsigned int)(v*7.0f);
        // PTX: nanosleep 指令，强制 warp 睡眠
        asm volatile("nanosleep.u32 %0;" : "+r"(ns) : : "memory");
        acc += ns;
    }
    out[idx] = v + (float)acc;
}

// 8) branch_resolving: 数据相关分支，条件依赖上一条结果
__global__ void kernel_branch(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    float acc = 0.0f;
    for (int i = 0; i < 200; i++) {
        v = v*1.01f + 0.1f;
        if (v > 1.0f) acc += v; else acc -= v*0.5f;
        float t = v + acc*0.001f;
        if (t < 0.0f) acc = 0.0f; else acc += t;
        out[idx] = acc;
    }
    out[idx] = acc;
}

// 9) mio_throttle: 密集 shared memory 跨 bank 访问
__global__ void kernel_mio_throttle(float *data, float *out, int N) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx < N) smem[tid] = data[idx];
    __syncthreads();
    float acc = 0.0f;
    for (int i = 0; i < 200; i++) {
        acc += smem[(tid*7+1)%256];
        acc += smem[(tid*13+3)%256];
        acc += smem[(tid*5+9)%256];
        acc += smem[(tid*11+17)%256];
        acc += smem[(tid*17+5)%256];
    }
    if (idx < N) out[idx] = acc;
}

// 10) wait: cuda::barrier arrive_and_wait
__global__ void kernel_wait(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) init(&bar, blockDim.x);
    __syncthreads();
    float v = data[idx];
    float acc = 0.0f;
    for (int i = 0; i < 100; i++) {
        v = v*1.01f + 0.1f;
        bar.arrive_and_wait();
        acc += v;
    }
    out[idx] = acc;
}

// 11) dispatch_stall: 密集 SFU (sin/rcp/cos/sqrt)
__global__ void kernel_dispatch_stall(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    float acc = 0.0f;
    for (int i = 0; i < 200; i++) {
        float s = sinf(v + (float)i);
        float r = __frcp_rn(v + s + 1.0f);
        float c = cosf(v - r);
        float t = __fsqrt_rn(c*c + 1.0f);
        acc += s + r + c + t;
        v = v*1.001f + 0.1f;
        out[idx] = acc;
    }
    out[idx] = acc;
}

// 12) no_instruction: 单线程 block，其余 lane 立即退出
__global__ void kernel_no_instruction(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x != 0) return;
    if (idx >= N) return;
    float v = data[idx];
    float acc = 0.0f;
    for (int i = 0; i < 80; i++) {
        acc += v*1.01f;
        v = v + 0.1f;
    }
    out[idx] = acc;
}

// 13) drain: 尾部一次性发起大量长延迟 global load
__global__ void kernel_drain(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    float acc = 0.0f;
    for (int i = 0; i < 20; i++) { v=v*1.01f+0.1f; acc += v; }
    float sum = 0.0f;
    for (int i = 0; i < 96; i++) {
        sum += data[(idx*31 + i*131) % N];
    }
    out[idx] = acc + sum;
}

// 14) long_scoreboard: 确定性依赖 load（本机归 lg_throttle，作对照）
__global__ void kernel_long_scoreboard(float *data, int *nxt, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float sum = 0.0f;
    int p = idx;
    for (int i = 0; i < 80; i++) {
        float x = data[p];
        p = nxt[p % N];
        sum += x + (float)p;
    }
    out[idx] = sum;
}

void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    const int N = 1 << 20;
    const int bytes = N * sizeof(float);
    const int ibytes = N * sizeof(int);
    float *d_a,*d_b,*d_c,*d_d,*d_e,*d_f,*d_g,*d_h,*d_out;
    int   *d_idx,*d_nxt;
    check_cuda(cudaMalloc(&d_a,bytes),"a"); check_cuda(cudaMalloc(&d_b,bytes),"b");
    check_cuda(cudaMalloc(&d_c,bytes),"c"); check_cuda(cudaMalloc(&d_d,bytes),"d");
    check_cuda(cudaMalloc(&d_e,bytes),"e"); check_cuda(cudaMalloc(&d_f,bytes),"f");
    check_cuda(cudaMalloc(&d_g,bytes),"g"); check_cuda(cudaMalloc(&d_h,bytes),"h");
    check_cuda(cudaMalloc(&d_out,bytes),"out");
    check_cuda(cudaMalloc(&d_idx,ibytes),"idx");
    check_cuda(cudaMalloc(&d_nxt,ibytes),"nxt");
    float *h = (float*)malloc(bytes);
    int   *hi = (int*)malloc(ibytes);
    for (int i=0;i<N;i++){ h[i]=(float)(i%100)*0.01f; hi[i]=(i*7+13)%N; }
    check_cuda(cudaMemcpy(d_a,h,bytes,cudaMemcpyHostToDevice),"ca");
    check_cuda(cudaMemcpy(d_b,h,bytes,cudaMemcpyHostToDevice),"cb");
    check_cuda(cudaMemcpy(d_c,h,bytes,cudaMemcpyHostToDevice),"cc");
    check_cuda(cudaMemcpy(d_d,h,bytes,cudaMemcpyHostToDevice),"cd");
    check_cuda(cudaMemcpy(d_e,h,bytes,cudaMemcpyHostToDevice),"ce");
    check_cuda(cudaMemcpy(d_f,h,bytes,cudaMemcpyHostToDevice),"cf");
    check_cuda(cudaMemcpy(d_g,h,bytes,cudaMemcpyHostToDevice),"cg");
    check_cuda(cudaMemcpy(d_h,h,bytes,cudaMemcpyHostToDevice),"ch");
    check_cuda(cudaMemcpy(d_idx,hi,ibytes,cudaMemcpyHostToDevice),"ci");
    check_cuda(cudaMemcpy(d_nxt,hi,ibytes,cudaMemcpyHostToDevice),"cn");

    int threads=256, blocks=(N+threads-1)/threads;
    printf("=== stall_demo (sm_120) ===\nN=%d blocks=%d threads=%d\n\n",N,blocks,threads);
    printf("[1] short_scoreboard\n");  kernel_short_scoreboard<<<blocks,threads>>>(d_out,N); check_cuda(cudaDeviceSynchronize(),"1");
    printf("[2] lg_throttle\n");       kernel_lg_throttle<<<blocks,threads>>>(d_a,d_idx,d_out,N); check_cuda(cudaDeviceSynchronize(),"2");
    printf("[3] selected\n");          kernel_selected<<<blocks,threads>>>(d_a,d_b,d_c,d_d,d_e,d_f,d_g,d_h,d_out,N); check_cuda(cudaDeviceSynchronize(),"3");
    printf("[4] barrier\n");           kernel_barrier<<<blocks,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"4");
    printf("[5] math_pipe_throttle\n");kernel_math_pipe_throttle<<<blocks,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"5");
    printf("[6] membar\n");            kernel_membar<<<blocks,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"6");
    printf("[7] sleeping\n");          kernel_sleeping<<<blocks,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"7");
    printf("[8] branch\n");            kernel_branch<<<blocks,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"8");
    printf("[9] mio_throttle\n");      kernel_mio_throttle<<<blocks,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"9");
    printf("[10] wait\n");             kernel_wait<<<blocks,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"10");
    printf("[11] dispatch_stall\n");   kernel_dispatch_stall<<<blocks,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"11");
    printf("[12] no_instruction\n");   kernel_no_instruction<<<1,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"12");
    printf("[13] drain\n");            kernel_drain<<<blocks,threads>>>(d_a,d_out,N); check_cuda(cudaDeviceSynchronize(),"13");
    printf("[14] long_scoreboard\n");  kernel_long_scoreboard<<<blocks,threads>>>(d_a,d_idx,d_out,N); check_cuda(cudaDeviceSynchronize(),"14");
    printf("\n=== done ===\n");
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);cudaFree(d_d);cudaFree(d_e);
    cudaFree(d_f);cudaFree(d_g);cudaFree(d_h);cudaFree(d_out);cudaFree(d_idx);cudaFree(d_nxt);
    free(h);free(hi);
    return 0;
}
