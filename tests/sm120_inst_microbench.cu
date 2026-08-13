// SM120 instruction microbench for hand-scheduling notes.
//
//   /usr/local/cuda-13.2/bin/nvcc -O3 -std=c++20 \
//        -gencode arch=compute_120a,code=sm_120a \
//        tests/sm120_inst_microbench.cu -o /tmp/sm120_inst_microbench
//
// Latency = dependent chain + %clock.  Throughput = 8 independent accums.
// QMMA.SF uses identity ue8m0=127 (same math as unscaled, full-rate SASS).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

__device__ __forceinline__ uint32_t clock32() {
    uint32_t t;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t) :: "memory");
    return t;
}

__device__ __forceinline__ void bar_sync() {
    asm volatile("bar.sync 0;");
}

__device__ __forceinline__ void mma_fp8(
        float& d0, float& d1, float& d2, float& d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1,
        float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void mma_fp8_sf(
        float& d0, float& d1, float& d2, float& d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1,
        float c0, float c1, float c2, float c3) {
    const uint32_t sfa = 127u, sfb = 127u;
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

constexpr int kIters = 64;
constexpr int kUnroll = 32;

template <bool kSF>
__global__ void k_bench_mma_lat(uint32_t* out_cycles, float* sink) {
    const int lane = threadIdx.x & 31;
    uint32_t a0 = 0x01010101u + lane, a1 = 0x02020202u, a2 = 0x03030303u, a3 = 0x04040404u;
    uint32_t b0 = 0x05050505u, b1 = 0x06060606u;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    bar_sync();
    uint32_t t0 = clock32();
    #pragma unroll
    for (int i = 0; i < kIters * kUnroll; ++i) {
        if constexpr (kSF)
            mma_fp8_sf(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
        else
            mma_fp8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
    uint32_t t1 = clock32();
    bar_sync();
    if (lane == 0) out_cycles[0] = t1 - t0;
    if (lane == 0) sink[0] = d0 + d1 + d2 + d3;
}

template <bool kSF>
__global__ void k_bench_mma_tput(uint32_t* out_cycles, float* sink) {
    const int lane = threadIdx.x & 31;
    uint32_t a0 = 0x01010101u + lane, a1 = 0x02020202u, a2 = 0x03030303u, a3 = 0x04040404u;
    uint32_t b0 = 0x05050505u, b1 = 0x06060606u;
    float d[8][4];
    #pragma unroll
    for (int j = 0; j < 8; ++j) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f;
    bar_sync();
    uint32_t t0 = clock32();
    #pragma unroll 1
    for (int i = 0; i < kIters; ++i) {
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if constexpr (kSF)
                    mma_fp8_sf(d[j][0], d[j][1], d[j][2], d[j][3], a0, a1, a2, a3, b0, b1,
                               d[j][0], d[j][1], d[j][2], d[j][3]);
                else
                    mma_fp8(d[j][0], d[j][1], d[j][2], d[j][3], a0, a1, a2, a3, b0, b1,
                            d[j][0], d[j][1], d[j][2], d[j][3]);
            }
            asm volatile("" ::: "memory");
        }
    }
    uint32_t t1 = clock32();
    bar_sync();
    if (lane == 0) out_cycles[0] = t1 - t0;
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
    if (lane == 0) sink[0] = s;
}

__global__ void k_bench_ffma_lat(uint32_t* out_cycles, float* sink) {
    const int lane = threadIdx.x & 31;
    float a = 1.0001f, b = 1.0002f, c = 0.001f * (lane + 1);
    asm volatile("" : "+f"(a), "+f"(b), "+f"(c));
    bar_sync();
    uint32_t t0 = clock32();
    #pragma unroll
    for (int i = 0; i < kIters * kUnroll; ++i)
        asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(c) : "f"(a), "f"(b));
    uint32_t t1 = clock32();
    bar_sync();
    if (lane == 0) out_cycles[0] = t1 - t0;
    if (lane == 0) sink[0] = c;
}

__global__ void k_bench_ffma_tput(uint32_t* out_cycles, float* sink) {
    const int lane = threadIdx.x & 31;
    float a = 1.0001f, b = 1.0002f, c[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = 0.001f * (lane + j + 1);
    asm volatile("" : "+f"(a), "+f"(b));
    bar_sync();
    uint32_t t0 = clock32();
    #pragma unroll 1
    for (int i = 0; i < kIters; ++i) {
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(c[j]) : "f"(a), "f"(b));
        }
    }
    uint32_t t1 = clock32();
    bar_sync();
    if (lane == 0) out_cycles[0] = t1 - t0;
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += c[j];
    if (lane == 0) sink[0] = s;
}

__global__ void k_bench_imad_lat(uint32_t* out_cycles, uint32_t* sink) {
    const int lane = threadIdx.x & 31;
    uint32_t a = 3u + lane, b = 5u, c = 7u;
    asm volatile("" : "+r"(a), "+r"(b), "+r"(c));
    bar_sync();
    uint32_t t0 = clock32();
    #pragma unroll
    for (int i = 0; i < kIters * kUnroll; ++i)
        asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(c) : "r"(a), "r"(b));
    uint32_t t1 = clock32();
    bar_sync();
    if (lane == 0) out_cycles[0] = t1 - t0;
    if (lane == 0) sink[0] = c;
}

static uint32_t run_u32(void (*k)(uint32_t*, uint32_t*)) {
    uint32_t *d_cyc, *d_sink, h = 0;
    CUDA_CHECK(cudaMalloc(&d_cyc, 4));
    CUDA_CHECK(cudaMalloc(&d_sink, 4));
    k<<<1, 32>>>(d_cyc, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h, d_cyc, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cyc));
    CUDA_CHECK(cudaFree(d_sink));
    return h;
}

static uint32_t run_f32(void (*k)(uint32_t*, float*)) {
    uint32_t *d_cyc, h = 0;
    float *d_sink;
    CUDA_CHECK(cudaMalloc(&d_cyc, 4));
    CUDA_CHECK(cudaMalloc(&d_sink, 4));
    k<<<1, 32>>>(d_cyc, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h, d_cyc, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cyc));
    CUDA_CHECK(cudaFree(d_sink));
    return h;
}

static void print_row(const char* name, uint32_t cycles, int nops, const char* notes) {
    printf("%-28s %10u %10.2f  %s\n", name, cycles, cycles / double(nops), notes);
}

int main() {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    printf("device: %s  sm_%d%d  SMs=%d  clock=%d kHz  smem/SM=%zu\n\n",
           p.name, p.major, p.minor, p.multiProcessorCount, clock_khz, p.sharedMemPerMultiprocessor);

    const int n_dep = kIters * kUnroll;
    const int n_tput = kIters * kUnroll * 8;

    printf("%-28s %10s %10s  %s\n", "op", "cycles", "cyc/op", "notes");
    print_row("QMMA latency",       run_f32(k_bench_mma_lat<false>),  n_dep,  "unscaled, dependent D");
    print_row("QMMA tput",          run_f32(k_bench_mma_tput<false>), n_tput, "unscaled, 8 accums");
    print_row("QMMA.SF latency",    run_f32(k_bench_mma_lat<true>),   n_dep,  "ue8m0=127, dependent D");
    print_row("QMMA.SF tput",       run_f32(k_bench_mma_tput<true>),  n_tput, "ue8m0=127, 8 accums");
    print_row("FFMA latency",       run_f32(k_bench_ffma_lat),        n_dep,  "dependent FMA");
    print_row("FFMA tput",          run_f32(k_bench_ffma_tput),       n_tput, "8 independent");
    print_row("IMAD.LO latency",    run_u32(k_bench_imad_lat),        n_dep,  "dependent integer mad");
    return 0;
}
