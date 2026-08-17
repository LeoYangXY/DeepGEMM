// tmp demo: kernel-internal fine-grained timing via clock64()
// compile: nvcc -O3 -arch=sm_120 -o kernel_clock_demo kernel_clock_demo.cu
// run:     ./kernel_clock_demo
//
// NOTE: we use asm volatile("" ::: "memory") between phases to stop the
// compiler from reordering/merging the clock64() reads, and we write the
// timestamps through a volatile pointer so the optimizer keeps them live.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// We record 4 timestamps per block => 3 phases:
//   t0 -> t1 : phase A (load A/B tile into shared mem)
//   t1 -> t2 : phase B (compute FMA on the tile)
//   t2 -> t3 : phase C (store result tile to global)
// timestamps are in GPU clock cycles (clock64()).
static const int NTS = 4;

static const int TILE = 64;   // output tile size (64x64)
static const int BDIM = 16;   // block dim: 16x16 = 256 threads (<=1024 limit)
static const int PER = TILE / BDIM;  // each thread handles PER x PER = 4x4 outputs

__global__ void timed_gemm_block(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int N,
                                 uint64_t* __restrict__ ts) {
    extern __shared__ float sbuf[];
    float* sA = sbuf;
    float* sB = sbuf + TILE * TILE;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    uint64_t t0 = clock64();
    asm volatile("" ::: "memory");

    // ---- Phase A: cooperatively load the 64x64 tile into shared mem ----
    // 256 threads, 4096 elements => each thread loads 16 elements (4x4 block)
    for (int i = ty * PER; i < ty * PER + PER; ++i) {
        for (int j = tx * PER; j < tx * PER + PER; ++j) {
            int gr = by * TILE + i;
            int gc = bx * TILE + j;
            if (gr < N && gc < N) {
                sA[i * TILE + j] = A[gr * N + gc];
                sB[i * TILE + j] = B[gr * N + gc];
            }
        }
    }
    __syncthreads();
    uint64_t t1 = clock64();
    asm volatile("" ::: "memory");

    // ---- Phase B: each thread computes its PER x PER outputs ----
    float acc[PER][PER];
    #pragma unroll
    for (int p = 0; p < PER; ++p)
        #pragma unroll
        for (int q = 0; q < PER; ++q)
            acc[p][q] = 0.f;

    for (int k = 0; k < TILE; ++k) {
        float a[PER], b[PER];
        #pragma unroll
        for (int p = 0; p < PER; ++p) a[p] = sA[(ty*PER + p) * TILE + k];
        #pragma unroll
        for (int q = 0; q < PER; ++q) b[q] = sB[k * TILE + (tx*PER + q)];
        #pragma unroll
        for (int p = 0; p < PER; ++p)
            #pragma unroll
            for (int q = 0; q < PER; ++q)
                acc[p][q] += a[p] * b[q];
    }
    uint64_t t2 = clock64();
    asm volatile("" ::: "memory");

    // ---- Phase C: store results back to global C ----
    for (int p = 0; p < PER; ++p) {
        for (int q = 0; q < PER; ++q) {
            int gr = by * TILE + ty * PER + p;
            int gc = bx * TILE + tx * PER + q;
            if (gr < N && gc < N) C[gr * N + gc] = acc[p][q];
        }
    }
    __syncthreads();
    uint64_t t3 = clock64();

    // write timestamps for this block (only thread 0), via volatile
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int bid = blockIdx.y * gridDim.x + blockIdx.x;
        volatile uint64_t* vts = ts + bid * NTS;
        vts[0] = t0; vts[1] = t1; vts[2] = t2; vts[3] = t3;
    }
}

int main() {
    const int N = 512;               // matrix dim
    dim3 block(BDIM, BDIM);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    int nBlocks = grid.x * grid.y;

    size_t matBytes = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    uint64_t *dTs;
    cudaMalloc(&dA, matBytes);
    cudaMalloc(&dB, matBytes);
    cudaMalloc(&dC, matBytes);
    cudaMalloc(&dTs, (size_t)nBlocks * NTS * sizeof(uint64_t));

    // init
    float *hA = new float[(size_t)N * N];
    float *hB = new float[(size_t)N * N];
    for (size_t i = 0; i < (size_t)N * N; ++i) { hA[i] = 1.f; hB[i] = 1.f; }
    cudaMemcpy(dA, hA, matBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, matBytes, cudaMemcpyHostToDevice);

    size_t shm = 2 * TILE * TILE * sizeof(float);
    timed_gemm_block<<<grid, block, shm>>>(dA, dB, dC, N, dTs);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("LAUNCH ERR: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("SYNC ERR: %s\n", cudaGetErrorString(err)); return 1; }

    uint64_t *hTs = new uint64_t[(size_t)nBlocks * NTS];
    cudaMemcpy(hTs, dTs, (size_t)nBlocks * NTS * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // clock rate for cycle -> seconds conversion
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clkKHz = 0;
    cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);  // kHz
    double ghz = clkKHz / 1e6;

    // aggregate phases
    unsigned long long sumA = 0, sumB = 0, sumC = 0;
    unsigned long long maxA = 0, maxB = 0, maxC = 0;
    for (int b = 0; b < nBlocks; ++b) {
        uint64_t t0 = hTs[b*NTS+0], t1 = hTs[b*NTS+1], t2 = hTs[b*NTS+2], t3 = hTs[b*NTS+3];
        unsigned long long a = t1 - t0, bb = t2 - t1, c = t3 - t2;
        sumA += a; sumB += bb; sumC += c;
        if (a > maxA) maxA = a; if (bb > maxB) maxB = bb; if (c > maxC) maxC = c;
    }

    auto pct = [](unsigned long long s, unsigned long long tot){
        return tot ? (100.0 * s / tot) : 0.0;
    };
    unsigned long long totSum = sumA + sumB + sumC;
    unsigned long long totMax = maxA + maxB + maxC;

    printf("=== kernel-internal phase timing (per block, GPU cycles) ===\n");
    printf("GPU clock: %.3f GHz (sm_%d)\n", ghz, prop.major*10+prop.minor);
    printf("blocks: %d  N=%d  tile=%d  block=%dx%d\n\n", nBlocks, N, TILE, BDIM, BDIM);
    printf("%-10s %12s %12s %12s %12s\n", "phase", "avg(cyc)", "max(cyc)", "avg(us)", "share%");
    printf("%-10s %12llu %12llu %12.3f %11.1f%%\n", "A load", sumA/nBlocks, maxA, (sumA/nBlocks)/ghz/1000.0, pct(sumA, totSum));
    printf("%-10s %12llu %12llu %12.3f %11.1f%%\n", "B compute", sumB/nBlocks, maxB, (sumB/nBlocks)/ghz/1000.0, pct(sumB, totSum));
    printf("%-10s %12llu %12llu %12.3f %11.1f%%\n", "C store", sumC/nBlocks, maxC, (sumC/nBlocks)/ghz/1000.0, pct(sumC, totSum));
    printf("-------------------------------------------------------------\n");
    printf("%-10s %12llu %12llu %12.3f %11.1f%%\n", "TOTAL", totSum/nBlocks, totMax, (totSum/nBlocks)/ghz/1000.0, 100.0);

    // verify one result element.
    // NOTE: each 64x64 tile does a partial GEMM of length TILE=64 (not full N),
    // so C[0][0] = TILE (since A,B are all-ones). This confirms the math ran.
    float check;
    cudaMemcpy(&check, dC, sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nC[0] = %.1f (expected %.1f, since A,B are all-ones and tile K=%d)\n", check, (float)TILE, TILE);

    delete[] hA; delete[] hB; delete[] hTs;
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dTs);
    return 0;
}
