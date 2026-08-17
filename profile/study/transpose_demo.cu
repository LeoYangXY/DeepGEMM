// tmp demo: matrix transpose - naive vs shared-memory (coalesced) version
//
// We focus on the ncu "Memory Workload Analysis" section to show WHY the naive
// version is slow: strided global stores => poor DRAM/sector access, low L2 hit,
// lots of wasted bytes.
//
// compile: nvcc -O3 -arch=sm_120 -o transpose_demo transpose_demo.cu
// profile naive:
//   ncu -k transpose_naive -s 1 --memory-workload-analysis -o transpose_naive ./transpose_demo
// profile smem:
//   ncu -k transpose_smem  -s 1 --memory-workload-analysis -o transpose_smem  ./transpose_demo

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

static const int N = 1024;       // square matrix N x N
static const int TILE = 32;      // shared-mem tile

// ---------- naive: coalesced READ, STRIDED WRITE ----------
__global__ void transpose_naive(const float* A, float* B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        B[col * N + row] = A[row * N + col];   // write is strided across warps
    }
}

// ---------- shared-mem: coalesced READ and WRITE ----------
__global__ void transpose_smem(const float* A, float* B) {
    __shared__ float tile[TILE][TILE + 1];   // +1 to avoid bank conflict on write
    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int x = bx + threadIdx.x, y = by + threadIdx.y;

    // coalesced read into shared (A is row-major, x is contiguous)
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    __syncthreads();

    // transposed write-back: block (by,bx) writes tile to B at (bx,by)
    x = by + threadIdx.x;
    y = bx + threadIdx.y;
    if (x < N && y < N) B[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

// ---------- user's idea: row-WRITE, column-READ, NO padding ----------
// Stage 1: thread t reads A's row t (coalesced global read)  -> writes tile[t][0..31] (row write, bank = col, no conflict)
// Stage 2: thread t reads tile[col 0..31][t]  (column read, bank = t, leading_dim=32 => no conflict)
//           -> writes B's row t (coalesced global write)
// Because TILE == 32 == #banks and leading dim is 32 (multiple of 32), no padding needed.
__global__ void transpose_smem_nopad(const float* A, float* B) {
    __shared__ float tile[TILE][TILE];   // NO +1 padding on purpose
    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;

    // Stage 1: coalesced read A, write tile by ROW
    int gx = bx + threadIdx.x, gy = by + threadIdx.y;
    if (gx < N && gy < N) tile[threadIdx.y][threadIdx.x] = A[gy * N + gx];
    __syncthreads();

    // Stage 2: each thread t reads its COLUMN (tile[0..31][t]) and writes B row t
    // A tile read from (by,bx) -> transposed output to B block at (bx,by).
    // thread t writes B row (bx + t), columns (by + 0..31)
    int out_row = bx + threadIdx.x;
    if (out_row < N) {
        for (int r = 0; r < TILE; ++r) {
            int out_col = by + r;
            if (out_col < N)
                B[out_row * N + out_col] = tile[r][threadIdx.x];  // column read of smem
        }
    }
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *dA, *dB;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);

    float *hA = new float[(size_t)N * N];
    for (size_t i = 0; i < (size_t)N * N; ++i) hA[i] = (float)(i % 7);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // warmup + time naive
    transpose_naive<<<grid, block>>>(dA, dB);
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms_naive = 0;
    cudaEventRecord(s); for (int i=0;i<10;++i) transpose_naive<<<grid,block>>>(dA,dB);
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms_naive, s, e);
    ms_naive /= 10;

    // time smem
    float ms_smem = 0;
    cudaEventRecord(s); for (int i=0;i<10;++i) transpose_smem<<<grid,block>>>(dA,dB);
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms_smem, s, e);
    ms_smem /= 10;

    // time smem_nopad (user's no-padding row-write/col-read idea)
    float ms_nopad = 0;
    cudaEventRecord(s); for (int i=0;i<10;++i) transpose_smem_nopad<<<grid,block>>>(dA,dB);
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms_nopad, s, e);
    ms_nopad /= 10;

    // verify
    float *hB = new float[(size_t)N*N];
    cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int r=0;r<N && bad<5;++r) for (int c=0;c<N;++c)
        if (hB[c*N+r] != hA[r*N+c]) { bad++; }
    printf("matrix %dx%d\n", N, N);
    printf("naive : %.4f ms  (%.1f GB/s effective for %zu MB x2)\n",
           ms_naive, (bytes*2/1e9)/ (ms_naive/1e3), bytes/1048576);
    printf("smem  : %.4f ms  (%.1f GB/s effective)\n",
           ms_smem, (bytes*2/1e9)/ (ms_smem/1e3));
    printf("speedup smem/naive: %.2fx\n", ms_naive/ms_smem);
    printf("correctness: %s\n", bad==0 ? "OK" : "FAIL");

    // verify nopad (independent buffers to avoid scope clash)
    {
        float *hB2 = new float[(size_t)N*N];
        cudaMemcpy(hB2, dB, bytes, cudaMemcpyDeviceToHost);
        int bad2 = 0;
        for (int r=0;r<N && bad2<5;++r) for (int c=0;c<N;++c)
            if (hB2[c*N+r] != hA[r*N+c]) { bad2++; }
        printf("nopad : %.4f ms  (%.1f GB/s effective)  correctness: %s\n",
               ms_nopad, (bytes*2/1e9)/ (ms_nopad/1e3), bad2==0 ? "OK" : "FAIL");
        delete[] hB2;
    }

    cudaEventDestroy(s); cudaEventDestroy(e);
    delete[] hA; delete[] hB;
    cudaFree(dA); cudaFree(dB);
    return 0;
}
