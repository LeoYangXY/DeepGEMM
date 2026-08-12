/**
 * =============================================================================
 * HGEMM with TMA + mbarrier + Multi-Stage Pipeline + Warp Specialization
 * =============================================================================
 *
 * Target: SM120 (RTX 5050 Blackwell consumer)
 * Tile: BM=128, BN=128, BK=16, Stages=7
 * Threads: 160 = 5 warps (warp0-3: MMA via WMMA API, warp4: TMA)
 *
 * Uses nvcuda::wmma API for Tensor Core (guaranteed correct register layout)
 * Uses TMA + mbarrier for async data movement with multi-stage pipeline
 * Uses warp specialization: producer warp (TMA) + consumer warps (MMA)
 *
 * Compile: nvcc -arch=sm_120 -O3 -lcuda -o hgemm_tma hgemm_tma_warp_spec.cu
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <math.h>

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 16
#define NUM_STAGES 7

#define NUM_MMA_WARPS 4
#define TMA_WARP_ID 4
#define WARP_SIZE 32
#define THREADS_PER_CTA ((NUM_MMA_WARPS + 1) * WARP_SIZE)

// Each warp handles 64x32 output via wmma 16x16x16
// Warp layout: 2x2 over BM x BN → each warp: 64x64
// Within each warp: 4x4 = 16 wmma tiles of 16x16
#define WM 64
#define WN 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define A_SMEM_BYTES (BM * BK * sizeof(half))
#define B_SMEM_BYTES (BN * BK * sizeof(half))
#define TX_BYTES (A_SMEM_BYTES + B_SMEM_BYTES)

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA Error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CU_CHECK(call) do { CUresult e=(call); if(e!=CUDA_SUCCESS){ \
    const char* s; cuGetErrorString(e,&s); fprintf(stderr,"CU Error %s:%d: %s\n",__FILE__,__LINE__,s);exit(1);}}while(0)

// =============================================================================
// PTX Helpers (for TMA + mbarrier only)
// =============================================================================
__device__ __forceinline__ uint32_t cvta_to_shared(const void* p) {
    uint32_t a; asm volatile("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
        : "=r"(a) : "l"(p)); return a;
}

__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t cnt) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(cvta_to_shared(b)), "r"(cnt) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        :: "r"(cvta_to_shared(b)), "r"(tx) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
    asm volatile(
        "{ .reg .pred p;\n"
        "LOOP: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra LOOP; }\n"
        :: "r"(cvta_to_shared(b)), "r"(phase) : "memory");
}

__device__ __forceinline__ void tma_load_2d(
    void* dst, uint64_t desc, int cx, int cy, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n"
        :: "r"(cvta_to_shared(dst)), "l"(desc), "r"(cvta_to_shared(bar)),
           "r"(cx), "r"(cy) : "memory");
}

// =============================================================================
// Shared Memory
// =============================================================================
struct Smem {
    uint64_t full_bar[NUM_STAGES];
    int32_t  done[NUM_STAGES];
    alignas(128) half sA[NUM_STAGES][BM * BK];  // [stage][row*BK+col], A[M][K]
    alignas(128) half sB[NUM_STAGES][BN * BK];  // [stage][row*BK+col], B[N][K]
};

// =============================================================================
// TMA Descriptor
// =============================================================================
static CUtensorMap make_tma(const half* p, int rows, int cols, int br, int bc) {
    CUtensorMap d;
    cuuint64_t dims[2]={(cuuint64_t)cols,(cuuint64_t)rows};
    cuuint64_t str[1]={(cuuint64_t)(cols*2)};
    cuuint32_t box[2]={(cuuint32_t)bc,(cuuint32_t)br};
    cuuint32_t es[2]={1,1};
    CU_CHECK(cuTensorMapEncodeTiled(&d,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,2,(void*)p,
        dims,str,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return d;
}

// =============================================================================
// Kernel
// =============================================================================
__global__ void __launch_bounds__(THREADS_PER_CTA)
hgemm_kernel(const __grid_constant__ CUtensorMap tma_A,
             const __grid_constant__ CUtensorMap tma_B,
             half* __restrict__ C, int M, int N, int K) {

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    extern __shared__ char raw[];
    Smem* sm = reinterpret_cast<Smem*>(raw);

    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbar_init(&sm->full_bar[s], 1);
            sm->done[s] = -1;
        }
    }
    __syncthreads();

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int nk = K / BK;

    // =========================================================================
    // PRODUCER (TMA warp)
    // =========================================================================
    if (warp_id == TMA_WARP_ID) {
        if (lane_id == 0) {
            for (int kt = 0; kt < nk; kt++) {
                int s = kt % NUM_STAGES;
                if (kt >= NUM_STAGES) {
                    int expected = kt - NUM_STAGES;
                    while (*(volatile int32_t*)&sm->done[s] < expected) {
                        __nanosleep(32);
                    }
                }
                mbar_expect_tx(&sm->full_bar[s], (uint32_t)TX_BYTES);
                int koff = kt * BK;
                tma_load_2d(&sm->sA[s][0], reinterpret_cast<uint64_t>(&tma_A),
                            koff, m0, &sm->full_bar[s]);
                tma_load_2d(&sm->sB[s][0], reinterpret_cast<uint64_t>(&tma_B),
                            koff, n0, &sm->full_bar[s]);
            }
        }
        return;
    }

    // =========================================================================
    // CONSUMER (MMA warps 0-3) using WMMA API
    // =========================================================================
    if (warp_id >= NUM_MMA_WARPS) return;

    // Warp layout: 2x2 over BM x BN
    const int wm = (warp_id / 2) * WM;  // 0 or 64
    const int wn = (warp_id % 2) * WN;  // 0 or 64

    // Each warp computes 64x64 output = 4x4 WMMA tiles of 16x16
    // Accumulator fragments: 4 along M × 4 along N
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);

    for (int kt = 0; kt < nk; kt++) {
        int s = kt % NUM_STAGES;
        int phase = (kt / NUM_STAGES) & 1;

        mbar_wait(&sm->full_bar[s], phase);

        // Load A and B fragments from SMEM and compute
        // A is [BM][BK] row-major in sA, stride = BK = 16
        // B is [BN][BK] row-major in sB, stride = BK = 16
        // For WMMA: A is row_major [16][16], B is col_major (= B^T row_major [N][K])
        #pragma unroll
        for (int mi = 0; mi < 4; mi++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            // A tile: rows [wm+mi*16 : wm+mi*16+16], cols [0:16]
            const half* a_ptr = &sm->sA[s][(wm + mi * WMMA_M) * BK];
            wmma::load_matrix_sync(a_frag, a_ptr, BK);  // leading dim = BK = 16

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                // B tile: rows [wn+ni*16 : wn+ni*16+16] of sB, cols [0:16]
                // col_major B means B^T[K][N], stored as B[N][K] row-major
                const half* b_ptr = &sm->sB[s][(wn + ni * WMMA_N) * BK];
                wmma::load_matrix_sync(b_frag, b_ptr, BK);  // leading dim = BK

                wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag, c_frag[mi][ni]);
            }
        }

        // Sync consumer warps before signaling done
        asm volatile("bar.sync 1, %0;\n" :: "r"(NUM_MMA_WARPS * WARP_SIZE) : "memory");

        if (tid == 0) {
            sm->done[s] = kt;
            __threadfence_block();
        }
    }

    // Epilogue: store accumulators to global memory
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            // Convert FP32 accumulator to FP16 and store
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_half;
            // Convert float to half
            for (int i = 0; i < c_frag[mi][ni].num_elements; i++)
                c_half.x[i] = __float2half(c_frag[mi][ni].x[i]);

            int r = m0 + wm + mi * WMMA_M;
            int c = n0 + wn + ni * WMMA_N;
            if (r + 16 <= M && c + 16 <= N) {
                wmma::store_matrix_sync(C + r * N + c, c_half, N, wmma::mem_row_major);
            }
        }
    }
}

// =============================================================================
int main() {
    CU_CHECK(cuInit(0));
    int M=4096, N=4096, K=4096;
    printf("=============================================================\n");
    printf("HGEMM: TMA + mbarrier + Pipeline + Warp Spec + WMMA Tensor Core\n");
    printf("Target: SM120 (RTX 5050 Blackwell)\n");
    printf("BM=%d BN=%d BK=%d Stages=%d Threads=%d\n", BM,BN,BK,NUM_STAGES,THREADS_PER_CTA);
    printf("C[%d,%d] = A[%d,%d] * B^T[%d,%d]\n", M,N,M,K,N,K);
    printf("=============================================================\n\n");

    size_t bA=(size_t)M*K*2, bB=(size_t)N*K*2, bC=(size_t)M*N*2;
    half *hA=(half*)malloc(bA), *hB=(half*)malloc(bB), *hC=(half*)malloc(bC);
    srand(42);
    for(size_t i=0;i<(size_t)M*K;i++) hA[i]=__float2half((rand()%10-5)*0.1f);
    for(size_t i=0;i<(size_t)N*K;i++) hB[i]=__float2half((rand()%10-5)*0.1f);

    half *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,bA)); CUDA_CHECK(cudaMalloc(&dB,bB)); CUDA_CHECK(cudaMalloc(&dC,bC));
    CUDA_CHECK(cudaMemcpy(dA,hA,bA,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,bB,cudaMemcpyHostToDevice));

    CUtensorMap tA=make_tma(dA,M,K,BM,BK), tB=make_tma(dB,N,K,BN,BK);
    dim3 grid((N+BN-1)/BN,(M+BM-1)/BM), block(THREADS_PER_CTA);
    int smem=sizeof(Smem);
    printf("SMEM: %d B (%.1f KB)\n", smem, smem/1024.f);
    CUDA_CHECK(cudaFuncSetAttribute(hgemm_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));

    hgemm_kernel<<<grid,block,smem>>>(tA,tB,dC,M,N,K);
    CUDA_CHECK(cudaDeviceSynchronize());
    if(cudaGetLastError()!=cudaSuccess){printf("❌ %s\n",cudaGetErrorString(cudaGetLastError()));goto end;}

    // Correctness
    { CUDA_CHECK(cudaMemcpy(hC,dC,bC,cudaMemcpyDeviceToHost));
      float mx_abs=0, mx_rel=0;
      for(int i=0;i<64;i++)for(int j=0;j<64;j++){
        float s=0; for(int k=0;k<K;k++) s+=__half2float(hA[i*K+k])*__half2float(hB[j*K+k]);
        float gpu=__half2float(hC[i*N+j]);
        float d=fabsf(gpu-s); if(d>mx_abs)mx_abs=d;
        float rel=(fabsf(s)>1e-3f)?d/fabsf(s):0; if(rel>mx_rel)mx_rel=rel;}
      printf("Correctness(64x64): max_abs=%.4f max_rel=%.4f %s\n",
             mx_abs, mx_rel, mx_rel<0.02f?"PASS ✅":"FAIL ❌"); }

    // Benchmark
    { for(int i=0;i<5;i++) hgemm_kernel<<<grid,block,smem>>>(tA,tB,dC,M,N,K);
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t t0,t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
      CUDA_CHECK(cudaEventRecord(t0));
      for(int i=0;i<100;i++) hgemm_kernel<<<grid,block,smem>>>(tA,tB,dC,M,N,K);
      CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
      float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));
      float us=ms*1000.f/100;
      printf("\n📊 Performance: %.1f µs | %.2f TFLOPS\n", us, 2.*M*N*K/(us*1e6));
      CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1)); }

end: free(hA);free(hB);free(hC); cudaFree(dA);cudaFree(dB);cudaFree(dC); return 0;
}