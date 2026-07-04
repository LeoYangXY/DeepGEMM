// Standalone test harness for fa3_ws_fwd_kernel (no torch) so that
// compute-sanitizer can run without torch's slow import.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fa3_kernel_core.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            printf("CUDA error %s:%d : %s\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(_e));                                  \
            exit(1);                                                         \
        }                                                                   \
    } while (0)
#endif

namespace fa3 {

// Build TMA descriptors on the host and launch the kernel for one (B,H,N,D) blob.
void run(int B, int H, int N, int D, bool causal, int warmup, int iters) {
    using Element = cutlass::half_t;
    size_t nelems = (size_t)B * H * N * D;
    size_t bytes = nelems * sizeof(Element);

    Element *dq, *dk, *dv, *do_;
    CUDA_CHECK(cudaMalloc(&dq, bytes));
    CUDA_CHECK(cudaMalloc(&dk, bytes));
    CUDA_CHECK(cudaMalloc(&dv, bytes));
    CUDA_CHECK(cudaMalloc(&do_, bytes));

    // Simple deterministic fill (finite, small) to avoid NaN.
    std::vector<float> hq(nelems), hk(nelems), hv(nelems);
    for (size_t i = 0; i < nelems; ++i) {
        hq[i] = 0.01f * ((i % 7) - 3);
        hk[i] = 0.01f * ((i % 5) - 2);
        hv[i] = 0.01f * ((i % 11) - 5);
    }
    std::vector<Element> hqh(nelems), hkh(nelems), hvh(nelems);
    for (size_t i = 0; i < nelems; ++i) {
        hqh[i] = Element(hq[i]);
        hkh[i] = Element(hk[i]);
        hvh[i] = Element(hv[i]);
    }
    CUDA_CHECK(cudaMemcpy(dq, hqh.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, hkh.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, hvh.data(), bytes, cudaMemcpyHostToDevice));

    auto stride_qk = cute::make_stride(int64_t(D), int64_t(1),
                                       int64_t(N * D), int64_t(H * N * D));
    auto shape_qkv = cute::make_shape(int64_t(N), int64_t(D), int64_t(H), int64_t(B));
    auto shape_V   = cute::make_shape(int64_t(kHeadDimV), int64_t(N), int64_t(H), int64_t(B));

    auto mQ = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<Element const*>(dq)), shape_qkv, stride_qk);
    auto mK = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<Element const*>(dk)), shape_qkv, stride_qk);
    auto mV = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<Element const*>(dv)),
        shape_V, cute::select<1, 0, 2, 3>(stride_qk));
    auto mO = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<Element*>(do_)), shape_qkv, stride_qk);

    TmaTraitsQ tma_load_Q = cute::make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, TileShape_MNK{}, ClusterShape{});
    TmaTraitsK tma_load_K = cute::make_tma_copy_B_sm90(
        cute::SM90_TMA_LOAD{}, mK, cute::take<0, 2>(SmemLayoutK{}), TileShape_MNK{}, ClusterShape{});
    TmaTraitsV tma_load_V = cute::make_tma_copy(
        cute::SM90_TMA_LOAD{}, mV, cute::take<0, 2>(SmemLayoutVt{}),
        cute::select<1, 2>(TileShape_MNK_PV{}), cute::Int<1>{});
    TmaTraitsO tma_store_O = cute::make_tma_copy(
        cute::SM90_TMA_STORE{}, mO, SmemLayoutO{},
        cute::select<0, 1>(TileShape_MNK_PV{}), cute::Int<1>{});

    Fa3TmaParams host_params{tma_load_Q, tma_load_K, tma_load_V, tma_store_O};
    Fa3TmaParams* dev_params = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_params, sizeof(Fa3TmaParams)));
    CUDA_CHECK(cudaMemcpy(dev_params, &host_params, sizeof(Fa3TmaParams), cudaMemcpyHostToDevice));

    int smem_size = int(sizeof(SharedStorage));
    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<void*>(fa3_ws_fwd_kernel<false>),
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<void*>(fa3_ws_fwd_kernel<true>),
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    int num_m_tiles = (N + kBlockM - 1) / kBlockM;
    int total_blocks = num_m_tiles * H * B;
    float scale_log2 = float(D > 0 ? 1.0 / sqrtf((float)D) : 1.0f) * 1.4426950408889634f;

    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

    auto launch = [&](auto* kernel) {
        kernel<<<dim3(total_blocks), dim3(256), smem_size, s1>>>(
            dev_params,
            reinterpret_cast<Element const*>(dq),
            reinterpret_cast<Element const*>(dk),
            reinterpret_cast<Element const*>(dv),
            reinterpret_cast<Element*>(do_),
            N, N, H, B, scale_log2);
    };

    for (int i = 0; i < warmup + iters; ++i) {
        if (causal) launch(fa3_ws_fwd_kernel<true>);
        else        launch(fa3_ws_fwd_kernel<false>);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("KERNEL ERROR (iter %d): %s\n", i, cudaGetErrorString(e));
            CUDA_CHECK(cudaFree(dev_params)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk));
            CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(do_));
            CUDA_CHECK(cudaStreamDestroy(s1)); CUDA_CHECK(cudaStreamDestroy(s2));
            exit(1);
        }
    }

    // Poll s1 for completion (timeout ~12s) without blocking on the default stream.
    cudaError_t e = cudaErrorNotReady;
    bool done = false;
    for (int t = 0; t < 120; ++t) {
        e = cudaStreamQuery(s1);
        if (e == cudaSuccess) { done = true; break; }
        if (e != cudaErrorNotReady) { break; }
        usleep(100000);
    }
    // Force a device sync so any kernel printf() buffer is flushed to stdout.
    { cudaError_t se = cudaDeviceSynchronize(); (void)se; }

    if (!done) {
        printf("SYNC ERROR / HANG: %s\n", cudaGetErrorString(e));
        CUDA_CHECK(cudaFree(dev_params)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk));
        CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(do_));
        CUDA_CHECK(cudaStreamDestroy(s1)); CUDA_CHECK(cudaStreamDestroy(s2));
        exit(1);
    }
    printf("OK: ran %d iters (B=%d H=%d N=%d D=%d causal=%d)\n", iters, B, H, N, D, (int)causal);

    // dump O to host to check if output is all-zero (debug)
    std::vector<Element> ho(nelems);
    CUDA_CHECK(cudaMemcpy(ho.data(), do_, bytes, cudaMemcpyDeviceToHost));
    float mx = 0.f, s = 0.f;
    for (auto v : ho) { mx = std::max(mx, std::abs((float)v)); s += (float)v; }
    printf("O max_abs=%f  O mean=%f  O[0]=%f O[1]=%f O[2]=%f\n", mx, s / nelems, (float)ho[0], (float)ho[1], (float)ho[2]);

    CUDA_CHECK(cudaFree(dev_params)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk));
    CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(do_));
    CUDA_CHECK(cudaStreamDestroy(s1)); CUDA_CHECK(cudaStreamDestroy(s2));
}

}  // namespace fa3

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 1;
    int H = argc > 2 ? atoi(argv[2]) : 1;
    int N = argc > 3 ? atoi(argv[3]) : 128;
    int causal = argc > 4 ? atoi(argv[4]) : 0;
    fa3::run(B, H, N, 128, causal != 0, 1, 1);
    return 0;
}
