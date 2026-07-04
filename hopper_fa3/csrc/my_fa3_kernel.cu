// Warp-specialized FlashAttention-3 style forward kernel (Hopper / SM90).
// Host wrapper + pybind; the device kernel lives in fa3_kernel_core.h.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "fa3_kernel_core.h"

// pybind 扩展里缺少 CUDA_CHECK 宏（standalone 测试里有）。用 TORCH_CHECK 抛出，
// 与 torch 扩展风格一致。
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t _e = (call);                                  \
        TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",           \
                    cudaGetErrorString(_e));                      \
    } while (0)
#endif

namespace fa3 {

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q/k/v must be contiguous");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shape must be same");
    TORCH_CHECK(q.dim() == 4, "expect q/k/v shape = [B, H, N, D]");
    TORCH_CHECK(q.size(3) == kHeadDim, "only head_dim == 128 is supported in this kernel");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "only fp16/bf16 inputs are supported");
}

torch::Tensor my_fa3_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double softmax_scale,
    bool causal) {

    check_inputs(q, k, v);

    bool const is_bf16 = (q.dtype() == torch::kBFloat16);
    torch::Tensor qh = q, kh = k, vh = v;
    if (is_bf16) {
        qh = q.to(torch::kFloat16);
        kh = k.to(torch::kFloat16);
        vh = v.to(torch::kFloat16);
    }

    auto o = torch::empty_like(qh);
    int const B = qh.size(0), H = qh.size(1), N = qh.size(2), D = qh.size(3);
    int const num_m_tiles = (N + kBlockM - 1) / kBlockM;
    int const total_blocks = num_m_tiles * H * B;
    float const scale_log2 = float(softmax_scale) * 1.4426950408889634f;  // * log2(e)

    auto stream = at::cuda::getCurrentCUDAStream();

    auto stride_qk = cute::make_stride(int64_t(kHeadDim), int64_t(1),
                                       int64_t(N * kHeadDim),
                                       int64_t(H * N * kHeadDim));
    auto shape_qkv = cute::make_shape(int64_t(N), int64_t(kHeadDim), int64_t(H), int64_t(B));
    auto shape_V   = cute::make_shape(int64_t(kHeadDimV), int64_t(N), int64_t(H), int64_t(B));

    auto mQ = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<cutlass::half_t const*>(qh.data_ptr())),
        shape_qkv, stride_qk);
    auto mK = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<cutlass::half_t const*>(kh.data_ptr())),
        shape_qkv, stride_qk);
    auto mV = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<cutlass::half_t const*>(vh.data_ptr())),
        shape_V, cute::select<1, 0, 2, 3>(stride_qk));
    auto mO = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<cutlass::half_t*>(o.data_ptr())),
        shape_qkv, stride_qk);

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
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_params), sizeof(Fa3TmaParams)));
    CUDA_CHECK(cudaMemcpy(dev_params, &host_params, sizeof(Fa3TmaParams), cudaMemcpyHostToDevice));

    int smem_size = int(sizeof(SharedStorage));
    auto set_smem_attr = [&](auto* kernel) {
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<void*>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    };
    dim3 grid(total_blocks), block(256);

    auto launch = [&](auto* kernel) {
        set_smem_attr(kernel);
        kernel<<<grid, block, smem_size, stream>>>(
            dev_params,
            reinterpret_cast<cutlass::half_t const*>(qh.data_ptr()),
            reinterpret_cast<cutlass::half_t const*>(kh.data_ptr()),
            reinterpret_cast<cutlass::half_t const*>(vh.data_ptr()),
            reinterpret_cast<cutlass::half_t*>(o.data_ptr()),
            N, N, H, B, scale_log2);
    };

    if (causal) launch(fa3_ws_fwd_kernel<true>);
    else        launch(fa3_ws_fwd_kernel<false>);

    {
        cudaError_t _e = cudaGetLastError();
        TORCH_CHECK(_e == cudaSuccess, "my_fa3_forward kernel launch failed: ", cudaGetErrorString(_e));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dev_params));

    if (is_bf16) { o = o.to(torch::kBFloat16); }
    return o;
}

}  // namespace fa3

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_fa3_forward", &fa3::my_fa3_forward, "My FA3 forward (warp-specialized, TMA + WGMMA)");
}
