#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include <torch/python.h>

#include "../utils/compatibility.hpp"
#include "../utils/exception.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_bf16_ring_ag_gemm.hpp"
#endif

namespace deep_gemm::ring_ag {

// ---------------------------------------------------------------------------
// 单机多卡的"对称显存"：用 CUDA IPC 把每个 rank 的 buffer 映射到其它 rank 的地址
// 空间里，这样 kernel 里就能直接对 peer 显存做 load/store（走 NVLink）。
// 语义上等价于 DeepGEMM 里 mega kernel 用的 torch symmetric memory，只是那套需要
// torch >= 2.6，本机 torch 是 2.1.2，所以自己用 IPC 实现一份。
// ---------------------------------------------------------------------------

// 用 cudaMalloc（而不是 torch 的 caching allocator）分配，保证指针就是分配基址，
// 这样 IPC handle 打开后偏移为 0
static int64_t sym_alloc(const int64_t& num_bytes) {
    void* ptr = nullptr;
    DG_CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, num_bytes));
    DG_CUDA_RUNTIME_CHECK(cudaMemset(ptr, 0, num_bytes));
    return reinterpret_cast<int64_t>(ptr);
}

static void sym_free(const int64_t& ptr) {
    DG_CUDA_RUNTIME_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
}

static pybind11::bytes sym_get_handle(const int64_t& ptr) {
    cudaIpcMemHandle_t handle;
    DG_CUDA_RUNTIME_CHECK(cudaIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr)));
    return {reinterpret_cast<const char*>(&handle), sizeof(handle)};
}

static int64_t sym_open_handle(const pybind11::bytes& data) {
    const std::string str = data;
    DG_HOST_ASSERT(str.size() == sizeof(cudaIpcMemHandle_t));

    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, str.data(), sizeof(handle));

    void* ptr = nullptr;
    DG_CUDA_RUNTIME_CHECK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<int64_t>(ptr);
}

static void sym_close_handle(const int64_t& ptr) {
    DG_CUDA_RUNTIME_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
}

// 打开 P2P，失败时（已经打开过）忽略
static bool enable_peer_access(const int& peer_device) {
    int can_access = 0;
    int current_device = 0;
    DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&current_device));
    if (current_device == peer_device)
        return true;

    DG_CUDA_RUNTIME_CHECK(cudaDeviceCanAccessPeer(&can_access, current_device, peer_device));
    if (not can_access)
        return false;

    const auto err = cudaDeviceEnablePeerAccess(peer_device, 0);
    if (err != cudaSuccess and err != cudaErrorPeerAccessAlreadyEnabled) {
        DG_CUDA_RUNTIME_CHECK(err);
    }
    cudaGetLastError();
    return true;
}

// ---------------------------------------------------------------------------
// 通算融合：ring all-gather + GEMM
// ---------------------------------------------------------------------------
static void bf16_ring_all_gather_gemm(const torch::Tensor& a_local,
                                      const torch::Tensor& b,
                                      const torch::Tensor& d,
                                      const int64_t& self_recv_ptr, const int64_t& peer_recv_ptr,
                                      const int64_t& self_flag_ptr, const int64_t& peer_flag_ptr,
                                      const int64_t& self_consume_ptr, const int64_t& peer_consume_ptr,
                                      const int& rank, const int& num_ranks, const int& epoch,
                                      const std::string& compiled_dims) {
#if DG_TENSORMAP_COMPATIBLE
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 9 and "Ring AllGather GEMM currently only supports SM90");
    sm90_bf16_ring_ag_gemm(a_local, b, d,
                           self_recv_ptr, peer_recv_ptr,
                           self_flag_ptr, peer_flag_ptr,
                           self_consume_ptr, peer_consume_ptr,
                           rank, num_ranks, epoch, compiled_dims);
#else
    DG_HOST_UNREACHABLE("CUDA version is too old for the ring all-gather GEMM");
#endif
}

static void register_apis(pybind11::module_& m) {
    m.def("sym_alloc", &sym_alloc, pybind11::arg("num_bytes"));
    m.def("sym_free", &sym_free, pybind11::arg("ptr"));
    m.def("sym_get_handle", &sym_get_handle, pybind11::arg("ptr"));
    m.def("sym_open_handle", &sym_open_handle, pybind11::arg("data"));
    m.def("sym_close_handle", &sym_close_handle, pybind11::arg("ptr"));
    m.def("enable_peer_access", &enable_peer_access, pybind11::arg("peer_device"));

    m.def("bf16_ring_all_gather_gemm", &bf16_ring_all_gather_gemm,
          pybind11::arg("a_local"), pybind11::arg("b"), pybind11::arg("d"),
          pybind11::arg("self_recv_ptr"), pybind11::arg("peer_recv_ptr"),
          pybind11::arg("self_flag_ptr"), pybind11::arg("peer_flag_ptr"),
          pybind11::arg("self_consume_ptr"), pybind11::arg("peer_consume_ptr"),
          pybind11::arg("rank"), pybind11::arg("num_ranks"), pybind11::arg("epoch"),
          pybind11::arg("compiled_dims") = "nk");
}

} // namespace deep_gemm::ring_ag
