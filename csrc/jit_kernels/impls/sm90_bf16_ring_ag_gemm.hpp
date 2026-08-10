#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// Ring AllGather + GEMM 通算融合 kernel 的 host 侧启动器（SM90 / BF16）
class SM90BF16RingAGGemmRuntime final: public LaunchRuntime<SM90BF16RingAGGemmRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;

        int shape_mp, num_ranks, rank;
        uint32_t flag_expected, peer_consume_expected;

        void *local_a_ptr, *self_recv_ptr, *peer_recv_ptr;
        void *self_flag_ptr, *peer_flag_ptr;
        void *self_consume_ptr, *peer_consume_ptr;

        CUtensorMap tensor_map_a_local;
        CUtensorMap tensor_map_a_recv;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_bf16_ring_ag_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_ring_ag_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {}
    >);
}};
)",
        get_compiled_dim(args.shape_mp, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode,
        args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.launch_config.num_sms, args.num_ranks);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.shape_mp, args.gemm_desc.n, args.gemm_desc.k, args.rank,
            args.local_a_ptr, args.self_recv_ptr, args.peer_recv_ptr,
            args.self_flag_ptr, args.peer_flag_ptr,
            args.self_consume_ptr, args.peer_consume_ptr,
            args.flag_expected, args.peer_consume_expected,
            args.tensor_map_a_local, args.tensor_map_a_recv,
            args.tensor_map_b, args.tensor_map_d));
    }
};

// a_local : [Mp, K]      本 rank 持有的 A 分片
// b       : [N, K]       本 rank 的权重
// d       : [M, N]       输出，M = num_ranks * Mp，第 c 段对应 rank c 的分片
// self_recv / peer_recv  : 本轮使用的接收区（[num_ranks-1, Mp, K] 的裸指针）
static void sm90_bf16_ring_ag_gemm(const torch::Tensor& a_local,
                                   const torch::Tensor& b,
                                   const torch::Tensor& d,
                                   const int64_t& self_recv_ptr, const int64_t& peer_recv_ptr,
                                   const int64_t& self_flag_ptr, const int64_t& peer_flag_ptr,
                                   const int64_t& self_consume_ptr, const int64_t& peer_consume_ptr,
                                   const int& rank, const int& num_ranks, const int& epoch,
                                   const std::string& compiled_dims) {
    DG_HOST_ASSERT(a_local.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(a_local.dim() == 2 and b.dim() == 2 and d.dim() == 2);
    DG_HOST_ASSERT(num_ranks >= 1 and rank >= 0 and rank < num_ranks);
    DG_HOST_ASSERT(epoch >= 1);

    const int shape_mp = static_cast<int>(a_local.size(0));
    const int shape_k = static_cast<int>(a_local.size(1));
    const int shape_n = static_cast<int>(b.size(0));
    const int shape_m = shape_mp * num_ranks;
    DG_HOST_ASSERT(b.size(1) == shape_k);
    DG_HOST_ASSERT(d.size(0) == shape_m and d.size(1) == shape_n);
    DG_HOST_ASSERT(a_local.is_contiguous() and b.is_contiguous());
    // 跨卡搬运按 16B 向量化
    DG_HOST_ASSERT((static_cast<int64_t>(shape_mp) * shape_k * 2) % 16 == 0);

    // 用整体 shape 做 heuristic（总工作量是 M x N）
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::KernelNoSF,
        .m = shape_m, .n = shape_n, .k = shape_k, .num_groups = 1,
        .a_dtype = a_local.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    auto config = get_best_config<SM90ArchSpec>(desc);

    // 1) 本 kernel 不支持 TMA multicast（cluster > 1），强制退回 cluster = 1
    config.layout.cluster_m = 1;
    config.layout.cluster_n = 1;

    // 2) ring 分段要求每个 chunk 的行数能被 BLOCK_M 整除，否则 chunk 边界上的
    //    m block 会跨到下一个 chunk 的数据
    while (config.layout.block_m > 16 and shape_mp % config.layout.block_m != 0)
        config.layout.block_m /= 2;
    DG_HOST_ASSERT(shape_mp % config.layout.block_m == 0 and "Mp must be divisible by a legal BLOCK_M");

    // 布局改了，重新推导 storage / pipeline / launch 配置
    config.storage_config = SM90ArchSpec::get_storage_config(desc, config.layout);
    config.pipeline_config = SM90ArchSpec::get_pipeline_config(desc, config.layout, config.storage_config);
    config.launch_config = SM90ArchSpec::get_launch_config(desc, config.layout);

    // A 的两个 TMA descriptor：step 0 读本地分片，之后读接收区
    const auto tensor_map_a_local = make_tma_a_desc(cute::UMMA::Major::K, a_local, shape_mp, shape_k,
                                                    config.storage_config.load_block_m,
                                                    config.layout.block_k,
                                                    shape_k, 1,
                                                    config.storage_config.swizzle_a_mode);

    // 接收区是 [(num_ranks - 1) * Mp, K] 的连续内存
    const auto recv_rows = std::max(num_ranks - 1, 1) * shape_mp;
    const auto recv_tensor = torch::from_blob(reinterpret_cast<void*>(self_recv_ptr),
                                              {recv_rows, shape_k},
                                              torch::TensorOptions().dtype(torch::kBFloat16).device(a_local.device()));
    const auto tensor_map_a_recv = make_tma_a_desc(cute::UMMA::Major::K, recv_tensor, recv_rows, shape_k,
                                                   config.storage_config.load_block_m,
                                                   config.layout.block_k,
                                                   shape_k, 1,
                                                   config.storage_config.swizzle_a_mode);

    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, shape_n, shape_k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(-2)), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, shape_m, shape_n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);

    const auto num_sms = config.launch_config.num_sms;
    const SM90BF16RingAGGemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size, 1),
        .shape_mp = shape_mp,
        .num_ranks = num_ranks,
        .rank = rank,
        // flag 是单调累加的，第 epoch 轮期望值为 epoch * num_sms
        .flag_expected = static_cast<uint32_t>(epoch) * static_cast<uint32_t>(num_sms),
        // 接收区做了 2 组双缓冲，覆盖前必须确认 peer 已经消费完第 epoch-2 轮
        .peer_consume_expected = epoch >= 3 ? static_cast<uint32_t>(epoch - 2) * static_cast<uint32_t>(num_sms) : 0u,
        .local_a_ptr = a_local.data_ptr(),
        .self_recv_ptr = reinterpret_cast<void*>(self_recv_ptr),
        .peer_recv_ptr = reinterpret_cast<void*>(peer_recv_ptr),
        .self_flag_ptr = reinterpret_cast<void*>(self_flag_ptr),
        .peer_flag_ptr = reinterpret_cast<void*>(peer_flag_ptr),
        .self_consume_ptr = reinterpret_cast<void*>(self_consume_ptr),
        .peer_consume_ptr = reinterpret_cast<void*>(peer_consume_ptr),
        .tensor_map_a_local = tensor_map_a_local,
        .tensor_map_a_recv = tensor_map_a_recv,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
    };
    const auto code = SM90BF16RingAGGemmRuntime::generate(args);
    const auto runtime = compiler->build("sm90_bf16_ring_ag_gemm", code);
    SM90BF16RingAGGemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
