#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <cute/int_tuple.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

// ============================================================================
//  SM120 raw FP8 GEMM — Blackwell consumer (RTX 5050 / sm_120a).
//
//  D_fp32 = C_fp32 + A_e4m3 @ B_e4m3^T
//  No per-128-K FP32 1D1D scales.  Host still goes through `fp8_fp4_gemm_nt`
//  (callers pass dummy SF tensors); this kernel never loads them.
//  Same math as `cublaslt_gemm_nt` on the raw FP8 tensors.
//
//  What changed vs. the SM90 1D1D kernel (structurally):
//    * The math backend uses warp-level `mma.sync` QMMA.SF (identity ue8m0)
//      instead of Hopper's `wgmma.mma_async` (sm_120 has no wgmma).
//    * All `warpgroup_reg_alloc / dealloc / arrive / commit_batch / wait /
//      fence_operand` intrinsics are removed (they don't exist on sm_120).
//    * TMA multicast / cluster >= 2 are disabled by the host-side heuristics
//      (`kNumTMAMulticast == 1` here), so the cluster-sync paths reduce to
//      simple `__syncthreads()` and the multicast-load path is never taken.
//    * A/B shared tiles are laid out K-major with TMA 128B swizzle, and the
//      warp-level MMA fragments are gathered with `ldmatrix.x4` (A: one
//      m16k32, B: two n8 tiles) — see `deep_gemm/mma/sm120.cuh`.
//    * No SF_A / SF_B TMA and no FFMA scale promote.
//
//  What is preserved:
//    * Persistent scheduler & per-CTA looping over (m_block, n_block) tiles
//    * TMA async loads of A / B into a `kNumStages`-deep ring
//    * mbarrier-based full/empty producer-consumer pipeline
//    * TMA store (reduce-add 2D) to write the FP32 result back to global mem
//    * KGroupedContiguous branch (dynamic tensor-map rewrite in smem)
// ============================================================================
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, typename cd_dtype_t,
          uint32_t kMinBlocksPerSM>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, kMinBlocksPerSM) void
sm120_fp8_gemm_1d1d_impl(__nv_fp8_e4m3* gmem_a_ptr, __nv_fp8_e4m3* gmem_b_ptr,
                        int* grouped_layout,
                        cute::TmaDescriptor* tensor_map_buffer,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // -------- Compile-time checks --------
    DG_STATIC_ASSERT((kNumTMAThreads == 32 or kNumTMAThreads == 128) and kNumMathThreads % 128 == 0, "Invalid Threads");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous, "Invalid GEMM type");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");
    DG_STATIC_ASSERT(kNumTMAMulticast == 1, "SM120 kernel does not support TMA multicast / cluster");
    // 128B TMA swizzle + ldmatrix.x4 (16B atoms XOR-permuted by row%8).
    DG_STATIC_ASSERT(kSwizzleAMode == 128 and kSwizzleBMode == 128, "SM120 kernel expects 128B-swizzled tiles");
    DG_STATIC_ASSERT(BLOCK_K * sizeof(__nv_fp8_e4m3) == 128, "128B swizzle requires BLOCK_K == 128 for FP8");
    DG_STATIC_ASSERT(BLOCK_N % 16 == 0, "BLOCK_N must be a multiple of 16 for ldmatrix.x4 B (two n8 tiles)");

    // Type aliases
    using MMA = typename mma::sm120::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % MMA::M == 0, "Invalid block size (BLOCK_M must be a multiple of 64)");

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // -------- Shared-memory layout --------
    static constexpr uint32_t SMEM_TENSOR_MAP_SIZE = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 2 : 0);
    // TMA CD box is 64 x BLOCK_N (store_block_m). Two math WGs share this
    // buffer and serialize stores when BLOCK_M == 128.
    static constexpr uint32_t SMEM_D_SIZE = MMA::M * BLOCK_N * sizeof(float);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    auto smem_tensor_map_a = reinterpret_cast<cute::TmaDescriptor*>(smem_buffer);
    auto smem_tensor_map_b = smem_tensor_map_a + 1;
    auto gmem_tensor_map_a = tensor_map_buffer + blockIdx.x * 2;
    auto gmem_tensor_map_b = gmem_tensor_map_a + 1;

    auto smem_d = reinterpret_cast<float*>(smem_buffer + SMEM_TENSOR_MAP_SIZE);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    constexpr auto SMEM_BARRIER_OFFSET = SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + i * static_cast<uint32_t>(sizeof(Barrier))));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + (kNumStages + i) * static_cast<uint32_t>(sizeof(Barrier))));
    });

    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
        if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            *smem_tensor_map_a = tensor_map_a_base;
            *smem_tensor_map_b = tensor_map_b_base;
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathThreads / 32);
        }
        cutlass::arch::fence_barrier_init();
    }

    // No cluster on SM120: a plain block-wide sync is enough
    __syncthreads();

    // Full unroll of the K pipeline in Normal mode; disable it for KGrouped
    // where K is dynamic per group. Math must unroll with the TMA loop:
    // a rolled math body rematerializes swizzle IADD into the QMMA window.
    constexpr uint32_t kNumPipelineUnrolls = (kGemmType == GemmType::KGroupedContiguous ? 0 : kNumStages);

    // Programmatic Dependent Launch handoff (safe no-op if not enabled)
    cudaGridDependencySynchronize();

    // -------- Persistent scheduler --------
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups,
                                      kNumTMAMulticast, kIsTMAMulticastOnA,
                                      kNumSMs, true, 128u, 128u>(
        shape_m, shape_n, shape_k, grouped_layout);

    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1};
    };
    uint32_t iter_idx = 0;

    // ========================================================================
    //  TMA warp-group (32 or 128 threads; one warp issues TMAs)
    // ========================================================================
    if (warp_idx >= kNumMathThreads / 32) {
        if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
            uint32_t last_group_idx = kNumGroups;

            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Multicast is disabled at compile time; keep the counts = 1
                const uint32_t num_tma_multicast_a = 1;
                const uint32_t num_tma_multicast_b = 1;

                const uint32_t num_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                const uint32_t m_idx = m_block_idx * BLOCK_M;
                const uint32_t n_idx = n_block_idx * BLOCK_N;

                // KGroupedContiguous: rewrite the tensor map in smem when the group changes
                if (kGemmType == GemmType::KGroupedContiguous and last_group_idx != scheduler.current_group_idx) {
                    last_group_idx = scheduler.current_group_idx;

                    const uint64_t current_k_offset = scheduler.current_k_cumsum;
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_a, gmem_a_ptr + current_k_offset * shape_m);
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_b, gmem_b_ptr + current_k_offset * shape_n);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a, scheduler.current_shape_k, scheduler.current_shape_k);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_b, scheduler.current_shape_k, scheduler.current_shape_k);

                    cute::tma_desc_commit_group();
                    cute::tma_desc_wait_group();
                    __syncwarp(1u << lane_idx);

                    *(gmem_tensor_map_a) = *(smem_tensor_map_a);
                    *(gmem_tensor_map_b) = *(smem_tensor_map_b);
                    ptx::tensor_map_release_gpu();
                    ptx::tensor_map_acquire_gpu(gmem_tensor_map_a);
                    ptx::tensor_map_acquire_gpu(gmem_tensor_map_b);
                }

                #pragma unroll kNumPipelineUnrolls
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const auto tensor_map_a_ptr = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_a : &tensor_map_a_base);
                    const auto tensor_map_b_ptr = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_b : &tensor_map_b_base);

                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(tensor_map_a_ptr, &full_barrier, smem_a[stage_idx], k_idx, m_idx, num_tma_multicast_a);
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(tensor_map_b_ptr, &full_barrier, smem_b[stage_idx], k_idx, n_idx, num_tma_multicast_b);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                }
            }
        }
    // ========================================================================
    //  Math warp-groups (tensor-core MMA + smem/TMA store)
    // ========================================================================
    } else {
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            DG_STATIC_ASSERT(BLOCK_M == MMA::M * (BLOCK_M <= 64 ? 1 : 2), "Invalid block sizes");
            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t current_group_idx = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_group_idx : 0);
            const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);

            float accum[MMA::kNumAccum] = {0};

            auto empty_barrier_arrive = [&](uint32_t s) {
                lane_idx == 0 ? empty_barriers[s]->arrive() : void();
            };

            #pragma unroll kNumPipelineUnrolls
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                full_barriers[stage_idx]->wait(phase);

                // Warp `warp_idx%4` owns 16 rows of the WG's 64-row slab.
                const uint32_t warp_in_wg  = warp_idx & 3;
                const uint32_t m_row_base  = math_wg_idx * MMA::M + warp_in_wg * 16;
                constexpr uint32_t LDA = BLOCK_K;

                auto* a_tile = smem_a[stage_idx] + m_row_base * LDA;
                auto* b_tile = smem_b[stage_idx];

                mma::sm120::mma_kblock_ldsm<BLOCK_N>(accum, a_tile, b_tile, lane_idx, [&]() {
                    empty_barrier_arrive(stage_idx);
                });
            }

            // -------- Epilogue (64-row D buffer) --------
            // r_0 indexes the full BLOCK_M tile (TMA gmem M); r_d_* indexes the
            // 64-row smem box. tma_store_wait is per-issuer, so the two math
            // WGs for BLOCK_M==128 wait on the thread that issued the previous
            // TMA before reusing the buffer.
            const auto r_d_0 = (warp_idx % 4) * 16 + row_idx;
            const auto r_d_1 = r_d_0 + 8;
            const bool is_tma_issuer = (warp_idx % 4 == 0) and cute::elect_one_sync();
            constexpr uint32_t kEpiBar = 2;

            auto wait_own_tma = [&]() {
                if (is_tma_issuer)
                    cute::tma_store_wait<0>();
            };
            auto store_and_tma = [&]() {
                wait_own_tma();
                cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

                const auto smem_d_0 = reinterpret_cast<float2*>(smem_d + r_d_0 * BLOCK_N + col_idx * 2);
                const auto smem_d_1 = reinterpret_cast<float2*>(smem_d + r_d_1 * BLOCK_N + col_idx * 2);
                #pragma unroll
                for (auto i = 0; i < MMA::kNumAccum / 4; ++ i) {
                    ptx::st_shared(smem_d_0 + i * 4, {accum[i * 4 + 0], accum[i * 4 + 1]});
                    ptx::st_shared(smem_d_1 + i * 4, {accum[i * 4 + 2], accum[i * 4 + 3]});
                }
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

                if (is_tma_issuer) {
                    cute::SM90_TMA_REDUCE_ADD_2D::copy(
                        &tensor_map_cd, smem_d_0, n_block_idx * BLOCK_N,
                        current_group_idx * shape_m + m_block_idx * BLOCK_M + math_wg_idx * MMA::M);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            };

            if constexpr (BLOCK_M > 64) {
                if (math_wg_idx == 0) {
                    store_and_tma();
                    wait_own_tma();
                }
                cutlass::arch::NamedBarrier::sync(256, kEpiBar);
                if (math_wg_idx == 1) {
                    store_and_tma();
                    wait_own_tma();
                }
                cutlass::arch::NamedBarrier::sync(256, kEpiBar);
            } else {
                store_and_tma();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_120a / Blackwell consumer");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
