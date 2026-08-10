#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

// ============================================================================
// Ring AllGather + GEMM 通算融合 kernel（SM90 / Hopper，BF16）
// ============================================================================
//
// 语义
// ----
// world_size = R，每个 rank 持有 A 的一个分片 `a_local` [Mp, K]，以及本 rank 的
// 权重 B [N, K]。all-gather 之后 A_full = concat(A_0, ..., A_{R-1}) [M, K]，
// M = R * Mp。本 kernel 在一个 kernel 内同时完成：
//     1. ring 方式的 all-gather（每个 step 传一个 buffer 给下一个 rank）
//     2. D = A_full @ B^T   ([M, N])
//
// ring 调度
// ---------
// 记 chunk(s) = (rank - s + R) % R，即 rank 在第 s 个 step 参与计算/转发的分片：
//     step 0    : 手上是自己的 a_local            -> 计算它 & 把它推给 rank+1
//     step s>=1 : 等 rank-1 推过来的 recv_slot[s-1] -> 计算它 & 把它推给 rank+1
// 所以一共 R-1 次跨卡搬运，与 ring all-gather 的通信量完全一致
// （每个 rank 收发 (R-1)/R * |A_full| 字节）。
//
// 死锁问题
// --------
// 这里用的是**单边 put**（直接往 peer 的显存里写 + release 递增 flag），
// 不是两边 send/recv 配对，因此不存在"所有 rank 都先 recv 卡死"的问题。
// 但为了严格遵循 ring 的收发次序（也让 NVLink 流量错峰），仍然按奇偶卡分开：
//     偶数 rank：先 push（send）后 wait（recv）
//     奇数 rank：先 wait（recv）后 push（send）
// R=4 时依赖链为 0->1->2->3->0，偶数卡无条件先发，奇数卡等到就能发，不会成环。
//
// warp specialization
// -------------------
// 一个 block 里 kNumMathThreads + 128 个线程，分工如下（w = warp_idx - kNumMathThreads/32）：
//     math warp-groups : WGMMA 计算 + STSM/TMA-store 写回
//     w == 2           : TMA warp，发 `cp.async.bulk.tensor` 把 A/B 搬进 smem
//                        （在搬某个 chunk 之前会先等它的到达 flag）
//     w == 0, 1, 3     : comm warp，做跨卡 put（ld.global + st.global 到 peer 显存）
// 三类 warp 之间通过 mbarrier（full/empty）与 system-scope 的 flag 解耦，
// 因此「本 step 的 GEMM」与「下一 step 的跨卡搬运」天然是 overlap 的。
//
// 跨迭代的 buffer 复用
// --------------------
// recv buffer 做了 kNumRecvSlots(=2) 组双缓冲，并且每次调用结束时每个 rank 会把
// 自己的 `consume` 计数 +1；rank r 在往 rank r+1 写第 e 轮数据之前，会确认
// rank r+1 已经消费完第 e-2 轮，从而避免"跑得快的卡覆盖慢卡还在读的数据"。
// ============================================================================

namespace ring_ag {

// system scope 的 acquire load / release add，用于跨卡的到达通知
__device__ __forceinline__ uint32_t ld_acquire_sys_global(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

__device__ __forceinline__ void red_release_sys_global_add(uint32_t* ptr, uint32_t value) {
    asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ void spin_wait_ge(const uint32_t* ptr, const uint32_t& expected) {
    while (ld_acquire_sys_global(ptr) < expected) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(20);
#endif
    }
}

// ring 版本的持久化 block 调度器：按 step（ring 顺序）分段，段内沿用 DeepGEMM
// 的 "先 M 后 N 分组" 的 swizzle，保证 B 在 L2 里的复用
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumRanks, uint32_t kNumSMs>
struct RingScheduler {
    static constexpr uint32_t kNumMBlocksPerGroup = 16;

    uint32_t num_m_blocks_per_chunk;
    uint32_t num_n_blocks;
    uint32_t num_blocks_per_chunk;
    uint32_t num_total_blocks;
    uint32_t current_iter = 0xffffffff;
    uint32_t rank;

    __device__ __forceinline__ RingScheduler(const uint32_t& shape_mp, const uint32_t& shape_n,
                                             const uint32_t& rank) {
        this->rank = rank;
        num_m_blocks_per_chunk = math::ceil_div(shape_mp, BLOCK_M);
        num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
        num_blocks_per_chunk = num_m_blocks_per_chunk * num_n_blocks;
        num_total_blocks = num_blocks_per_chunk * kNumRanks;
    }

    // 返回：step（ring 步数）、chunk（全局分片号）、chunk 内的 m block、n block
    __device__ __forceinline__ bool get_next_block(uint32_t& step, uint32_t& chunk_idx,
                                                   uint32_t& m_block_in_chunk, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;
        if (next_block_idx >= num_total_blocks)
            return false;

        step = next_block_idx / num_blocks_per_chunk;
        chunk_idx = (rank + kNumRanks - step) % kNumRanks;

        // 段内 swizzle
        const auto in_chunk_idx = next_block_idx % num_blocks_per_chunk;
        const auto num_blocks_per_group = num_n_blocks * kNumMBlocksPerGroup;
        const auto group_idx = in_chunk_idx / num_blocks_per_group;
        const auto first_m_block_idx = group_idx * kNumMBlocksPerGroup;
        const auto num_m_blocks_in_group = min(kNumMBlocksPerGroup,
                                               num_m_blocks_per_chunk - first_m_block_idx);
        const auto in_group_idx = in_chunk_idx % num_blocks_per_group;
        m_block_in_chunk = first_m_block_idx + in_group_idx % num_m_blocks_in_group;
        n_block_idx = in_group_idx / num_m_blocks_in_group;
        return true;
    }
};

} // namespace ring_ag

template <uint32_t SHAPE_MP, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs, uint32_t kNumRanks>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_bf16_ring_ag_gemm_impl(uint32_t shape_mp, uint32_t shape_n, uint32_t shape_k,
                            uint32_t rank,
                            // ring 通信相关的指针（都是 IPC 打开后的可直接寻址地址）
                            const uint4* local_a_ptr,       // 本 rank 自己的分片（只读）
                            const uint4* self_recv_ptr,     // 本 rank 的接收区（本轮 slot）
                            uint4* peer_recv_ptr,           // rank+1 的接收区（本轮 slot）
                            uint32_t* self_flag_ptr,        // 本 rank 的到达 flag[R-1]
                            uint32_t* peer_flag_ptr,        // rank+1 的到达 flag[R-1]
                            uint32_t* self_consume_ptr,     // 本 rank 的消费计数
                            const uint32_t* peer_consume_ptr,
                            uint32_t flag_expected,         // = epoch * kNumSMs
                            uint32_t peer_consume_expected, // = max(epoch - 2, 0) * kNumSMs
                            const __grid_constant__ cute::TmaDescriptor tensor_map_a_local,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_a_recv,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using cd_dtype_t = cutlass::bfloat16_t;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // 编译期 shape（若给了常量则覆盖运行期值）
    shape_mp = SHAPE_MP != 0 ? SHAPE_MP : shape_mp;
    shape_n  = SHAPE_N  != 0 ? SHAPE_N  : shape_n;
    shape_k  = SHAPE_K  != 0 ? SHAPE_K  : shape_k;

    // Shared memory 布局
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(cd_dtype_t)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);

    // 角色划分
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kTMAWarpBase = kNumMathThreads / 32;
    constexpr uint32_t kNumCommWarps = 3;
    constexpr uint32_t kNumCommThreads = kNumCommWarps * 32;
    constexpr uint32_t kCommBarrierId = 4;
    constexpr uint32_t kEndBarrierId = 5;
    DG_STATIC_ASSERT(kNumTMAThreads == 128, "Ring AG GEMM requires a 128-thread TMA warp-group");

    // 最前面预取 TMA descriptor
    if (warp_idx == kTMAWarpBase and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_local);
        cute::prefetch_tma_descriptor(&tensor_map_a_recv);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory of A/B/D must be aligned to 1024 bytes");

    auto smem_d = reinterpret_cast<cd_dtype_t*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });

    if (warp_idx == kTMAWarpBase + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathThreads / 32);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    cudaGridDependencySynchronize();

    // ring 调度器
    uint32_t step, chunk_idx, m_block_in_chunk, n_block_idx;
    auto scheduler = ring_ag::RingScheduler<BLOCK_M, BLOCK_N, kNumRanks, kNumSMs>(shape_mp, shape_n, rank);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kTMAWarpBase) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        const uint32_t warp_off = warp_idx - kTMAWarpBase;

        if (warp_off == 2) {
            // ---------------- TMA warp：把 A/B 搬进 shared memory ----------------
            if (cute::elect_one_sync()) {
                // 已经确认到达的最大 step（step 0 是本地数据，永远可用）
                uint32_t ready_step = 0;

                while (scheduler.get_next_block(step, chunk_idx, m_block_in_chunk, n_block_idx)) {
                    // 该 chunk 是跨卡搬过来的，先等它到齐
                    if constexpr (kNumRanks > 1) {
                        while (ready_step < step) {
                            ring_ag::spin_wait_ge(self_flag_ptr + ready_step, flag_expected);
                            ++ ready_step;
                        }
                    }

                    // step 0 读本地分片，其余读接收区第 step-1 个 slot
                    const auto* tensor_map_a = (step == 0) ? &tensor_map_a_local : &tensor_map_a_recv;
                    const uint32_t m_idx = (step == 0)
                                         ? (m_block_in_chunk * BLOCK_M)
                                         : ((step - 1) * shape_mp + m_block_in_chunk * BLOCK_M);
                    const uint32_t n_idx = n_block_idx * BLOCK_N;

                    const auto num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        auto& full_barrier = *full_barriers[stage_idx];
                        const uint32_t k_idx = k_block_idx * BLOCK_K;

                        tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, false>(
                            tensor_map_a, &full_barrier, smem_a[stage_idx], k_idx, m_idx, 1, 0);
                        tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, false>(
                            &tensor_map_b, &full_barrier, smem_b[stage_idx], k_idx, n_idx, 1, 0);
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                    }
                }
            }
        } else if constexpr (kNumRanks > 1) {
            // ---------------- comm warps：ring 方式的跨卡 put ----------------
            const uint32_t comm_warp_slot = (warp_off == 3) ? 2 : warp_off;   // {0, 1, 3} -> {0, 1, 2}
            const uint32_t comm_local_tid = comm_warp_slot * 32 + lane_idx;
            const uint32_t comm_tid = blockIdx.x * kNumCommThreads + comm_local_tid;
            constexpr uint32_t kCommStride = kNumSMs * kNumCommThreads;
            constexpr uint32_t kUnroll = 4;

            // 一个 chunk 有多少个 16B 向量
            const uint32_t chunk_num_vec = (shape_mp * shape_k * sizeof(__nv_bfloat16)) / sizeof(uint4);

            // 覆盖 peer 上第 e-2 轮的数据之前，先确认 peer 已经消费完
            if (peer_consume_expected > 0 and comm_local_tid == 0)
                ring_ag::spin_wait_ge(peer_consume_ptr, peer_consume_expected);
            cutlass::arch::NamedBarrier::sync(kNumCommThreads, kCommBarrierId);

            // 把 slot `s` 的数据推给下一个 rank，然后 release 递增它的 flag[s]
            auto push_to_next = [&](const uint32_t& s) {
                const uint4* src = (s == 0) ? local_a_ptr : (self_recv_ptr + (s - 1) * chunk_num_vec);
                uint4* dst = peer_recv_ptr + s * chunk_num_vec;

                for (uint32_t i = comm_tid; i < chunk_num_vec; i += kCommStride * kUnroll) {
                    uint4 v[kUnroll];
                    #pragma unroll
                    for (uint32_t u = 0; u < kUnroll; ++ u) {
                        const uint32_t idx = i + u * kCommStride;
                        if (idx < chunk_num_vec)
                            v[u] = src[idx];
                    }
                    #pragma unroll
                    for (uint32_t u = 0; u < kUnroll; ++ u) {
                        const uint32_t idx = i + u * kCommStride;
                        if (idx < chunk_num_vec)
                            dst[idx] = v[u];
                    }
                }

                // 保证数据先于 flag 对 peer 可见
                __threadfence_system();
                cutlass::arch::NamedBarrier::sync(kNumCommThreads, kCommBarrierId);
                if (comm_local_tid == 0)
                    ring_ag::red_release_sys_global_add(peer_flag_ptr + s, 1);
            };

            // 等 slot `j` 到齐（block 内只让一个线程 spin）
            auto wait_slot = [&](const uint32_t& j) {
                if (comm_local_tid == 0)
                    ring_ag::spin_wait_ge(self_flag_ptr + j, flag_expected);
                cutlass::arch::NamedBarrier::sync(kNumCommThreads, kCommBarrierId);
            };

            const bool send_first = (rank % 2 == 0);
            for (uint32_t s = 0; s < kNumRanks - 1; ++ s) {
                if (send_first) {
                    // 偶数卡：先 send，再 recv 下一步要转发的 buffer
                    push_to_next(s);
                    if (s + 1 < kNumRanks - 1)
                        wait_slot(s);
                } else {
                    // 奇数卡：先 recv 本步要转发的 buffer，再 send
                    if (s > 0)
                        wait_slot(s - 1);
                    push_to_next(s);
                }
            }
        }
    } else {
        // ---------------- math warp-groups：WGMMA ----------------
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        auto a_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], math_wg_idx * WGMMA::M, 0);
        auto b_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        while (scheduler.get_next_block(step, chunk_idx, m_block_in_chunk, n_block_idx)) {
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= 64 or warp_idx < kNumWGMMAStoreThreads / 32;

            const auto num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                full_barriers[stage_idx]->wait(phase);

                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode, nv_bfloat16>(
                            a_desc_base_lo, local_idx * WAVE_BLOCK_M, k * WGMMA::K, 0);
                        b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode, nv_bfloat16>(
                            b_desc_base_lo, 0, k * WGMMA::K, 0);
                        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
                    }
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
            }

            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                             "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            if (not do_wgmma_store)
                continue;

            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            DG_STATIC_ASSERT(kSwizzleDMode > 0, "Invalid swizzling type");
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    constexpr uint32_t kNumBankGroupBytes = 16;
                    auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                    auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
                    constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                    auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                    auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                    col ^= row % (kSwizzleDMode / 16);

                    auto smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                        warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +
                        m_offset * kSwizzleDMode +
                        atom_offset * BLOCK_M * kSwizzleDMode +
                        row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            // D 的全局行号 = chunk 在 A_full 里的偏移 + chunk 内的 m 偏移
            const uint32_t m_idx = chunk_idx * shape_mp + m_block_in_chunk * BLOCK_M;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr,
                                              n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }

    // 全 block 汇合后，把"本轮已消费完"通知给上游 rank（它靠这个判断能否覆盖我的接收区）
    if constexpr (kNumRanks > 1) {
        cutlass::arch::NamedBarrier::sync(kNumTMAThreads + kNumMathThreads, kEndBarrierId);
        if (threadIdx.x == 0)
            ring_ag::red_release_sys_global_add(self_consume_ptr, 1);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
