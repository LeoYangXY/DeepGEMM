#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/int_tuple.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

// ============================================================================
//  SM90 FP8 GEMM 主 kernel（A/B 缩放因子均为 1D，即每 128 个 K 通道一个 scale）
//  支持两种 GEMM 类型：
//    - Normal             : 普通 dense GEMM，C/D 为 FP32
//    - KGroupedContiguous : weight 反向场景，按 K 轴分组（不同组 K 长度不同、连续排布）
//  整体采用「持久化调度（persistent scheduling）+ TMA 异步搬运 + WGMMA tensor core
//  + 双 warp-group 分工 + mbarrier 软件流水线」这套 Hopper 高性能范式。
// ============================================================================
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, typename cd_dtype_t>
// launch_bounds: 一块 CTA 共 kNumTMAThreads + kNumMathThreads 个线程，最少 1 个 block/SM
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d1d_impl(__nv_fp8_e4m3* gmem_a_ptr, __nv_fp8_e4m3* gmem_b_ptr,
                        int* grouped_layout,
                        cute::TmaDescriptor* tensor_map_buffer,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // -------- 编译期合法性检查 --------
    // TMA warp-group 固定 128 线程（1 个 warp 干活，其余空转），math warp-group 须是 128 的倍数
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid Threads");
    // FP8 缩放因子是 per-128-channel，所以 K 方向 tile 必须=128
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    // 本 kernel 只处理 Normal / KGroupedContiguous 两种类型
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous, "Invalid GEMM type");

    // C/D 数据类型只支持 FP32（带累加）
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // -------- 关键类型别名 --------
    // WGMMA：根据 BLOCK_N 选出对应的 Hopper GMMA MMA atom（来自 CuTe，如 MMA_64xNx32_F32E4M3E4M3_SS_TN）
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    // Barrier：cluster 内跨 CTA 同步用的事务屏障（TMA 生产者—消费者信号量）
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // 若编译期已给定形状常量（JIT 会把常量化），则用常量覆盖运行期参数（利于常量折叠/unroll）
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // -------- 共享内存各区域大小预算 --------
    // KGroupedContiguous 需要在 smem 里放两张 tensor map（A/B），用于运行时换组
    static constexpr uint32_t SMEM_TENSOR_MAP_SIZE = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 2 : 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(float);                // 输出累加缓冲（FP32）
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3); // 每个流水线 stage 的 A tile
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3); // 每个 stage 的 B tile
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);               // A 的 scale（[BLOCK_M, K/128]）
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * sizeof(float);               // B 的 scale（[BLOCK_N, K/128]）
    static constexpr uint32_t ALIGNED_SMEM_SFB_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFB_SIZE_PER_STAGE, 128u); // 128B 对齐
    DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 128 == 0, "Invalid TMA alignment");

    // 线程号 -> warp 号 / lane 号
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // -------- 预取 TMA descriptor 到 L2（让后续 TMA 指令更快命中） --------
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // 动态共享内存，按 1024B 对齐（配合 swizzle-128B 访存，避免 bank conflict）
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // -------- 在 smem 中划分各区域指针 --------
    // KGroupedContiguous 时，把 A/B 的 tensor map 拷到 smem，方便换组时改写
    auto smem_tensor_map_a = reinterpret_cast<cute::TmaDescriptor*>(smem_buffer);
    auto smem_tensor_map_b = smem_tensor_map_a + 1;
    auto gmem_tensor_map_a = tensor_map_buffer + blockIdx.x * 2; // 每个 CTA 在全局有一份自己的 tensor map 备份
    auto gmem_tensor_map_b = gmem_tensor_map_a + 1;

    // 数据区：D 缓冲 + 环形流水线的 A/B（kNumStages 份）+ A/B 的 scale
    auto smem_d = reinterpret_cast<float*>(smem_buffer + SMEM_TENSOR_MAP_SIZE);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE)); 
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });
    constexpr auto SMEM_SF_OFFSET = SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + i * SMEM_SFA_SIZE_PER_STAGE));
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * ALIGNED_SMEM_SFB_SIZE_PER_STAGE));
    });

    // -------- 屏障区：full / empty 各 kNumStages 个 --------
    // full_barrier ：TMA 生产者写完一块 tile 后到达（消费者 math 在此等待数据就绪）
    // empty_barrier：math 消费者用完该 stage 后到达（生产者 TMA 在此等待 stage 空闲可复用）
    constexpr auto SMEM_BARRIER_OFFSET = SMEM_SF_OFFSET + kNumStages * (SMEM_SFA_SIZE_PER_STAGE + ALIGNED_SMEM_SFB_SIZE_PER_STAGE);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + i * static_cast<uint32_t>(sizeof(Barrier))));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + (kNumStages + i) * static_cast<uint32_t>(sizeof(Barrier))));
    });

    // 由「专用 warp」初始化所有屏障（full 初值1=一个生产者；empty 初值=消费者线程束数，表示要等所有 math warp 到齐）
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // KGroupedContiguous：先把基 tensor map 拷进 smem，换组时直接改 smem 里的副本
        if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            *smem_tensor_map_a = tensor_map_a_base;
            *smem_tensor_map_b = tensor_map_b_base;
        }

        // 初始化 kNumStages 个 stage 的 full/empty 屏障
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // 让屏障的初始化对异步代理（TMA/async proxy）可见
        cutlass::arch::fence_barrier_init();
    }

    // 多 CTA cluster 时做 cluster 级同步；否则普通 __syncthreads，保证屏障对所有线程可见
    (kNumTMAMulticast > 1) ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // -------- 流水线展开控制 & 寄存器重配 --------
    // Normal 模式：把 K 维主循环完全 unroll 到 kNumStages 份（流水展开）；KGrouped 因 K 动态，不展开
    constexpr uint32_t kNumPipelineUnrolls = (kGemmType == GemmType::KGroupedContiguous ? 0 : kNumStages);
    // TMA warp 用少量寄存器；math warp 展开的越多寄存器需求越大（240 vs 232）
    constexpr uint32_t kNumTMARegisters = (kNumPipelineUnrolls == 0 ? 40 : 24);
    constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);

    // PDL（Programmatic Dependent Launch）：若有前置 kernel 依赖，则等待其完成再开始
    cudaGridDependencySynchronize();
    
    // -------- 持久化 block 调度器 --------
    // 一个 CTA 启动后不会只算一个 tile，而是反复向 scheduler 领取 (m_block, n_block) 直到全部算完，
    // 这样能消除 kernel 启动开销、保持 SM 繁忙。
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, true, 128u, 128u>(shape_m, shape_n, shape_k, grouped_layout);

    // -------- 流水线索引 -> (stage, phase) 的映射 --------
    // 用 iter_idx 在环形 kNumStages 上取模得到 stage，除以 kNumStages 取奇偶得到 phase（双缓冲翻转标志）
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1}; // Pipeline stage and phase
    };
    uint32_t iter_idx = 0;

    // ========================================================================
    //  分支 1：TMA warp-group（只用一个 warp 真正发 TMA，其余线程在此分支空转）
    //  职责：把 A、B、sfA、sfB 从全局内存异步搬进共享内存（双缓冲/多 stage 流水）
    // ========================================================================
    if (warp_idx >= kNumMathThreads / 32) {
        // 把本 warp-group 寄存器配额降到 kNumTMARegisters（腾出寄存器给 math warp-group）
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // 仅选出一个 warp 作为 TMA 主力（elect_one），避免多 warp 重复发 TMA
        if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
            uint32_t last_group_idx = kNumGroups;

            // 持久化循环：不断领取下一个待算的 (m_block, n_block)
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // 决定 A/B 各自启用几条 TMA multicast（cluster 内多 CTA 同时搬同一份数据，省带宽）
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");
                
                // 本 tile 所需的 K 方向 block 数
                const uint32_t num_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                const uint32_t m_idx = m_block_idx * BLOCK_M;
                const uint32_t n_idx = n_block_idx * BLOCK_N;

                // KGroupedContiguous：当进入新的一组（group）时，改写 A/B 的 tensor map 指向该组的实际地址/形状
                if (kGemmType == GemmType::KGroupedContiguous and last_group_idx != scheduler.current_group_idx) {
                    last_group_idx = scheduler.current_group_idx;

                    // 直接在 smem 里的 tensor map 副本上替换「全局地址」和「inner dim stride」
                    const uint64_t current_k_offset = scheduler.current_k_cumsum;
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_a, gmem_a_ptr + current_k_offset * shape_m);
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_b, gmem_b_ptr + current_k_offset * shape_n);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a, scheduler.current_shape_k, scheduler.current_shape_k);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_b, scheduler.current_shape_k, scheduler.current_shape_k);

                    // 等所有在途 TMA 完成，避免改写正在被使用的 tensor map
                    cute::tma_desc_commit_group();
                    cute::tma_desc_wait_group();
                    __syncwarp(1u << lane_idx);

                    // 把 smem 里改好的 tensor map 写回全局备份，并通知 GPU 重新获取（acquire）
                    *(gmem_tensor_map_a) = *(smem_tensor_map_a);
                    *(gmem_tensor_map_b) = *(smem_tensor_map_b);
                    ptx::tensor_map_release_gpu();
                    ptx::tensor_map_acquire_gpu(gmem_tensor_map_a);
                    ptx::tensor_map_acquire_gpu(gmem_tensor_map_b);
                }

                // K 维主循环：发出本 tile 所需的全部 stage 的 TMA 拷贝（Normal 下被完全 unroll）
                #pragma unroll kNumPipelineUnrolls
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                    // 取本迭代对应的 stage/phase，并先等「该 stage 空闲」（empty barrier 上一轮已释放）
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // 选定该 stage 的 full barrier（写完即到达此屏障）
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const uint32_t sf_k_idx = scheduler.current_sf_k_cumsum + k_block_idx; // scale 沿 K 的索引
                    // KGrouped 用 smem 里动态改写的 tensor map；否则用编译期常量 tensor map
                    const auto tensor_map_a_ptr = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_a : &tensor_map_a_base);
                    const auto tensor_map_b_ptr = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_b : &tensor_map_b_base);

                    // 四条 TMA：A 的 scale、B 的 scale、A 数据、B 数据，全部异步发射
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx], m_idx, sf_k_idx, num_tma_multicast_a);
                    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx], n_idx, sf_k_idx, num_tma_multicast_b);
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(tensor_map_a_ptr, &full_barrier, smem_a[stage_idx], k_idx, m_idx, num_tma_multicast_a);
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(tensor_map_b_ptr, &full_barrier, smem_b[stage_idx], k_idx, n_idx, num_tma_multicast_b);
                    // 告诉 full barrier：本次 TMA 预计写入多少字节（A+B+sfA+sfB），用于 arrive 计数
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
                }
            }

            // 多 CTA 场景下，退出前再等一轮 empty，确保分布式共享屏障能安全析构
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s) {
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                }
            }
        }
    // ========================================================================
    //  分支 2：Math warp-group（真正做 WGMMA 计算 + scale promote + 写回）
    // ========================================================================
    } else {
        // 把本 warp-group 寄存器配额升到 kNumMathRegisters（计算需要大量寄存器）
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // 用 __shfl_sync 促使 NVCC 用统一寄存器；math_wg_idx 标识是第几个 math warp-group（BLOCK_M>64 时有两个，各管 64 行）
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;
        // WGMMA 累加器按 4x4 片段组织：r_0/r_1 是同一线程内 M 方向上下两行偏移
        const auto r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;

        // 持久化循环：不断领取 (m_block, n_block)
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // 累加缓冲：accum 是本次 tile 的 WGMMA 累加结果；final_accum 是跨 K 累加后的 FP32 结果
            DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * (BLOCK_M <= 64 ? 1 : 2), "Invalid block sizes");
            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t current_group_idx = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_group_idx : 0);
            const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};
            float2 scales_b[WGMMA::kNumAccum / 4]; // B 的 scale（每线程 4 个 N 列，成对存为 float2）

            // 消费完一个 stage 后，通知 empty barrier（多 CTA 时按目标 CTA 到达）
            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            // K 维主循环（Normal 下 unroll）：每个 k_block 对应流水线一个 stage
            #pragma unroll kNumPipelineUnrolls
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                // 等本 stage 的数据 TMA 搬完（full barrier 到达）
                CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                full_barriers[stage_idx]->wait(phase);

                // 读 A 的 scale（必须在 warpgroup_arrive 之前读完 smem，避免下一个被调度的 block 污染）
                auto scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + r_0);
                auto scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + r_1);

                // 读 B 的 scale：每线程 4 个 N 列，按 col_idx 取对应的 float2（.x/.y 为相邻两列 scale）
                #pragma unroll
                for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                    scales_b[i] = ptx::ld_shared(reinterpret_cast<float2*>(smem_sfb[stage_idx] + i * 8 + col_idx * 2));

                // 提交 WGMMA 指令前，先把累加寄存器 fence 住（告诉编译器这些寄存器被异步指令使用）
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive(); // 声明「接下来这批 WGMMA 作为一个批次」
                // 在 K 方向（BLOCK_K=128 = 4 个 32 宽 WGMMA）上展开，逐片发出 wgmma.mma_async
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                    // 用 smem 地址构造 GMMA 描述符（desc），指向 A/B 的对应 32 宽片段
                    auto desc_a = mma::sm90::make_smem_desc(smem_a[stage_idx] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                    auto desc_b = mma::sm90::make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
                    WGMMA::wgmma(desc_a, desc_b, accum, k); // 真正发射 tensor core 指令（底层来自 CuTe 的 MMA atom）
                }
                ptx::warpgroup_commit_batch(); // 提交批次
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>(); // 等本批次所有 WGMMA 完成（0 表示等到全部完成）

                // 本 stage 数据已用完，通知 empty barrier，TMA 侧可复用该 stage
                empty_barrier_arrive(stage_idx);

                // -------- FP8 scale promote（反量化） --------
                // WGMMA 输出的是「未乘 scale 的整数域累加」，需乘回 scale_a * scale_b 还原到 FP32
                // 4x4 片段：行(0,1)->scale_a_0；行(2,3)->scale_a_1；列(0,2)->scale_b_0；列(1,3)->scale_b_1
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    const float &scale_b_0 = scales_b[i].x;
                    const float &scale_b_1 = scales_b[i].y;
                    final_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
                    final_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
                    final_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
                    final_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
                }
            }

            // -------- Epilogue：把 FP32 结果写回全局内存 --------
            // 先等上一轮 TMA store 完成（仅每 4 个 warp 选一个代表线程等待）
            if (warp_idx % 4 == 0 and cute::elect_one_sync())
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // 把 final_accum 写进 smem 的 D 缓冲（按 WGMMA 片段布局落到对应 (row, col)）
            const auto smem_d_0 = reinterpret_cast<float2*>(smem_d + r_0 * BLOCK_N + col_idx * 2);
            const auto smem_d_1 = reinterpret_cast<float2*>(smem_d + r_1 * BLOCK_N + col_idx * 2);
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                ptx::st_shared(smem_d_0 + i * 4, {final_accum[i * 4 + 0], final_accum[i * 4 + 1]});
                ptx::st_shared(smem_d_1 + i * 4, {final_accum[i * 4 + 2], final_accum[i * 4 + 3]});
            }
            cute::tma_store_fence(); // 保证 smem 写入对 TMA store 代理可见
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // 用 TMA store（reduce-add 2D）把 D 缓冲原子加回全局 C/D（支持 C=A@B+C 的 epilogue）
            if (warp_idx % 4 == 0 and cute::elect_one_sync()) {
                cute::SM90_TMA_REDUCE_ADD_2D::copy(
                    &tensor_map_cd, smem_d_0, n_block_idx * BLOCK_N,
                    current_group_idx * shape_m + m_block_idx * BLOCK_M + r_0);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    // 非 sm_90a 架构直接报错（本 kernel 强依赖 Hopper 的 wgmma/tma 指令）
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
