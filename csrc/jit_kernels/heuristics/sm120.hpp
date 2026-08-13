#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm {

// ---------------------------------------------------------------------------
//  SM120 (Blackwell consumer, e.g. RTX 5050) — raw FP8 GEMM heuristics.
//
//  Design differences vs. `SM90ArchSpec`:
//    * cluster_m == cluster_n == 1  (multicast disabled)
//    * swizzle_a_mode == swizzle_b_mode == 128
//      (TMA 128B swizzle + ldmatrix.x4; 16B atoms XOR-permuted by row%8 so
//       the 8 addresses of an ldmatrix group hit 8 distinct bank groups)
//    * BLOCK_M / BLOCK_N chosen to be sm_120-friendly: BLOCK_M in {64, 128},
//      BLOCK_N in a coarser grid (each n8 tile is one mma issue, and mma
//      throughput on sm_120 is *much* lower per issue than wgmma, so smaller
//      is better than very wide). BLOCK_N is a multiple of 16 so B can be
//      loaded two n8 tiles per ldmatrix.x4.
//    * BLOCK_K == 128 for 128B TMA swizzle (not 1D1D scale granularity).
//    * No SF smem: kernel is unscaled A_fp8 @ B_fp8, FP32 D.
//    * SMEM budget conservatively assumed to be 100 KB — RTX 5050 supports up
//      to 100 KB per block dynamic smem (99 KB usable in practice).
// ---------------------------------------------------------------------------
struct SM120ArchSpec {
    // Conservative smem budget for a Blackwell-consumer SM (99 KB usable).
    static constexpr int smem_capacity = 99 * 1024;

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        DG_HOST_ASSERT(desc.gemm_type == GemmType::Normal or
                       desc.gemm_type == GemmType::KGroupedContiguous);
        DG_HOST_ASSERT(desc.kernel_type == KernelType::KernelNoSF);
        DG_HOST_ASSERT(desc.cd_dtype == torch::kFloat);
        DG_HOST_ASSERT(desc.get_mma_kind() == MmaKind::MXFP8FP4);

        // Block M / N candidates. `DG_SM120_BLOCK_M` + `DG_SM120_BLOCK_N` pin a
        // single tile for sweeps (must still fit >= 2 stages).
        std::vector<int> block_m_candidates = {64, 128};
        std::vector<int> block_n_candidates;
        for (int n = 16; n <= 128; n += 16)
            block_n_candidates.push_back(n);
        if (const int forced_m = get_env<int>("DG_SM120_BLOCK_M"); forced_m > 0) {
            const int forced_n = get_env<int>("DG_SM120_BLOCK_N");
            DG_HOST_ASSERT(forced_n > 0 and "DG_SM120_BLOCK_N required with DG_SM120_BLOCK_M");
            block_m_candidates = {forced_m};
            block_n_candidates = {forced_n};
        }

        const int block_k = 128;

        std::vector<Layout> candidates;
        for (int block_m: block_m_candidates) {
            for (int block_n: block_n_candidates) {
                if (block_m > 128 and block_n > 128)
                    continue;
                const auto layout = Layout{0, block_m, block_n, block_k, 1, 1};

                const auto storage_config = get_storage_config(desc, layout);
                const int num_stages = get_pipeline_config(desc, layout, storage_config).num_stages;
                if (num_stages < 2)
                    continue;
                candidates.push_back(layout);
            }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& /*desc*/, const Layout& layout) {
        // 128B TMA swizzle: BLOCK_K == 128 FP8 bytes, so get_swizzle_mode
        // would also return 128.  Forced here so the kernel's ldmatrix path
        // and the TMA descriptor always agree.
        constexpr int mma_m = 64;
        const int load_block_m = layout.block_m;
        const int load_block_n = layout.block_n;
        const int store_block_m = mma_m;
        const int store_block_n = layout.block_n;
        return {
            load_block_m, load_block_n,
            store_block_m, store_block_n,
            /*swizzle_a=*/128, /*swizzle_b=*/128, /*swizzle_cd=*/0
        };
    }

    static int tma_threads() {
        const int v = get_env<int>("DG_SM120_TMA_THREADS", 32);
        return v == 128 ? 128 : 32;
    }

    static int requested_cta_per_sm() {
        int v = get_env<int>("DG_SM120_CTA_PER_SM", 1);
        return v < 1 ? 1 : (v > 2 ? 2 : v);
    }

    // 384-thread CTAs at 2/SM overflow the 64K register file (~85 regs/thread).
    static int cta_per_sm_for_layout(const Layout& layout) {
        int cta = requested_cta_per_sm();
        const int threads = tma_threads() + (layout.block_m <= 64 ? 128 : 256);
        // 2 CTA/SM needs <= 128 regs/thread: 65536 / (2 * threads) >= 128 → threads <= 256.
        if (cta >= 2 and threads > 256)
            cta = 1;
        return cta;
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 8;

        // Epilogue TMA box is store_block_m x store_block_n (64 x N). Two math
        // warp-groups for BLOCK_M==128 serialize onto that buffer, so do not
        // budget a full BLOCK_M-row D tile — that 20–40 KB is what pinned
        // 128x80 at 2 stages / 1 CTA. Without 1D1D scale smem, 128x80 fits 3.
        const int smem_cd = align(storage_config.store_block_m * storage_config.store_block_n *
                                  static_cast<int>(c10::elementSize(desc.cd_dtype)), 1024);
        const int smem_barriers = kNumMaxStages * 8 * 2;

        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype);
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype);

        const int smem_tensormap =
            desc.gemm_type == GemmType::KGroupedContiguous ? 4 * static_cast<int>(sizeof(CUtensorMap)) : 0;

        const int smem_extra = smem_cd + smem_barriers + smem_tensormap;
        const int smem_per_stage = smem_a_per_stage + smem_b_per_stage;
        const int budget = smem_capacity / cta_per_sm_for_layout(layout);
        int num_stages = std::min((budget - smem_extra) / std::max(1, smem_per_stage), kNumMaxStages);
        num_stages = std::max(num_stages, 1);

        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages
        };
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        const int num_tma_threads = tma_threads();
        const int num_math_threads = layout.block_m <= 64 ? 128 : 256;
        const int cta = cta_per_sm_for_layout(layout);
        return {
            desc.num_sms * cta,
            /*num_sms_per_cluster=*/1,
            num_tma_threads + num_math_threads,
            num_tma_threads, num_math_threads,
            0, 0,
            cta
        };
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        // Simple cost model: prefer larger tiles that still fit >= 2 stages.
        const auto num_blocks =
            ceil_div(desc.get_expected_m(), layout.block_m) *
            ceil_div(desc.get_expected_n(), layout.block_n) *
            desc.get_expected_num_groups();
        const auto num_waves = ceil_div(num_blocks, desc.num_sms);
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;

        // Rough "less work per block" ~ smaller tile => more waves.
        int64_t num_cycles = static_cast<int64_t>(num_waves) * 1000 - last_wave_util;
        return {static_cast<int>(num_waves), static_cast<int>(last_wave_util), num_cycles, layout};
    }

    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        return a.num_cycles < b.num_cycles;
    }
};

} // namespace deep_gemm
