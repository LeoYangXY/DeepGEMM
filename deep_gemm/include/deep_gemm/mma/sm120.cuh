// ============================================================================
//  SM120 (Blackwell consumer, e.g. RTX 5050 / sm_120a) FP8 tensor-core helpers.
//
//  Purpose
//  -------
//  Consumer Blackwell (sm_120) has no wgmma.  Math is the warp-level
//    `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
//         .m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`
//  which lowers to SASS `QMMA.SF.16832.F32.E4M3.E4M3.E8`.
//  Unscaled `QMMA.16832.F32.E4M3.E4M3` is half-rate on RTX 50; SF with
//  identity ue8m0=127 (scale=1) is the full-rate path.  Kernel math is
//  raw FP8 (D = A@B); identity SF does not change numerics.
//
//  Shared-memory path
//  ------------------
//  A/B tiles are K-major with TMA 128B swizzle.  Fragments are gathered with
//    `ldmatrix.sync.aligned.x4.m8n8.shared.b16`   (A: one m16k32)
//    `ldmatrix.sync.aligned.x4.m8n8.shared.b16`   (B: two n8 tiles at once)
//  The 128B swizzle XOR-permutes 16B atoms inside each 8-row group so that
//  the 8 addresses of an ldmatrix group hit 8 distinct 16B bank groups
//  (zero 32-bank conflicts).  Physical address:
//      phys = row * 128 + ((col_bytes/16) XOR (row % 8)) * 16
//  which is CuTe Swizzle<3,4,3> on a 128-byte K-major row.
// ============================================================================
#pragma once

#include <cuda/std/cstdint>
#include <cuda_fp8.h>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::mma::sm120 {

// --------------------------------------------------------------------------
//  Warp-level FP8 mma (m16n8k32, row/col, D = A*B + C, e4m3 * e4m3 -> f32)
//
//  PTX per-lane fragment layout (for lane t in [0,32)):
//    A  (m16 x k32, row-major from mma's viewpoint):
//        a[0]  = A[row = (t>>2) + 0 , col = (t & 3) * 4 + 0 .. +3 ]  (packed as .b8x4)
//        a[1]  = A[row = (t>>2) + 8 , col = (t & 3) * 4 + 0 .. +3 ]
//        a[2]  = A[row = (t>>2) + 0 , col = (t & 3) * 4 + 16 .. +19]
//        a[3]  = A[row = (t>>2) + 8 , col = (t & 3) * 4 + 16 .. +19]
//    B  (k32 x n8, col-major from mma's viewpoint -> in memory K-major with
//        one 8-wide N slab):
//        b[0]  = B[col = (t>>2)      , row = (t & 3) * 4 + 0  .. +3 ]  (packed .b8x4)
//        b[1]  = B[col = (t>>2)      , row = (t & 3) * 4 + 16 .. +19]
//    D  (m16 x n8):
//        d[0]  = D[row = (t>>2) + 0 , col = (t & 3) * 2 + 0]
//        d[1]  = D[row = (t>>2) + 0 , col = (t & 3) * 2 + 1]
//        d[2]  = D[row = (t>>2) + 8 , col = (t & 3) * 2 + 0]
//        d[3]  = D[row = (t>>2) + 8 , col = (t & 3) * 2 + 1]
// --------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k32_f32_e4m3_e4m3(
        float& d0, float& d1, float& d2, float& d3,
        const uint32_t& a0, const uint32_t& a1, const uint32_t& a2, const uint32_t& a3,
        const uint32_t& b0, const uint32_t& b1,
        const float& c0, const float& c1, const float& c2, const float& c3) {
    // ue8m0(127) == 2^0 == 1.  Same numerics as unscaled MMA; full-rate SASS.
    const uint32_t sfa = 127u, sfb = 127u;
    const uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, "
        "{%15, %16}, "
        "{%17}, "
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"(bidA), "h"(tidA),
          "r"(sfb), "h"(bidB), "h"(tidB));
}

// --------------------------------------------------------------------------
//  128B TMA swizzle: 16B-atom XOR with (row % 8), row stride = 128 B.
//  `tile_base` must be the start of an 8-row-aligned K-major slab.
// --------------------------------------------------------------------------
__device__ __forceinline__ uint32_t swizzle_128b_addr(
        const uint32_t tile_base_u32, const uint32_t row, const uint32_t col_bytes) {
    // phys = row * 128 + ((col/16) ^ (row % 8)) * 16
    const uint32_t phys = (row << 7) + (((col_bytes >> 4) ^ (row & 7u)) << 4) + (col_bytes & 15u);
    return tile_base_u32 + phys;
}

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ldmatrix.x4 -> 4 x u32.  Each of 32 lanes supplies a 16B-aligned row address.
__device__ __forceinline__ void ldmatrix_x4(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
                                            const uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& d0, uint32_t& d1, const uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(d0), "=r"(d1)
        : "r"(addr));
}

__device__ __forceinline__ uint32_t a_ldsm_addr(
        const uint32_t a_tile_u32, const uint32_t k_off, const uint32_t lane_id) {
    const uint32_t row = lane_id & 15u;
    const uint32_t col = k_off + ((lane_id >> 4) << 4);
    return swizzle_128b_addr(a_tile_u32, row, col);
}

__device__ __forceinline__ uint32_t b_ldsm_x2n_addr(
        const uint32_t b_tile_u32, const uint32_t n_base,
        const uint32_t k_off, const uint32_t lane_id) {
    const uint32_t n_row = n_base + (lane_id & 7u) + ((lane_id & 16u) >> 1);
    const uint32_t col   = k_off + ((lane_id & 8u) << 1);
    return swizzle_128b_addr(b_tile_u32, n_row, col);
}

// --------------------------------------------------------------------------
//  A fragment: one m16 x k32 FP8 tile occupying 16 rows of a K-major 128B-
//  swizzled buffer.  CUTLASS / mma.sync convention:
//      lane provides 16B at (row = lane%16, col = k_off + (lane/16)*16)
//  The 4 destination registers ARE the MMA A fragment {a0,a1,a2,a3}.
// --------------------------------------------------------------------------
__device__ __forceinline__ void load_a_fragment_ldsm(
        uint32_t (&a)[4],
        const uint32_t a_tile_u32,     // start of this warp's 16 rows, k = 0
        const uint32_t k_off,          // 0, 32, 64, 96
        const uint32_t lane_id) {
    ldmatrix_x4(a[0], a[1], a[2], a[3], a_ldsm_addr(a_tile_u32, k_off, lane_id));
}

// --------------------------------------------------------------------------
//  B fragment x2: one k32 x n8 tile.  8 N-rows x 32 K-bytes = two 8x8 b16
//  matrices, loaded with ldmatrix.x2.
//      lanes  0-7 : n_base + (lane%8), k_off
//      lanes 8-15 : n_base + (lane%8), k_off+16
// --------------------------------------------------------------------------
__device__ __forceinline__ void load_b_fragment_ldsm(
        uint32_t (&b)[2],
        const uint32_t b_tile_u32,
        const uint32_t n_base,
        const uint32_t k_off,
        const uint32_t lane_id) {
    const uint32_t n_row = n_base + (lane_id & 7u);
    const uint32_t col   = k_off + ((lane_id & 8u) << 1);  // bit3 -> +16
    ldmatrix_x2(b[0], b[1], swizzle_128b_addr(b_tile_u32, n_row, col));
}

// --------------------------------------------------------------------------
//  B fragment x4: TWO consecutive n8 tiles (n_base and n_base+8) in one
//  ldmatrix.x4.  Register mapping:
//      {b[0], b[1]} = MMA B of n_base+0..7
//      {b[2], b[3]} = MMA B of n_base+8..15
//  Address mapping:
//      lanes  0-7  : n_base+0..7,  k_off
//      lanes  8-15 : n_base+0..7,  k_off+16
//      lanes 16-23 : n_base+8..15, k_off
//      lanes 24-31 : n_base+8..15, k_off+16
// --------------------------------------------------------------------------
__device__ __forceinline__ void load_b_fragment_ldsm_x2n(
        uint32_t (&b)[4],
        const uint32_t b_tile_u32,
        const uint32_t n_base,
        const uint32_t k_off,
        const uint32_t lane_id) {
    ldmatrix_x4(b[0], b[1], b[2], b[3], b_ldsm_x2n_addr(b_tile_u32, n_base, k_off, lane_id));
}

// --------------------------------------------------------------------------
//  Software-pipelined K-block MMA (BLOCK_K = 128 = 4 x k32).
//
//  A[k+1] is issued at the start of k-step k so it hides in the whole N sweep.
//  B is 3-live / 1-fill when BLOCK_N >= 48 (4 RF slots): LDSM B[g+3] writes the
//  free slot *between* the two QMMAs of pair g (LSU vs Tensor).  Narrower N
//  keeps 2-slot ping-pong.  `arrive_after_last_ldsm` runs after the last B
//  LDSM so TMA can refill this stage during the tail QMMAs.
// --------------------------------------------------------------------------
template <uint32_t BLOCK_N, typename Arrive>
__device__ __forceinline__ void mma_kblock_ldsm(
        float* accum,
        const __nv_fp8_e4m3* smem_a_warp,   // 16 rows x 128 K, 128B-swizzled
        const __nv_fp8_e4m3* smem_b,        // BLOCK_N x 128 K, 128B-swizzled
        const uint32_t lane_id,
        Arrive&& arrive_after_last_ldsm) {
    static constexpr uint32_t kKSteps   = 4;            // 128 / 32
    static constexpr uint32_t kNPairs   = BLOCK_N / 16; // two n8 tiles per LDSM.x4
    static constexpr uint32_t kTotal    = kKSteps * kNPairs;
    static constexpr uint32_t kBLive    = (kNPairs >= 3u) ? 3u : 1u;
    static constexpr uint32_t kSlots    = (kNPairs >= 3u) ? 4u : 2u;

    const uint32_t a_base = smem_u32(smem_a_warp);
    const uint32_t b_base = smem_u32(smem_b);

    uint32_t a_frag[2][4];
    uint32_t b_frag[kSlots][4];

    load_a_fragment_ldsm(a_frag[0], a_base, 0, lane_id);
    #pragma unroll
    for (uint32_t p = 0; p < kBLive; ++ p)
        load_b_fragment_ldsm_x2n(b_frag[p], b_base, p * 16u, 0, lane_id);

    #pragma unroll
    for (uint32_t k_step = 0; k_step < kKSteps; ++ k_step) {
        const uint32_t a_stage = k_step & 1u;
        const uint32_t k_off = k_step * 32u;
        if (k_step + 1u < kKSteps)
            load_a_fragment_ldsm(a_frag[a_stage ^ 1u], a_base, k_off + 32u, lane_id);

        #pragma unroll
        for (uint32_t n_pair = 0; n_pair < kNPairs; ++ n_pair) {
            const uint32_t global = k_step * kNPairs + n_pair;
            const uint32_t use_slot = global % kSlots;
            float* d0 = accum + (n_pair * 2u) * 4u;
            float* d1 = accum + (n_pair * 2u + 1u) * 4u;
            const uint32_t* a = a_frag[a_stage];
            const uint32_t* b = b_frag[use_slot];

            mma_m16n8k32_f32_e4m3_e4m3(
                d0[0], d0[1], d0[2], d0[3],
                a[0], a[1], a[2], a[3],
                b[0], b[1],
                d0[0], d0[1], d0[2], d0[3]);

            const uint32_t next = global + kBLive;
            if (next < kTotal) {
                const uint32_t fill_slot = (use_slot + kBLive) % kSlots;
                load_b_fragment_ldsm_x2n(b_frag[fill_slot], b_base,
                                         (next % kNPairs) * 16u,
                                         (next / kNPairs) * 32u, lane_id);
                if (next + 1u == kTotal)
                    arrive_after_last_ldsm();
            }

            mma_m16n8k32_f32_e4m3_e4m3(
                d1[0], d1[1], d1[2], d1[3],
                a[0], a[1], a[2], a[3],
                b[2], b[3],
                d1[0], d1[1], d1[2], d1[3]);
        }
    }
}

template <uint32_t BLOCK_N>
__device__ __forceinline__ void mma_kblock_ldsm(
        float* accum,
        const __nv_fp8_e4m3* smem_a_warp,
        const __nv_fp8_e4m3* smem_b,
        const uint32_t lane_id) {
    mma_kblock_ldsm<BLOCK_N>(accum, smem_a_warp, smem_b, lane_id, []() {});
}

// --------------------------------------------------------------------------
//  Selector wrapping a *warp-group's* (4 warps = 128 threads) FP8 MMA tile
//  covering M=64, N=BLOCK_N, K=BLOCK_K.  Mirrors the SM90 WGMMA contract so
//  that the main kernel body stays structurally identical.
//
//  Fragment layout parity with SM90 wgmma (m64nN, per lane):
//    * Each warp in the group owns 16 rows of M (warp 0..3 -> M[0..16), etc.)
//    * `kNumAccum` = 64 * N / 128 = N / 2   (same as sm90::FP8MMA<N>)
//    * Per-lane float layout: for a given n-8 tile, 4 floats are laid out as
//        (row = warp*16 + lane/4 + 0, col = (lane%4)*2 + 0)
//        (row = warp*16 + lane/4 + 0, col = (lane%4)*2 + 1)
//        (row = warp*16 + lane/4 + 8, col = (lane%4)*2 + 0)
//        (row = warp*16 + lane/4 + 8, col = (lane%4)*2 + 1)
//      — matching the SM90 per-lane layout so the epilogue TMA store can stay.
// --------------------------------------------------------------------------
template <int N_>
struct FP8MMA {
    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N>
struct FP8MMASelector {
    using type = FP8MMA<N>;
};

} // namespace deep_gemm::mma::sm120
