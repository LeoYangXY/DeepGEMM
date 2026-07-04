// Shared kernel core for the FA3-style warp-specialized forward kernel.
// Contains only device-side code + CUTE/CUTLASS type definitions (no torch),
// so it can be included both by the pybind module and by a standalone test
// harness (for compute-sanitizer debugging without torch's slow import).
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/copy_sm90.hpp>

#include <cutlass/gemm/collective/builders/sm90_common.inl>

#include "softmax.h"

#ifndef KRUN_WGMMA
#define KRUN_WGMMA 1
#endif

namespace fa3 {

using namespace cute;

// ---- Fixed tile geometry (d=128, fp16/bf16 path) -----------------------------
static constexpr int kBlockM = 64;
static constexpr int kBlockN = 128;
static constexpr int kHeadDim = 128;
static constexpr int kHeadDimV = 128;
static constexpr int kStages = 2;

using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using TileShape_MNK    = cute::Shape<cute::Int<kBlockM>,  cute::Int<kBlockN>,  cute::Int<kHeadDim>>;
using TileShape_MNK_PV = cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDimV>, cute::Int<kBlockN>>;

static constexpr int NumThreadsPerWarpGroup = 128;
static constexpr int NumMmaThreads = 128;            // 1 MMA warpgroup
static constexpr int kNRows = 2 * (2 * kBlockM / NumThreadsPerWarpGroup);  // rows/thread = 2 for M=64, 1 WG

static constexpr cute::GMMA::Major MmaMajorV = cute::GMMA::Major::MN;

using AtomLayoutQK = cute::Layout<cute::Shape<cute::Int<kBlockM / 64>, cute::_1, cute::_1>>;

using TiledMmaQK = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<cutlass::half_t, cutlass::half_t, float, TileShape_MNK>(),
    AtomLayoutQK{}));

using TiledMmaQK_SS = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<cutlass::half_t, cutlass::half_t, float, TileShape_MNK>(),
    AtomLayoutQK{}));

using TiledMmaPV = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<cutlass::half_t, cutlass::half_t, float, TileShape_MNK_PV,
                               cute::GMMA::Major::K, MmaMajorV>(),
    AtomLayoutQK{}));

// ---- Smem layouts (mirrors FA3 sm90 warp-specialized mainloop) --------------
using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, cutlass::half_t,
    decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
using SmemLayoutQ = decltype(cute::tile_to_shape(SmemLayoutAtomQ{}, cute::select<0, 2>(TileShape_MNK{})));

using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, cutlass::half_t,
    decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
using SmemLayoutK = decltype(cute::tile_to_shape(
    SmemLayoutAtomK{},
    cute::make_shape(cute::shape<1>(TileShape_MNK{}), cute::shape<2>(TileShape_MNK{}), cute::Int<kStages>{})));

using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<MmaMajorV, cutlass::half_t,
    cute::Int<kHeadDimV>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
using SmemLayoutVt = decltype(cute::tile_to_shape(
    SmemLayoutAtomVt{},
    cute::make_shape(cute::Int<kHeadDimV>{}, cute::shape<2>(TileShape_MNK_PV{}), cute::Int<kStages>{}),
    std::conditional_t<MmaMajorV == cute::GMMA::Major::K, cute::Step<cute::_1, cute::_2, cute::_3>, cute::Step<cute::_2, cute::_1, cute::_3>>{}));

using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, cutlass::half_t,
    decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
using SmemLayoutP = decltype(cute::tile_to_shape(SmemLayoutAtomP{}, cute::select<0, 1>(TileShape_MNK{})));

using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, cutlass::half_t,
    decltype(cute::get<0>(TileShape_MNK_PV{})), decltype(cute::get<1>(TileShape_MNK_PV{}))>());
using SmemLayoutO = decltype(cute::tile_to_shape(SmemLayoutAtomO{}, cute::select<0, 1>(TileShape_MNK_PV{})));

// Position-independent swizzle requires the smem buffer to be aligned to the
// swizzle's native alignment (FA3 does exactly this for smem_p).
static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
static constexpr size_t SmemAlignmentO = cutlass::detail::alignment_for_swizzle(SmemLayoutO{});

static constexpr uint32_t TmaBytesQ = uint32_t(kBlockM) * kHeadDim * sizeof(cutlass::half_t);
static constexpr uint32_t TmaBytesK = uint32_t(kBlockN) * kHeadDim * sizeof(cutlass::half_t);
static constexpr uint32_t TmaBytesV = uint32_t(kHeadDimV) * kBlockN * sizeof(cutlass::half_t);

using MainloopPipeline = cutlass::PipelineTmaAsync<kStages>;
using PipelineState = cutlass::PipelineState<kStages>;

// ---- Shared storage ----------------------------------------------------------
struct SharedStorage {
    cute::array_aligned<cutlass::half_t, cute::cosize_v<SmemLayoutQ>, 128> smem_q;
    cute::array_aligned<cutlass::half_t, cute::cosize_v<SmemLayoutK>, 128> smem_k;
    union {
        cute::array_aligned<cutlass::half_t, cute::cosize_v<SmemLayoutVt>, 128> smem_v;
        cute::array_aligned<cutlass::half_t, cute::cosize_v<SmemLayoutO>, SmemAlignmentO> smem_o;
    };
    cute::array_aligned<cutlass::half_t, cute::cosize_v<SmemLayoutP>, SmemAlignmentP> smem_p;
    alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_Q;
    alignas(16) MainloopPipeline::SharedStorage pipeline_k;
    alignas(16) MainloopPipeline::SharedStorage pipeline_v;
};

using GmemShape  = cute::Shape<int64_t, int64_t, int64_t, int64_t>;
using GmemStride = cute::Stride<int64_t, int64_t, int64_t, int64_t>;
using GmemTensorC = cute::Tensor<cute::ViewEngine<cute::gmem_ptr<cutlass::half_t const*>>,
                                 cute::Layout<GmemShape, GmemStride>>;
using GmemTensorN = cute::Tensor<cute::ViewEngine<cute::gmem_ptr<cutlass::half_t*>>,
                                 cute::Layout<GmemShape, GmemStride>>;

using TmaTraitsQ = decltype(cute::make_tma_copy_A_sm90(
    cute::SM90_TMA_LOAD{}, std::declval<GmemTensorC const&>(),
    SmemLayoutQ{}, TileShape_MNK{}, ClusterShape{}));
using TmaTraitsK = decltype(cute::make_tma_copy_B_sm90(
    cute::SM90_TMA_LOAD{}, std::declval<GmemTensorC const&>(),
    cute::take<0, 2>(SmemLayoutK{}), TileShape_MNK{}, ClusterShape{}));
using TmaTraitsV = decltype(cute::make_tma_copy(
    cute::SM90_TMA_LOAD{}, std::declval<GmemTensorC const&>(),
    cute::take<0, 2>(SmemLayoutVt{}), cute::select<1, 2>(TileShape_MNK_PV{}), cute::Int<1>{}));
using TmaTraitsO = decltype(cute::make_tma_copy(
    cute::SM90_TMA_STORE{}, std::declval<GmemTensorN const&>(),
    SmemLayoutO{}, cute::select<0, 1>(TileShape_MNK_PV{}), cute::Int<1>{}));

struct Fa3TmaParams {
    TmaTraitsQ tma_load_Q;
    TmaTraitsK tma_load_K;
    TmaTraitsV tma_load_V;
    TmaTraitsO tma_store_O;
};

// =============================================================================
template <bool Is_causal>
__global__ void __launch_bounds__(256, 1)
fa3_ws_fwd_kernel(
    Fa3TmaParams const* tma_params,
    cutlass::half_t const* __restrict__ q_ptr,
    cutlass::half_t const* __restrict__ k_ptr,
    cutlass::half_t const* __restrict__ v_ptr,
    cutlass::half_t*       __restrict__ o_ptr,
    int const seqlen_q, int const seqlen_k,
    int const nheads,  int const batch,
    float const softmax_scale_log2) {

    TmaTraitsQ const& tma_load_Q = tma_params->tma_load_Q;
    TmaTraitsK const& tma_load_K = tma_params->tma_load_K;
    TmaTraitsV const& tma_load_V = tma_params->tma_load_V;
    TmaTraitsO const& tma_store_O = tma_params->tma_store_O;

    using Element = cutlass::half_t;

    int const thread_idx = threadIdx.x;
    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const warp_group_idx = cutlass::canonical_warp_group_idx();
    int const warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;



    int const num_m_tiles = (seqlen_q + kBlockM - 1) / kBlockM;
    int const m_block = blockIdx.x % num_m_tiles;
    int const bidh    = (blockIdx.x / num_m_tiles) % nheads;
    int const bidb    = (blockIdx.x / num_m_tiles) / nheads;

    extern __shared__ __align__(128) char shared_storage_bytes[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_storage_bytes);

    Element* smem_q_base = shared_storage.smem_q.data();
    Element* smem_k_base = shared_storage.smem_k.data();
    Element* smem_v_base = shared_storage.smem_v.data();
    Element* smem_p_base = shared_storage.smem_p.data();
    Element* smem_o_base = shared_storage.smem_o.data();

    Tensor sQ = cute::make_tensor(cute::make_smem_ptr(smem_q_base), SmemLayoutQ{});
    Tensor sK = cute::make_tensor(cute::make_smem_ptr(smem_k_base), SmemLayoutK{});
    Tensor sV = cute::make_tensor(cute::make_smem_ptr(smem_v_base), SmemLayoutVt{});
    Tensor sP = cute::make_tensor(cute::make_smem_ptr(smem_p_base), SmemLayoutP{});
    Tensor sO = cute::make_tensor(cute::make_smem_ptr(smem_o_base), SmemLayoutO{});

    // ---- Pipeline + Q barrier setup (all threads) ---------------------------
    if (warp_idx == 0 && lane_predicate) { shared_storage.barrier_Q.init(1); }

    MainloopPipeline::Params params_k;
    params_k.role = (warp_group_idx == 0)
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    params_k.transaction_bytes = TmaBytesK;
    params_k.is_leader = (warp_group_thread_idx == 0);
    params_k.num_consumers = NumMmaThreads;

    MainloopPipeline::Params params_v = params_k;
    params_v.transaction_bytes = TmaBytesV;

    MainloopPipeline pipeline_k(shared_storage.pipeline_k, params_k, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, params_v, ClusterShape{});

    __syncthreads();

    // =========================================================================
    // PRODUCER warpgroup: TMA load Q (once) + stream K/V tiles
    // =========================================================================
    if (warp_group_idx == 0) {
#if KRUN_WGMMA
#endif

        auto shape_V = cute::make_shape(int64_t(kHeadDimV), int64_t(seqlen_k),
                                        int64_t(nheads), int64_t(batch));

        Tensor mQ_tiled = tma_load_Q.get_tma_tensor(
            cute::make_shape(int64_t(seqlen_q), int64_t(kHeadDim), int64_t(nheads), int64_t(batch)))
            (cute::_ , cute::_ , bidh, bidb);
        Tensor gQ = cute::local_tile(mQ_tiled, cute::select<0, 2>(TileShape_MNK{}),
                                     cute::make_coord(m_block, cute::_0{}));
        auto block_tma_Q = tma_load_Q.get_slice(cute::_0{});
        Tensor tQgQ = cute::group_modes<0, 3>(block_tma_Q.partition_S(gQ));
        Tensor tQsQ = cute::group_modes<0, 3>(block_tma_Q.partition_D(sQ));

        Tensor mK_tiled = tma_load_K.get_tma_tensor(
            cute::make_shape(int64_t(seqlen_k), int64_t(kHeadDim), int64_t(nheads), int64_t(batch)))
            (cute::_, cute::_, bidh, cute::_);
        Tensor gK_TMA = cute::local_tile(mK_tiled, cute::select<1, 2>(TileShape_MNK{}),
                                         cute::make_coord(cute::_, cute::_0{}, cute::_));
        auto block_tma_K = tma_load_K.get_slice(0);
        Tensor tKgK_TMA = cute::group_modes<0, 3>(block_tma_K.partition_S(gK_TMA));
        Tensor tKsK_TMA = cute::group_modes<0, 3>(block_tma_K.partition_D(sK));

        Tensor mV_tiled = tma_load_V.get_tma_tensor(shape_V)(cute::_, cute::_, bidh, cute::_);
        Tensor gVt_TMA = cute::local_tile(mV_tiled, cute::select<1, 2>(TileShape_MNK_PV{}),
                                          cute::make_coord(cute::_0{}, cute::_, cute::_));
        auto block_tma_V = tma_load_V.get_slice(0);
        Tensor tVgVt_TMA = cute::group_modes<0, 3>(block_tma_V.partition_S(gVt_TMA));
        Tensor tVsVt_TMA = cute::group_modes<0, 3>(block_tma_V.partition_D(sV));

        if (thread_idx == 0) {
            shared_storage.barrier_Q.arrive_and_expect_tx(TmaBytesQ);
            cute::copy(tma_load_Q.with(
                reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                    shared_storage.barrier_Q),
                uint16_t(0), cute::TMA::CacheHintSm90::EVICT_LAST),
                tQgQ, tQsQ);
        }

        int const n_block_max = (seqlen_k + kBlockN - 1) / kBlockN;
        PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
        for (int n_block = 0; n_block < n_block_max; ++n_block) {
            pipeline_k.producer_acquire(smem_pipe_write);
            if (thread_idx == 0) {
                cute::copy(tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write),
                            uint16_t(0), cute::TMA::CacheHintSm90::EVICT_LAST),
                    tKgK_TMA(cute::_, n_block, bidb), tKsK_TMA(cute::_, smem_pipe_write.index()));
            }
#ifndef KSKIPV
            pipeline_v.producer_acquire(smem_pipe_write);
            if (thread_idx == 0) {
                cute::copy(tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write),
                            uint16_t(0), cute::TMA::CacheHintSm90::EVICT_LAST),
                    tVgVt_TMA(cute::_, n_block, bidb), tVsVt_TMA(cute::_, smem_pipe_write.index()));
            }
#endif
            ++smem_pipe_write;
        }
        pipeline_k.producer_tail(smem_pipe_write);
        pipeline_v.producer_tail(smem_pipe_write);
        return;
    }

    // =========================================================================
    // CONSUMER warpgroup: Q in regs + WGMMA(QK) -> online softmax -> WGMMA(PV)
    // =========================================================================
#if KRUN_WGMMA
#endif

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;

    static constexpr int MmaWarpGroups = int(cute::size(TiledMmaPV{})) / NumThreadsPerWarpGroup;
    auto warp_group_thread_layout = cute::make_layout(
        cute::make_shape(cute::Int<MmaWarpGroups>{}),
        cute::make_stride(cute::Int<NumThreadsPerWarpGroup>{}));
    // The producer occupies warpgroup 0; the consumer is the first MMA warpgroup,
    // i.e. its local MMA index is warp_group_idx - 1 (== 0 here, since there is a
    // single MMA warpgroup). Using the global warp_group_idx would index past the
    // TiledMma partition and fault.
    int const mma_wg = warp_group_idx - 1;
    auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(mma_wg));
    auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(mma_wg));

    flash::Softmax<kNRows> softmax(softmax_scale_log2);
    Tensor tOrO = cute::partition_fragment_C(tiled_mma_pv, cute::select<0, 1>(TileShape_MNK_PV{}));
    cute::clear(tOrO);

    shared_storage.barrier_Q.wait(0);
    Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);
    {
        using SmemCopyAtomQ = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
        auto smem_tiled_copy_Q = cute::make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(warp_group_thread_idx);
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(cute::as_position_independent_swizzle_tensor(sQ));
        cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
    }

    Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);
    Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);
    Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);

    using SmemCopyAtomP = cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
    auto smem_tiled_copy_P = cute::make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_qk);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(warp_group_thread_idx);
    Tensor tPsP = smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP));

    Tensor cS_id = cute::make_identity_tensor(cute::select<0, 1>(TileShape_MNK{}));
    // 关键：必须用当前线程（thread_idx）的 MMA slice 来 partition_C，才能得到本线程
    // 负责的 (row, col) 坐标。若直接用 wg_mma_qk.partition_C（warpgroup 级 slice），
    // 返回的是整个 warpgroup 的分片，其 (mi, ni) 索引对应的是“其它线程”的元素坐标，
    // 会导致 causal mask 的 row/col 计算错误（non-causal 不用坐标故不受影响）。
    auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    Tensor taccScS = thread_mma_qk.partition_C(cS_id);
    Tensor taccScS_rowcol = cute::make_tensor(
        taccScS.data(), flash::convert_layout_acc_rowcol(taccScS.layout()));

    auto apply_mask = [&](auto& tSrS, int n_block) {
        Tensor scores = cute::make_tensor(
            tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
        #pragma unroll
        for (int mi = 0; mi < cute::size<0>(scores); ++mi) {
            int row = m_block * kBlockM + int(cute::get<0>(taccScS_rowcol(mi, cute::_0{})));
            #pragma unroll
            for (int ni = 0; ni < cute::size<1>(scores); ++ni) {
                int col = n_block * kBlockN + int(cute::get<1>(taccScS_rowcol(mi, ni)));
                if ((Is_causal && row < col) || col >= seqlen_k) {
                    scores(mi, ni) = -INFINITY;
                }
            }
        }
    };

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
        PipelineState flipped{smem_pipe_read.index(), smem_pipe_read.phase() ^ 1, smem_pipe_read.count()};
        auto barrier_token = pipeline.consumer_try_wait(flipped);
        pipeline.consumer_wait(flipped, barrier_token);
    };

    auto write_P_to_smem = [&](auto& tOrP) {
        cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP);
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(NumThreadsPerWarpGroup, /*id=*/2);
    };

    auto convert_S_to_P = [&](auto& tSrS, auto& tOrP) {
        Tensor tOrP_acc = cute::make_tensor(
            tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
        flash::convert_type_out(tOrP_acc, tOrP);
    };

    typename flash::Softmax<kNRows>::TensorT scores_scale;

    int const n_block_max = (seqlen_k + kBlockN - 1) / kBlockN;
    PipelineState smem_pipe_read = cutlass::make_producer_start_state<MainloopPipeline>();
    int n_block = 0;
    bool is_first = true;

    Tensor tSrS = cute::partition_fragment_C(tiled_mma_qk, cute::select<0, 1>(TileShape_MNK{}));

    // 正向遍历 n_block（0 -> n_block_max-1）。这样在 causal 下“真实可见”的小 n_block
    // tile 先处理并设好 row_max；完全被 mask 的大 n_block tile 后处理时 prev_max 已是
    // 真实值，online softmax 的 scores_scale 不会退化成 0 去清零已累积的 O。
    for (; n_block < n_block_max; ++n_block) {
        if (!is_first) { ++smem_pipe_read; }

        consumer_wait(pipeline_k, smem_pipe_read);

        // Causal 下，当整个 K tile 列都落在 Q 行的右下三角之外（即 col > row 对所有
        // 行成立）时，该 tile 会被完全 mask。正向遍历时该 tile 在最后处理，prev_max 已
        // 是真实值，online softmax 能安全处理；这里仍跳过以省一次 GEMM/PV，但仍消费
        // pipeline 以免死锁。
        bool tile_fully_masked = false;
        if constexpr (Is_causal) {
            // 本 tile 最小列 = n_block*kBlockN；Q 行最大行 = (m_block+1)*kBlockM-1。
            // 当 最小列 > Q 最大行 时，整 tile 所有 (row, col) 都满足 row < col，全被 mask。
            tile_fully_masked = (n_block * kBlockN >= (m_block + 1) * kBlockM);
        }

#if KRUN_WGMMA
        if (!tile_fully_masked) {
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
            tiled_mma_qk, tSrQ, tSrK(cute::_, cute::_, cute::_, smem_pipe_read.index()), tSrS);
        } else {
            cute::clear(tSrS);
        }
#else
        cute::clear(tSrS);
#endif
        cute::warpgroup_wait<0>();
        pipeline_k.consumer_release(smem_pipe_read);

        if (tile_fully_masked) {
            // 完全被 mask 的 tile：跳过 softmax/PV，但仍需消费 V 的 pipeline，
            // 否则 producer 会死锁（它已为该 n_block 发了 V）。
            // 注意：apply_mask 不在此调用，因为该 tile 全 mask，无需处理 scores。
#ifndef KSKIPV
            consumer_wait(pipeline_v, smem_pipe_read);
            cute::warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read);
#endif
            continue;
        }

        apply_mask(tSrS, n_block);

        if (is_first) {
            scores_scale = softmax.max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            softmax.online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
        } else {
            cute::copy(softmax.max_get_scale</*Is_first=*/false, /*Check_inf=*/true>(tSrS), scores_scale);
            softmax.online_softmax</*Is_first=*/false, /*Check_inf=*/true>(tSrS);
            softmax.rescale_o(tOrO, scores_scale);
        }
        is_first = false;  // end of first iteration; subsequent tiles rescale O

        Tensor tOrP = cute::make_tensor_like<Element>(cute::make_tensor(
            tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout())));
        convert_S_to_P(tSrS, tOrP);
        write_P_to_smem(tOrP);

#ifndef KSKIPV
        consumer_wait(pipeline_v, smem_pipe_read);
#if KRUN_WGMMA
        if (is_first) {
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
                tiled_mma_pv, tOsP, tOrV(cute::_, cute::_, cute::_, smem_pipe_read.index()), tOrO);
        } else {
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
                tiled_mma_pv, tOsP, tOrV(cute::_, cute::_, cute::_, smem_pipe_read.index()), tOrO);
        }
#else
        cute::clear(tOrO);
#endif
        cute::warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read);
#else
        cute::clear(tOrO);
#endif
    }

    scores_scale = softmax.finalize(1.0f);
    softmax.rescale_o(tOrO, scores_scale);

    Tensor tOrO_fp16 = cute::make_tensor_like<Element>(tOrO);
    flash::convert_type_out(tOrO, tOrO_fp16);
    auto smem_tiled_copy_O = cute::make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_pv);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(warp_group_thread_idx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_fp16);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    Tensor mO_tiled = tma_store_O.get_tma_tensor(
        cute::make_shape(int64_t(seqlen_q), int64_t(kHeadDim), int64_t(nheads), int64_t(batch)))
        (cute::_, cute::_, bidh, bidb);
    Tensor gO = cute::local_tile(mO_tiled, cute::select<0, 1>(TileShape_MNK_PV{}),
                                 cute::make_coord(m_block, cute::_0{}));
    auto block_tma_O = tma_store_O.get_slice(cute::_0{});
    Tensor tOgO = block_tma_O.partition_D(gO);
    Tensor tOsO = block_tma_O.partition_S(sO);
    // TMA store 的 leader 必须是“本 warpgroup 内的 0 号线程”。consumer 是 warpgroup 1，
    // 全局线程号是 128..255，所以不能用 thread_idx==0（在 consumer 里恒为假），
    // 必须用 warp_group_thread_idx==0 —— 它在 wg0 映射到全局 thread0、在 wg1 映射到全局 thread128。
    if (warp_group_thread_idx == 0) {
        cute::copy(tma_store_O, tOsO, tOgO);
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
    }
}

}  // namespace fa3
