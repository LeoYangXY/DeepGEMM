import argparse
import time

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import JitArguments, benchmark
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineTmaAsync,
    PipelineUserType,
    make_pipeline_state,
)


class GemmWmmaTmaPipelineTunable:
    """TMA + software pipeline + Tensor Core GEMM with optional CTA swizzle."""

    def __init__(
        self,
        cta_tiler=(128, 128, 64),
        num_stages=2,
        cta_swizzle_group=1,
    ):
        self.tile_shape_mnk = cta_tiler
        self._bM, self._bN, self._bK = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self.warp_size = cute.arch.WARP_SIZE

        self.num_mma_warps = self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self.mma_warp_ids = tuple(range(self.num_mma_warps))
        self.tma_warp_id = self.num_mma_warps
        self.threads_per_cta = self.warp_size * (self.num_mma_warps + 1)

        self.num_stages = num_stages
        self.buffer_align_bytes = 1024
        self.cta_swizzle_group = max(1, int(cta_swizzle_group))

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype,
            num_stages=self.num_stages,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype,
            num_stages=self.num_stages,
        )

        tma_atom_a, tma_tensor_a = self._make_tma(
            a, self.a_smem_layout_staged, (self._bM, self._bK)
        )
        tma_atom_b, tma_tensor_b = self._make_tma(
            b, self.b_smem_layout_staged, (self._bN, self._bK)
        )

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            shape_mnk=self.mma_inst_shape,
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )

        @cute.struct
        class SharedStorage:
            pipeline_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        grid_m = (c.shape[0] + self._bM - 1) // self._bM
        grid_n = (c.shape[1] + self._bN - 1) // self._bN

        grid_dim = (grid_m, grid_n, 1)
        if self.cta_swizzle_group > 1:
            swizzle_group = self.cta_swizzle_group
            grid_x = grid_n // swizzle_group
            grid_y = grid_m
            grid_z = swizzle_group
            grid_dim = (grid_x, grid_y, grid_z)

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tiled_mma,
            c,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            grid_n,
        ).launch(grid=grid_dim, block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def kernel(
        self,
        tma_atom_a,
        mA_mk,
        tma_atom_b,
        mB_nk,
        tiled_mma,
        mC,
        a_smem_layout_staged,
        b_smem_layout_staged,
        total_n_tiles,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        is_mma_warp = warp_idx <= self.mma_warp_ids[-1]
        is_tma_warp = warp_idx == self.tma_warp_id

        if is_tma_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        m_idx = bidx
        n_idx = bidy
        if self.cta_swizzle_group > 1:
            gdimx, _, _ = cute.arch.grid_dim()
            m_idx = bidy
            n_idx = bidz * gdimx + bidx

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        gA = cute.local_tile(mA_mk, self.tile_shape_mnk, (m_idx, n_idx, None), proj=(1, None, 1))
        gB = cute.local_tile(mB_nk, self.tile_shape_mnk, (m_idx, n_idx, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.tile_shape_mnk, (m_idx, n_idx, None), proj=(1, 1, None))

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        thr_mma = tiled_mma.get_slice(tid)
        tCgC = thr_mma.partition_C(gC)

        sA_0 = cute.slice_(sA, (None, None, 0))
        sB_0 = cute.slice_(sB, (None, None, 0))
        tCsA_0 = thr_mma.partition_A(sA_0)
        tCsB_0 = thr_mma.partition_B(sB_0)
        tCrA = tiled_mma.make_fragment_A(tCsA_0)
        tCrB = tiled_mma.make_fragment_B(tCsB_0)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        atom_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.a_dtype,
        )
        atom_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.b_dtype,
        )
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_s2r_B.get_slice(tid)
        tCrA_copy_view = thr_s2r_A.retile(tCrA)
        tCrB_copy_view = thr_s2r_B.retile(tCrB)

        tma_transaction_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        )

        mainloop_pipeline = PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, self.num_mma_warps),
            barrier_storage=storage.pipeline_mbar_ptr.data_ptr(),
            tx_count=tma_transaction_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        producer_state = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        consumer_state = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)

        num_k_tiles = mA_mk.shape[1] // self._bK
        prefetch_tiles = min(self.num_stages, num_k_tiles)

        if is_tma_warp:
            for _ in range(prefetch_tiles):
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[None, producer_state.count],
                    tAsA[None, producer_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[None, producer_state.count],
                    tBsB[None, producer_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                )
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

            for _ in range(prefetch_tiles, num_k_tiles):
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[None, producer_state.count],
                    tAsA[None, producer_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[None, producer_state.count],
                    tBsB[None, producer_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                )
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        if is_mma_warp:
            tCrC.fill(0.0)

            for _ in range(num_k_tiles):
                mainloop_pipeline.consumer_wait(consumer_state)

                sA_stage = cute.slice_(sA, (None, None, consumer_state.index))
                sB_stage = cute.slice_(sB, (None, None, consumer_state.index))

                tCsA_copy_view = thr_s2r_A.partition_S(sA_stage)
                tCsB_copy_view = thr_s2r_B.partition_S(sB_stage)

                cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
                cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
                cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
            tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
            for i in range(cute.size(tCrC_out)):
                tCrC_out[i] = cutlass.Float16(tCrC[i])
            cute.copy(atom_store, tCrC_out, tCgC)

    @staticmethod
    def _make_tma(tensor, smem_layout_staged, smem_tile):
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile)


def _run_profile_range(kernel, args):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    kernel(*args)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _normalize_swizzle_group(N, tile_n, requested_group):
    if requested_group <= 1:
        return 1

    grid_n = (N + tile_n - 1) // tile_n
    g = min(int(requested_group), int(grid_n))
    while g > 1 and (grid_n % g) != 0:
        g -= 1
    return g


def _compile_and_check(M, N, K, cta_tiler, num_stages, cta_swizzle_group, atol=1e-1, rtol=1e-1):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    ref = torch.matmul(A, B.T)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    normalized_swizzle_group = _normalize_swizzle_group(
        N=N,
        tile_n=cta_tiler[1],
        requested_group=cta_swizzle_group,
    )

    kernel_obj = GemmWmmaTmaPipelineTunable(
        cta_tiler=cta_tiler,
        num_stages=num_stages,
        cta_swizzle_group=normalized_swizzle_group,
    )
    kernel = cute.compile(kernel_obj, A_, B_, C_)
    kernel(A_, B_, C_)

    ok = torch.allclose(C, ref, atol=atol, rtol=rtol)
    max_diff = (C.float() - ref.float()).abs().max().item()

    return kernel, (A_, B_, C_), ok, max_diff


def run_case(name, M, N, K, cta_tiler, num_stages, cta_swizzle_group, do_profile_range=False):
    effective_swizzle_group = _normalize_swizzle_group(
        N=N,
        tile_n=cta_tiler[1],
        requested_group=cta_swizzle_group,
    )

    t0 = time.time()
    kernel, kernel_args, ok, max_diff = _compile_and_check(
        M,
        N,
        K,
        cta_tiler=cta_tiler,
        num_stages=num_stages,
        cta_swizzle_group=effective_swizzle_group,
    )
    compile_and_check_ms = (time.time() - t0) * 1000

    elapsed_us = benchmark(kernel, kernel_arguments=JitArguments(*kernel_args))

    flops = 2 * M * N * K
    tflops = flops / (elapsed_us * 1e6)

    print("=" * 90, flush=True)
    print(
        f"[{name}] tile={cta_tiler}, stages={num_stages}, cta_swizzle_group={effective_swizzle_group}",
        flush=True,
    )
    print(
        f"[{name}] compile+check={compile_and_check_ms:.2f} ms | valid={ok} | max_diff={max_diff:.4f}",
        flush=True,
    )
    print(f"[{name}] latency={elapsed_us:.2f} us | throughput={tflops:.2f} TFLOPS", flush=True)

    if do_profile_range:
        _run_profile_range(kernel, kernel_args)
        print(f"[{name}] profile range executed (cudaProfilerStart/Stop).", flush=True)

    return {
        "name": name,
        "ok": ok,
        "max_diff": max_diff,
        "latency_us": elapsed_us,
        "tflops": tflops,
        "cta_tiler": cta_tiler,
        "num_stages": num_stages,
        "cta_swizzle_group": effective_swizzle_group,
    }


def main():
    parser = argparse.ArgumentParser("cutedsl TMA pipeline + CTA swizzle tuning demo")
    parser.add_argument("--mode", choices=["bench", "profile"], default="bench")
    parser.add_argument("--profile-target", choices=["baseline", "tuned"], default="baseline")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)

    parser.add_argument("--baseline-tile", nargs=3, type=int, default=[128, 128, 64])
    parser.add_argument("--baseline-stages", type=int, default=2)

    parser.add_argument("--tuned-tile", nargs=3, type=int, default=[128, 64, 64])
    parser.add_argument("--tuned-stages", type=int, default=2)
    parser.add_argument("--tuned-cta-swizzle-group", type=int, default=4)

    args = parser.parse_args()

    cc = torch.cuda.get_device_capability()
    if cc < (9, 0):
        print(f"SM90+ required, got SM{cc[0]}{cc[1]}")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)} | CC={cc}", flush=True)
    print(f"Problem size: M={args.M}, N={args.N}, K={args.K}", flush=True)

    if args.mode == "bench":
        baseline = run_case(
            name="baseline",
            M=args.M,
            N=args.N,
            K=args.K,
            cta_tiler=tuple(args.baseline_tile),
            num_stages=args.baseline_stages,
            cta_swizzle_group=1,
            do_profile_range=False,
        )

        tuned = run_case(
            name="tuned",
            M=args.M,
            N=args.N,
            K=args.K,
            cta_tiler=tuple(args.tuned_tile),
            num_stages=args.tuned_stages,
            cta_swizzle_group=args.tuned_cta_swizzle_group,
            do_profile_range=False,
        )

        if baseline["ok"] and tuned["ok"]:
            speedup = baseline["latency_us"] / tuned["latency_us"]
            print("=" * 90, flush=True)
            print(f"Speedup (baseline/tuned): {speedup:.4f}x", flush=True)
        else:
            print("One of the configurations failed numerical check.", flush=True)

    else:
        if args.profile_target == "baseline":
            run_case(
                name="baseline",
                M=args.M,
                N=args.N,
                K=args.K,
                cta_tiler=tuple(args.baseline_tile),
                num_stages=args.baseline_stages,
                cta_swizzle_group=1,
                do_profile_range=True,
            )
        else:
            run_case(
                name="tuned",
                M=args.M,
                N=args.N,
                K=args.K,
                cta_tiler=tuple(args.tuned_tile),
                num_stages=args.tuned_stages,
                cta_swizzle_group=args.tuned_cta_swizzle_group,
                do_profile_range=True,
            )


if __name__ == "__main__":
    main()
