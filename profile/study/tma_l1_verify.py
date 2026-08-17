"""
tma_l1_verify.py  -- 用 ncu 验证 "TMA G2S 不进 L1" (CuTe DSL / Blackwell sm_120)

为什么需要这个 demo:
    口头说 "TMA 绕过 L1" 不够, 我们用 ncu 计数器实锤:
      - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   (L1 上 global load 的 sector 数)
      - lts__t_sectors_srcunit_tex_op_read.sum          (L2 侧读 sector 数, 应 >0 说明确实过了 L2)
    命题: TMA kernel 的 L1 sector 应 ~0, 而普通 __ldg kernel 的 L1 sector 应很大。

两个 kernel 同做 transpose, 只差 input 搬运方式:
    transpose_ldg_kernel : 普通 cute.copy (AutoCopy) 合并读 A 进 shared  -> 走 L1 (LSU pipe)
    transpose_tma_kernel : TMA bulk load A 进 shared                     -> 绕过 L1

运行 + ncu 对比 (在你自己的终端):
    export PATH=/home/leo/.local/miniconda3/envs/cutedsl/bin:$PATH
    python tma_l1_verify.py

    # --- LDG 版 L1 应很大 ---
    ncu --kernel-name transpose_ldg_kernel \\
        -m l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \\
        -m lts__t_sectors_srcunit_tex_op_read.sum \\
        python tma_l1_verify.py

    # --- TMA 版 L1 应 ~0 ---
    ncu --kernel-name transpose_tma_kernel \\
        -m l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \\
        -m lts__t_sectors_srcunit_tex_op_read.sum \\
        python tma_l1_verify.py

预期:
    transpose_ldg_kernel : l1tex__t_sectors... > 0   (合并读, sector 利用率高但确实过 L1)
    transpose_tma_kernel : l1tex__t_sectors... == 0  (TMA G2S 直连 shared, 不进 L1)
    两者 lts__t_sectors_srcunit_tex_op_read 都 > 0   (都从 L2/HBM 取数)
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkTensorTileG2SOp,
)
from cutlass.cute.nvgpu.helpers import make_tiled_tma_atom

N = 1024
TILE = 32
BLOCKS = (N + TILE - 1) // TILE


# ---------- 普通 __ldg 合并读版 (进 L1) ----------
@cute.jit
def transpose_ldg_kernel(a: cute.Tensor, b: cute.Tensor):
    cta_tiler = (TILE, TILE)
    bidx, bidy, _ = cute.arch.block_idx()
    tidx, tidy = cute.arch.thread_idx().x, cute.arch.thread_idx().y

    # shared tile
    smem_layout = cute.make_layout(cta_tiler)
    @cute.struct
    class S:
        tile: cute.struct.Align[cute.struct.MemRange[a.element_type, TILE * TILE], 128]
    smem = cutlass.utils.SmemAllocator().allocate(S)
    sA = smem.tile.get_tensor(smem_layout)

    # 合并读: 每线程沿 row 连续取 (block 内 tile 起点 + 线程偏移)
    base_r = bidy * TILE
    base_c = bidx * TILE
    for j in range(0, TILE, 4):                 # 简单展平, 关键是用合并 load
        for i in range(0, TILE, 8):
            r = base_r + tidy * 8 + i
            c = base_c + tidx * 4 + j
            if r < N and c < N:
                # __ldg / 合并 load -> 走 L1
                sA[tidy * 8 + i, tidx * 4 + j] = a[r * N + c]
    cute.arch.sync_threads()

    # 转置写回 global (这里只验证 load 侧, 写回随便)
    for j in range(0, TILE, 4):
        for i in range(0, TILE, 8):
            r = base_r + tidy * 8 + i
            c = base_c + tidx * 4 + j
            if r < N and c < N:
                b[c * N + r] = sA[tidy * 8 + i, tidx * 4 + j]
    cute.arch.sync_threads()


# ---------- TMA bulk load 版 (绕过 L1) ----------
@cute.jit
def transpose_tma_kernel(a: cute.Tensor, b: cute.Tensor):
    cta_tiler = (TILE, TILE)
    tma_load, mA = make_tiled_tma_atom(
        CopyBulkTensorTileG2SOp(), a, cute.make_layout(cta_tiler), cta_tiler)

    smem_layout = cute.make_layout(cta_tiler)
    smem_bytes = cute.size_in_bytes(
        a.element_type,
        cute.make_tensor(cute.make_ptr(a.element_type, cute.AddressSpace.shared), smem_layout))

    @cute.struct
    class SharedStorage:
        mbar: cute.struct.MemRange[cutlass.Int64, 1]
        tile: cute.struct.Align[cute.struct.MemRange[a.element_type, TILE * TILE], 128]

    bidx, bidy, _ = cute.arch.block_idx()
    tile_coord = (bidx, bidy)
    gA = cute.local_tile(mA, cta_tiler, tile_coord)

    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_load, 0, cute.make_layout(1),
        cute.group_modes(cute.make_tensor(
            cute.make_ptr(a.element_type, cute.AddressSpace.shared), smem_layout), 0, 2),
        cute.group_modes(gA, 0, 2))

    smem = cutlass.utils.SmemAllocator().allocate(SharedStorage)
    sA = smem.tile.get_tensor(smem_layout)

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_load)
        # 单 stage, producer 即本 warp
        pl = cutlass.pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
            consumer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
            barrier_storage=smem.mbar.data_ptr(), tx_count=smem_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))
        ps = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        pl.producer_acquire(ps)
        cute.copy(tma_load, tAgA, tAsA, tma_bar_ptr=pl.producer_get_barrier(ps))
        pl.producer_commit(ps)
        cs = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        pl.consumer_wait(cs)
        # 转置写回 (简化: 直接写, 这里只验证 load 不进 L1)
        bidx2, bidy2, _ = cute.arch.block_idx()
        # 把 sA 以转置 view 写回 B
        for j in range(0, TILE, 4):
            for i in range(0, TILE, 8):
                r = bidy2 * TILE + i
                c = bidx2 * TILE + j
                if r < N and c < N:
                    b[c * N + r] = sA[i, j]


def run():
    a = torch.arange(N * N, dtype=torch.float32, device='cuda').reshape(N, N)
    b_ldg = torch.zeros(N, N, dtype=torch.float32, device='cuda')
    b_tma = torch.zeros(N, N, dtype=torch.float32, device='cuda')

    transpose_ldg_kernel(from_dlpack(a), from_dlpack(b_ldg)).launch(
        grid=(BLOCKS, BLOCKS, 1), block=(8, 4, 1))
    transpose_tma_kernel(from_dlpack(a), from_dlpack(b_tma)).launch(
        grid=(BLOCKS, BLOCKS, 1), block=(32, 1, 1))
    torch.cuda.synchronize()

    ok_ldg = torch.allclose(b_ldg, a.T, atol=1e-4)
    ok_tma = torch.allclose(b_tma, a.T, atol=1e-4)
    print(f"ldg correctness: {'OK' if ok_ldg else 'FAIL'} | "
          f"tma correctness: {'OK' if ok_tma else 'FAIL'}")
    print("now run ncu on each kernel to compare l1tex__t_sectors...")


if __name__ == "__main__":
    run()
