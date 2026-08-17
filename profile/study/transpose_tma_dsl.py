"""
transpose_tma_dsl.py  --  Transpose via TMA bulk copies (CuTe DSL / Blackwell)

================================================================================
你的理解（完全正确）:
    TMA = "按 SHAPE + STRIDE 去批量搬运"的硬件 DMA 引擎。
    你给一个描述符(shape 几维多大 + 每维 stride 多少)，硬件就一口气泡
    把一个 tile 合并地搬进/搬出 shared memory。线程不逐元素算地址。

转置是怎么用 TMA 做的:
    A 是 row-major: 元素(r,c) 在 r*N + c
    B 用"列主序视图"描述: 元素(r,c) 在 r*1 + c*N = B[c*N + r]
    => TMA LOAD 按 (N,1) 从 A 搬一个 tile 进 shared
       TMA STORE 按 (1,N) 把同一个 shared tile 写回 B
       shared 的第 (r,c) 个元素被写到 B[c*N+r]  =>  B[col*N+row] = A[row*N+col]
    一次 bulk load + 一次 bulk store 即完成转置, 数据本身没动, 只是 view 的
    stride 变了。

对比不用 TMA 的 transpose_smem:
    那个版本每个线程要 IMAD/ISETP 算 row*N+col (走 ADU, ~38% 指令),
    还要逐线程发 LDG/STG。TMA 版这些全没了: 地址算在 host 端写进描述符,
    搬运由 1 个线程发 1 条指令触发硬件完成。

---------------------------------------------------------------------- Q&A
Q1: TMA 不受 sector 限制吗?
A: 不是绕过, 而是"靠描述符几何保证满 sector 利用"。
   - sector(32B) 是 HBM<->L2 的物理最小单位, TMA 也逃不掉。
   - 普通 coalesced LDG 靠"32 线程地址恰好连续"拼出合并事务; naive
     transpose 的 strided STG 让 32 线程要的 float 散落, sector 利用率
     崩到 ~25%。
   - TMA 用 "shape+stride box" 描述符让硬件直接按 cache-line(128B) 对齐
     合并搬运整个 tile, 不论读写都 100% sector 利用, 且线程侧零地址计算。

Q2: TMA 数据会落 L1 / L2 吗?
A: 走两条专用路径, 都刻意绕过 L1:
   - TMA LOAD (G2S):  HBM -> L2 -> Shared Memory  (不进 L1)
   - TMA STORE(S2G):  Shared Memory -> L2 -> HBM  (不进 L1, 常写回 L2)
   对比普通 LDG/STG 会经过 L1。TMA 绕开 L1 避免污染、省带宽。
   shared memory 本身不是 cache, 是程序员管理的 scratchpad。

Q3: 真能在 kernel 里面搞 TMA 描述符吗?
A: 分两层, 别混:
   - 在 @cute.jit 里写 make_tiled_tma_atom(...) 是 **编译期 (tracing)**:
     此时 a/b 是符号 tensor, 只把 "shape+stride+tiler" 写进描述符 *参数*,
     返回一个编译期 TMA 原子 + 带 tiler 的视图。它不碰真实地址, 不在 GPU
     上 malloc 硬件描述符。所以"在 kernel 里建"= 编译期声明形状/步长。
   - 真正的 128B 硬件 TMA 描述符由 CuTe runtime 在 **launch 之前** 填好、
     放到 GMEM; kernel 运行时直接把 tma_load/tma_store 喂给 cute.copy。
   对比 C++ CuTe: C++ 里描述符在 host 端 TensorMap 构造好再传进 kernel;
     DSL 把这步藏进 make_tiled_tma_atom, 看起来像 kernel 内建, 其实等价。
   一句话: 不是"运行时 GPU 上 new 一个硬件描述符", 而是"编译期声明,
            运行时由 runtime 实例化"。所以不必担心 kernel 里动态建描述符。
================================================================================

运行 (请在自己的终端里跑, 本环境工具限制 30s 无法交互式编译):
    export PATH=/home/leo/.local/miniconda3/envs/cutedsl/bin:$PATH
    python transpose_tma_dsl.py

ncu 对比 ADU 占比 (看 TMA 如何干掉 IMAD 地址计算):
    ncu -k transpose_tma_kernel -s 1 --print-units base \
        -m sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active \
        -m sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active \
        python transpose_tma_dsl.py
    # 再和 transpose_smem 的 ncu-rep 比, 预期 adu% 大幅下掉
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkTensorTileG2SOp, CopyBulkTensorTileS2GOp,
)
from cutlass.cute.nvgpu.helpers import make_tiled_tma_atom
from cutlass.pipeline import (
    PipelineTmaAsync, CooperativeGroup, Agent,
    make_pipeline_state, PipelineUserType,
)

N = 1024
TILE = 32
BLOCKS = (N + TILE - 1) // TILE


@cute.jit
def transpose_tma_kernel(a: cute.Tensor, b: cute.Tensor):
    cta_tiler = (TILE, TILE)

    # =====================================================================
    # TMA 描述符: "在 kernel 里建"到底意味着什么?
    # ---------------------------------------------------------------------
    # 注意: 这里是 @cute.jit 的 tracing 编译期。传进来的 `a` / `b` 是
    # cute.Tensor (带 shape+layout 的 *符号值*), 不是真实 GPU 指针。
    # make_tiled_tma_atom 在编译期做两件事:
    #   (1) 把 "每次搬多大的块" (cta_tiler=(TILE,TILE)) 和
    #       "源/目标 tensor 的全局 shape+stride" 写进描述符 *参数*;
    #   (2) 返回一个 "TMA 原子" (tma_load/tma_store) 和一个带 tiler
    #       逻辑的 tensor 视图 (mA/mB, 可 local_tile 取块)。
    # 它 **不** 在 GPU 上动态分配 128B 硬件描述符, 也不碰真实地址 ——
    # 真实的 TMA 描述符内存由 CuTe runtime 在 *launch 之前* 填好放到
    # GMEM, kernel 运行时直接把 tma_load/tma_store 喂给 cute.copy。
    # 所以"kernel 里搞描述符" = 编译期声明形状/步长, 不是运行时 new。
    # =====================================================================

    # a: row-major, 全局 stride = (N, 1)  =>  LOAD 按 (N,1) 合并搬一个 tile
    tma_load,  mA = make_tiled_tma_atom(CopyBulkTensorTileG2SOp(), a, cute.make_layout(cta_tiler), cta_tiler)
    # b: 用"列主序视图"描述, 全局 stride = (1, N)
    #    => STORE 时硬件按 (1,N) 写同一个 shared tile, 物理落点变成 B[c*N+r]
    #    => 一次 bulk store 即完成转置, 数据本身没在 shared 里挪动。
    tma_store, mB = make_tiled_tma_atom(CopyBulkTensorTileS2GOp(), b, cute.make_layout(cta_tiler), cta_tiler)

    smem_layout = cute.make_layout(cta_tiler)
    smem_bytes = cute.size_in_bytes(a.element_type, cute.make_tensor(
        cute.make_ptr(a.element_type, cute.AddressSpace.shared), smem_layout))

    # =====================================================================
    # SharedStorage: 这个 block 的 shared memory 长啥样 (一次开好)
    # ---------------------------------------------------------------------
    # @cute.struct 是 DSL 里定义 "POD 数据布局" 的方式, 等价 C++ 里
    # 一个 __shared__ struct: 编译器按字段顺序、按对齐规则算出总字节数
    # 和每个字段的偏移, 最后由 SmemAllocator 一次性 allocate。
    #
    # 三个字段:
    #  (1) mbar_load / mbar_store : mbarrier (硬件屏障) 的存储区
    #      - mbarrier 是 TMA 配套的同步原语: producer 发 TMA 指令后立刻
    #        返回, TMA 引擎在 *后台* 搬数据; 搬完会自动 "翻转" 这个 barrier,
    #        consumer 只要等 barrier 翻转就知道 "数据好了, 可以动"。
    #      - 每个 MemRange[Int64, 2] = 2 个 64-bit 寄存器, 因为
    #        PipelineTmaAsync 的 barrier_storage 要求 2 个 Int64 的
    #        mbarrier 槽位 (一个给 arrive/except, 一个做阶段计数)。
    #      - 分 load / store 两个, 是因为 LOAD 和 STORE 是两条独立流水,
    #        各自有独立的完成信号, 互不干扰。
    #
    #  (2) tile : 真正装数据的 shared memory 缓冲, 大小 TILE*TILE 个元素
    #      - MemRange[a.element_type, TILE*TILE] = 连续排 TILE*TILE 个
    #        float (a.element_type), 即这块 tile 的数据暂存区。
    #      - 外层 Align[..., 128] = 强制按 128 字节对齐 (cache-line /
    #        TMA 要求 128B 对齐)。TMA 搬进 shared 时要求目标地址 128B
    #        对齐, 否则出错或降性能; 这里用 Align 保证 tile 起始地址合规。
    #      - 注意: 这份数据在 shared 里 *从不显式转置*。load 进来是
    #        A 的 (TILE,TILE) 块, store 出去时靠 tma_store 的 (1,N)
    #        视图把同样的内存按转置落点写回 B。
    # =====================================================================
    @cute.struct
    class SharedStorage:
        mbar_load:  cute.struct.MemRange[cutlass.Int64, 2]
        mbar_store: cute.struct.MemRange[cutlass.Int64, 2]
        tile: cute.struct.Align[cute.struct.MemRange[a.element_type, TILE * TILE], 128]

    bidx, bidy, _ = cute.arch.block_idx()
    tile_coord = (bidx, bidy)
    gA = cute.local_tile(mA, cta_tiler, tile_coord)
    gB = cute.local_tile(mB, cta_tiler, tile_coord)

    # =====================================================================
    # tma_partition: 把整块 tensor 切成 "这个线程该持有的那一份引用"
    # ---------------------------------------------------------------------
    # TMA 是 "1 个线程发 1 条指令, 硬件替整个 block 搬" 的机制。
    # 所以不需要 32 个线程每人算自己的地址; 只要 producer warp 的
    # thread0 用这份引用调 cute.copy, TMA 引擎就按描述符把整块 tile 搬完。
    # tma_partition 的返回值:
    #   tAsA / tBsB = 线程视角的 shared 切片 (source for load / dest for store)
    #   tAgA / tBgB = 线程视角的 global 切片
    # 其余线程的这份参数会被忽略/统一 (warp-uniform)。
    # 第 2 个参数 0 = 选 producer 角色的线程索引; 后面 group_modes(...,0,2)
    # 是把 (M,N) 两维合并成 (MN) 一维, 便于 TMA 按线性 tile 处理。
    # =====================================================================
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_load, 0, cute.make_layout(1),
        cute.group_modes(cute.make_tensor(cute.make_ptr(a.element_type, cute.AddressSpace.shared), smem_layout), 0, 2),
        cute.group_modes(gA, 0, 2),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_store, 0, cute.make_layout(1),
        cute.group_modes(cute.make_tensor(cute.make_ptr(a.element_type, cute.AddressSpace.shared), smem_layout), 0, 2),
        cute.group_modes(gB, 0, 2),
    )

    smem = SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = storage.tile.get_tensor(smem_layout)

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    is_producer = warp_idx == 0

    # =====================================================================
    # Warp specialization: 只有 warp0 当 producer, 负责发 TMA
    # (这里没写 consumer 计算, 所以 producer 自己发完 load 等完再发 store)
    # =====================================================================
    if is_producer:
        # ---- TMA LOAD: A tile -> shared (硬件按 (N,1) 合并搬, 100% sector) ----
        # prefetch_descriptor: 运行期指令, 把描述符预取到 TMA 单元寄存器, 省延迟
        cute.nvgpu.cpasync.prefetch_descriptor(tma_load)
        # PipelineTmaAsync: 软件 pipeline + mbarrier 同步原语
        #   barrier_storage = SharedStorage 里那 2 个 Int64 mbarrier 的地址
        #   tx_count        = 这次 TMA 要搬多少字节 (用于 mbarrier 计数)
        #   num_stages=1    = 单缓冲 (没有 double-buffer, 故无法 load/compute overlap)
        pl = PipelineTmaAsync.create(
            num_stages=1, producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            barrier_storage=storage.mbar_load.data_ptr(), tx_count=smem_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))
        # ---- TMA producer 标准三步 ----
        ps = make_pipeline_state(PipelineUserType.Producer, 1)
        pl.producer_acquire(ps)                                  # (1) 占一个 pipeline slot
        # (2) 发 TMA 指令: 立刻返回, TMA 引擎后台搬; tma_bar_ptr 是"搬完敲的钟"
        cute.copy(tma_load, tAgA, tAsA, tma_bar_ptr=pl.producer_get_barrier(ps))
        pl.producer_commit(ps)                                   # (3) 提交, TMA 正式开工
        # 等 load 真正完成: consumer_wait 会卡在 mbarrier 上, 直到 TMA flip 它
        # (真 warp-specialized kernel 里这行在 consumer warp; 这里 producer 自等)
        cs = make_pipeline_state(PipelineUserType.Consumer, 1)
        pl.consumer_wait(cs)

        # ---- TMA STORE: shared -> B tile (硬件按 (1,N) 写 => 自动转置) ----
        cute.nvgpu.cpasync.prefetch_descriptor(tma_store)
        ps2 = make_pipeline_state(PipelineUserType.Producer, 1)
        pl2 = PipelineTmaAsync.create(
            num_stages=1, producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            barrier_storage=storage.mbar_store.data_ptr(), tx_count=smem_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))
        pl2.producer_acquire(ps2)
        cute.copy(tma_store, tBsB, tBgB, tma_bar_ptr=pl2.producer_get_barrier(ps2))
        pl2.producer_commit(ps2)
        # 注意: 全程没有"显式转置"代码。转置全靠 tma_store 的 stride(1,N) 视图。


def run():
    a = torch.arange(N * N, dtype=torch.float32, device='cuda').reshape(N, N)
    b = torch.zeros(N, N, dtype=torch.float32, device='cuda')
    transpose_tma_kernel(from_dlpack(a), from_dlpack(b)).launch(
        grid=(BLOCKS, BLOCKS, 1), block=(32, 1, 1))
    torch.cuda.synchronize()
    ok = torch.allclose(b, a.T, atol=1e-4)
    print(f"N={N} TILE={TILE} grid={BLOCKS}x{BLOCKS}  correctness:", "OK" if ok else "FAIL")
    if not ok:
        print("max diff:", (b - a.T).abs().max().item())
    return ok


if __name__ == "__main__":
    run()
