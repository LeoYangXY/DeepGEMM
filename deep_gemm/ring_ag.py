"""Ring AllGather + GEMM 通算融合 kernel 的 Python 封装。

设计要点见 `deep_gemm/include/deep_gemm/impls/sm90_bf16_ring_ag_gemm.cuh` 的注释。

这里主要负责：
    1. 用 CUDA IPC 建立"对称显存"（每个 rank 一块同样大小的 buffer，互相映射），
       等价于 DeepGEMM mega kernel 里用的 torch symmetric memory，
       但不依赖 torch >= 2.6；
    2. 维护 ring 通信需要的接收区 / 到达 flag / 消费计数；
    3. 维护 epoch，使 flag 可以单调累加而不需要每次 memset。
"""

from typing import Optional

import torch
import torch.distributed as dist

from . import _C

__all__ = ['RingAllGatherGemm']


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class RingAllGatherGemm:
    """ring all-gather + GEMM 的上下文（持有对称显存，可反复调用）。

    语义：
        输入 a_local [Mp, K]（本 rank 的 A 分片）、b [N, K]（本 rank 的权重）
        输出 d [M, N]，M = world_size * Mp，其中 d[c*Mp:(c+1)*Mp] 对应 rank c 的分片。
    """

    # 接收区的双缓冲组数，避免"跑得快的卡覆盖慢卡还在读的数据"
    NUM_SLOTS = 2

    def __init__(self, shape_mp: int, shape_k: int,
                 group: Optional[dist.ProcessGroup] = None,
                 dtype: torch.dtype = torch.bfloat16):
        assert dtype == torch.bfloat16, '目前只实现了 BF16'
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.shape_mp = shape_mp
        self.shape_k = shape_k
        self.device = torch.cuda.current_device()
        self.epoch = 0

        elem_size = torch.tensor([], dtype=dtype).element_size()
        num_recv_chunks = max(self.world_size - 1, 1)

        # 对称显存的内部布局（每个 rank 一模一样）：
        #   [0, slot_bytes * NUM_SLOTS)  接收区，NUM_SLOTS 组双缓冲
        #   [flag_offset, +4*(R-1))      每个接收 slot 的到达计数（单调累加）
        #   [consume_offset, +4)         本 rank 已经完成的 epoch 计数（单调累加）
        self.slot_bytes = _align_up(num_recv_chunks * shape_mp * shape_k * elem_size, 1024)
        self.flag_offset = self.slot_bytes * self.NUM_SLOTS
        self.consume_offset = self.flag_offset + _align_up(num_recv_chunks * 4, 256)
        self.total_bytes = self.consume_offset + 256

        self.self_base = _C.sym_alloc(self.total_bytes)
        self.peer_base = self.self_base
        self._opened = False

        if self.world_size > 1:
            # 交换 IPC handle：本 rank 只需要下一个 rank（ring 的下游）的地址
            my_handle = _C.sym_get_handle(self.self_base)
            handles = [None] * self.world_size
            devices = [None] * self.world_size
            dist.all_gather_object(handles, my_handle, group)
            dist.all_gather_object(devices, self.device, group)

            next_rank = (self.rank + 1) % self.world_size
            _C.enable_peer_access(devices[next_rank])
            self.peer_base = _C.sym_open_handle(handles[next_rank])
            self._opened = True

            # 确保所有 rank 都完成映射后才可能有 kernel 往对面写
            dist.barrier(group)
            torch.cuda.synchronize()

    # ---- 各段地址 -----------------------------------------------------------
    def _recv_ptr(self, base: int, slot: int) -> int:
        return base + slot * self.slot_bytes

    @property
    def self_flag_ptr(self) -> int:
        return self.self_base + self.flag_offset

    @property
    def peer_flag_ptr(self) -> int:
        return self.peer_base + self.flag_offset

    @property
    def self_consume_ptr(self) -> int:
        return self.self_base + self.consume_offset

    @property
    def peer_consume_ptr(self) -> int:
        return self.peer_base + self.consume_offset

    # ---- 主入口 -------------------------------------------------------------
    def __call__(self, a_local: torch.Tensor, b: torch.Tensor,
                 d: Optional[torch.Tensor] = None,
                 compiled_dims: str = 'nk') -> torch.Tensor:
        assert a_local.shape == (self.shape_mp, self.shape_k), \
            f'期望 a_local 形状为 {(self.shape_mp, self.shape_k)}，实际为 {tuple(a_local.shape)}'
        shape_n = b.shape[0]
        shape_m = self.shape_mp * self.world_size
        if d is None:
            d = torch.empty((shape_m, shape_n), device=a_local.device, dtype=torch.bfloat16)

        self.epoch += 1
        slot = self.epoch % self.NUM_SLOTS
        _C.bf16_ring_all_gather_gemm(
            a_local, b, d,
            self._recv_ptr(self.self_base, slot),
            self._recv_ptr(self.peer_base, slot),
            self.self_flag_ptr, self.peer_flag_ptr,
            self.self_consume_ptr, self.peer_consume_ptr,
            self.rank, self.world_size, self.epoch, compiled_dims)
        return d

    def destroy(self) -> None:
        if self._opened:
            _C.sym_close_handle(self.peer_base)
            self._opened = False
        if self.self_base:
            _C.sym_free(self.self_base)
            self.self_base = 0
