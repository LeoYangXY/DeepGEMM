"""AllGather + GEMM 的 4 卡 baseline。

语义（Megatron 序列并行里的 all-gather + GEMM）：
    每个 rank 持有 A_local [Mp, K]（A 沿 M 切分）与自己的权重 B [N, K]；
    先 all_gather 得到 A_full [M, K]（M = world_size * Mp），
    再做 D = A_full @ B^T，输出 [M, N]。

这里给出三种参考实现：
    1. nccl_ag           : 纯 all_gather（`dist.all_gather_into_tensor`，直接映射到
                           `ncclAllGather`，是 NCCL 里最快的一版 all-gather，
                           相比 `dist.all_gather(list)` 少一次 device-to-device 拷贝）
    2. gemm              : 纯 GEMM（DeepGEMM bf16_gemm_nt / cuBLAS）
    3. baseline_seq      : all_gather 之后再 GEMM（串行，无 overlap）
    4. baseline_chunked  : 把 all-gather 按 rank 拆成 world_size 次 P2P ring 传输，
                           用第二条 stream 做通信、主 stream 分块做 GEMM，
                           是一个"尽力 overlap"的纯 PyTorch 参考实现

用法：
    ./setup_env.sh --dist tests/bench_ag_gemm_baseline.py
"""

import argparse
import os

import torch
import torch.distributed as dist

import deep_gemm


# ----------------------------------------------------------------------------- utils
def bench_cuda(fn, num_warmup: int = 10, num_iters: int = 30) -> float:
    """CUDA event 计时，返回单次耗时（秒）。所有 rank 会先做一次 barrier 对齐。"""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / num_iters / 1e3

    # 取所有 rank 的最大值，避免只看某一张卡
    t_tensor = torch.tensor([t], device='cuda')
    dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX)
    return t_tensor.item()


def gemm_nt(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, backend: str) -> None:
    if backend == 'deepgemm':
        deep_gemm.bf16_gemm_nt(a, b, d)
    else:
        torch.matmul(a, b.t(), out=d)


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='deepgemm', choices=['deepgemm', 'cublas'])
    parser.add_argument('--num-chunks', type=int, default=0,
                        help='baseline_chunked 每个 rank 分片再切几块，0 表示不再细切')
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')

    if rank == 0:
        print(f'world_size = {world_size}, backend = {args.backend}, '
              f'device = {torch.cuda.get_device_name(local_rank)}')
        print(f'{"shape (M,N,K)":>22} | {"nccl_ag":>10} | {"gemm":>10} | '
              f'{"seq(ag+gemm)":>13} | {"chunked":>10} | {"gemm TFLOPS":>11} | {"ag GB/s":>9}')
        print('-' * 106)

    # (M, N, K)：M 是 all-gather 之后的总行数
    shapes = [
        (4096, 4096, 4096),
        (8192, 4096, 4096),
        (8192, 8192, 4096),
        (16384, 4096, 7168),
        (8192, 2048, 7168),
        (4096, 1536, 7168),
    ]

    for (m, n, k) in shapes:
        assert m % world_size == 0
        mp = m // world_size

        a_local = torch.randn((mp, k), device='cuda', dtype=torch.bfloat16) / k ** 0.25
        a_full = torch.empty((m, k), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16) / k ** 0.25
        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

        # ---- 1. 纯 all_gather（NCCL 里最快的一版：all_gather_into_tensor）
        def run_ag():
            dist.all_gather_into_tensor(a_full, a_local)

        t_ag = bench_cuda(run_ag)

        # ---- 2. 纯 GEMM
        run_ag()
        torch.cuda.synchronize()

        def run_gemm():
            gemm_nt(a_full, b, d, args.backend)

        t_gemm = bench_cuda(run_gemm)

        # ---- 3. 串行 baseline
        def run_seq():
            dist.all_gather_into_tensor(a_full, a_local)
            gemm_nt(a_full, b, d, args.backend)

        t_seq = bench_cuda(run_seq)

        # ---- 4. ring P2P + 分块 GEMM 的 overlap 参考实现
        comm_stream = torch.cuda.Stream()
        recv_bufs = [torch.empty((mp, k), device='cuda', dtype=torch.bfloat16)
                     for _ in range(world_size - 1)]
        events = [torch.cuda.Event() for _ in range(world_size)]
        prev_rank = (rank - 1 + world_size) % world_size
        next_rank = (rank + 1) % world_size

        def run_chunked():
            main_stream = torch.cuda.current_stream()
            comm_stream.wait_stream(main_stream)

            # 每个 step 传一个 buffer：ring 方式
            with torch.cuda.stream(comm_stream):
                send_buf = a_local
                for s in range(world_size - 1):
                    recv_buf = recv_bufs[s]
                    # 偶数 rank 先 send 后 recv，奇数 rank 先 recv 后 send，避免死锁
                    if rank % 2 == 0:
                        ops = [dist.P2POp(dist.isend, send_buf, next_rank),
                               dist.P2POp(dist.irecv, recv_buf, prev_rank)]
                    else:
                        ops = [dist.P2POp(dist.irecv, recv_buf, prev_rank),
                               dist.P2POp(dist.isend, send_buf, next_rank)]
                    for req in dist.batch_isend_irecv(ops):
                        req.wait()
                    events[s].record(comm_stream)
                    send_buf = recv_buf

            # step 0 直接算自己的 buffer，与上面的通信并行
            for s in range(world_size):
                chunk_idx = (rank - s + world_size) % world_size
                if s == 0:
                    src = a_local
                else:
                    main_stream.wait_event(events[s - 1])
                    src = recv_bufs[s - 1]
                gemm_nt(src, b, d[chunk_idx * mp:(chunk_idx + 1) * mp], args.backend)

        t_chunked = bench_cuda(run_chunked)

        # ---- 校验 chunked 结果与串行一致
        run_seq()
        ref = d.clone()
        run_chunked()
        torch.cuda.synchronize()
        assert torch.equal(ref, d), 'chunked baseline 结果与串行不一致'

        if rank == 0:
            tflops = 2 * m * n * k / t_gemm / 1e12
            # ring all-gather 每个 rank 实际收发 (R-1)/R * total 字节
            ag_bytes = (world_size - 1) / world_size * m * k * 2
            gbs = ag_bytes / t_ag / 1e9
            print(f'{f"({m},{n},{k})":>22} | {t_ag * 1e6:9.1f}u | {t_gemm * 1e6:9.1f}u | '
                  f'{t_seq * 1e6:12.1f}u | {t_chunked * 1e6:9.1f}u | {tflops:11.1f} | {gbs:9.1f}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
