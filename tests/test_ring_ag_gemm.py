"""Ring AllGather + GEMM 通算融合 kernel 的正确性 & 性能测试。

用法：
    ./setup_env.sh --dist tests/test_ring_ag_gemm.py

对比对象：
    cublas         : torch.matmul（cuBLAS）做同样的 GEMM，A 已经 all-gather 好
    dg_gemm        : DeepGEMM 的 GEMM，A 已经 all-gather 好
                     cublas / dg_gemm 都是"通信被完全 overlap"时的理论下界
    nccl_ag        : 只做 all_gather（`dist.all_gather_into_tensor`，NCCL 最快的一版）
    seq_cublas     : NCCL all_gather + cuBLAS GEMM（串行，最常见的现网写法）
    seq_dg         : NCCL all_gather + DeepGEMM GEMM（串行）
    fused_ring     : 本文实现的通算融合 kernel（ring 传递 + WGMMA 在同一个 kernel 里）
"""

import argparse
import os

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import calc_diff


def bench_cuda(fn, num_warmup: int = 10, num_iters: int = 30) -> float:
    """返回单次耗时（秒），取所有 rank 的最大值。"""
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

    t_tensor = torch.tensor([t], device='cuda')
    dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX)
    return t_tensor.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=30)
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    torch.manual_seed(1234 + rank)

    if rank == 0:
        print(f'world_size = {world_size}, device = {torch.cuda.get_device_name(local_rank)}, '
              f'num_sms = {deep_gemm.get_num_sms()}')

    shapes = [
        (4096, 4096, 4096),
        (8192, 4096, 4096),
        (8192, 8192, 4096),
        (16384, 4096, 7168),
        (8192, 2048, 7168),
        (4096, 1536, 7168),
        (51200, 1280, 4096),   # 大 M / 窄 N，通信量大、GEMM 也大
    ]

    # ---------------------------------------------------------------- 正确性
    if rank == 0:
        print('\n== 正确性检查 ==')
    for (m, n, k) in shapes:
        mp = m // world_size
        a_local = (torch.randn((mp, k), device='cuda', dtype=torch.bfloat16) / k ** 0.25).contiguous()
        b = (torch.randn((n, k), device='cuda', dtype=torch.bfloat16) / k ** 0.25).contiguous()

        # 参考实现：NCCL all_gather + DeepGEMM GEMM
        a_full = torch.empty((m, k), device='cuda', dtype=torch.bfloat16)
        dist.all_gather_into_tensor(a_full, a_local)
        ref_d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(a_full, b, ref_d)

        ctx = deep_gemm.RingAllGatherGemm(mp, k)
        d = ctx(a_local, b)
        torch.cuda.synchronize()

        # 融合 kernel 里 A 的来源不同（跨卡搬运），但数值路径完全一致，应该逐位相等
        diff = calc_diff(d, ref_d)
        bitwise = torch.equal(d, ref_d)
        ok = bitwise or diff < 1e-6

        # 多次调用，验证 epoch / 双缓冲 / flag 累加逻辑
        for _ in range(5):
            d2 = ctx(a_local, b)
        torch.cuda.synchronize()
        ok = ok and torch.equal(d2, ref_d)

        flag = torch.tensor([1.0 if ok else 0.0], device='cuda')
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if rank == 0:
            print(f'  (M={m:6d}, N={n:5d}, K={k:5d}) | diff={diff:.3e} | bitwise={bitwise} | '
                  f'{"PASS" if flag.item() > 0 else "FAIL"}')
        assert flag.item() > 0, f'shape ({m}, {n}, {k}) 结果不正确'
        ctx.destroy()
        del a_full, ref_d, d, a_local, b
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 性能
    if rank == 0:
        print('\n== 性能（单次耗时，取 4 卡最大值）==')
        print(f'{"shape (M,N,K)":>21} | {"cublas":>9} | {"dg_gemm":>9} | {"nccl_ag":>9} | '
              f'{"seq_cublas":>11} | {"seq_dg":>9} | {"fused_ring":>11} | {"vs seq_cublas":>13} | '
              f'{"overlap":>8} | {"TFLOPS":>7}')
        print('-' * 137)

    for (m, n, k) in shapes:
        mp = m // world_size
        a_local = (torch.randn((mp, k), device='cuda', dtype=torch.bfloat16) / k ** 0.25).contiguous()
        b = (torch.randn((n, k), device='cuda', dtype=torch.bfloat16) / k ** 0.25).contiguous()
        a_full = torch.empty((m, k), device='cuda', dtype=torch.bfloat16)
        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        bt = b.t()   # (K, N)，cuBLAS 的 TN layout，无需额外转置

        ctx = deep_gemm.RingAllGatherGemm(mp, k)

        t_ag = bench_cuda(lambda: dist.all_gather_into_tensor(a_full, a_local), num_iters=args.iters)
        t_cublas = bench_cuda(lambda: torch.matmul(a_full, bt, out=d), num_iters=args.iters)
        t_dg = bench_cuda(lambda: deep_gemm.bf16_gemm_nt(a_full, b, d), num_iters=args.iters)

        def run_seq_cublas():
            dist.all_gather_into_tensor(a_full, a_local)
            torch.matmul(a_full, bt, out=d)

        def run_seq_dg():
            dist.all_gather_into_tensor(a_full, a_local)
            deep_gemm.bf16_gemm_nt(a_full, b, d)

        t_seq_cublas = bench_cuda(run_seq_cublas, num_iters=args.iters)
        t_seq_dg = bench_cuda(run_seq_dg, num_iters=args.iters)
        t_fused = bench_cuda(lambda: ctx(a_local, b, d), num_iters=args.iters)

        if rank == 0:
            tflops = 2 * m * n * k / t_fused / 1e12
            speedup = t_seq_cublas / t_fused
            # overlap 程度：融合 kernel 相对"最快的纯 GEMM"多花的时间占通信时间的比例
            t_gemm_best = min(t_cublas, t_dg)
            overlap = max(0.0, 1.0 - max(t_fused - t_gemm_best, 0.0) / t_ag)
            print(f'{f"({m},{n},{k})":>21} | {t_cublas * 1e6:8.1f}u | {t_dg * 1e6:8.1f}u | '
                  f'{t_ag * 1e6:8.1f}u | {t_seq_cublas * 1e6:10.1f}u | {t_seq_dg * 1e6:8.1f}u | '
                  f'{t_fused * 1e6:10.1f}u | {speedup:12.2f}x | {overlap * 100:7.1f}% | {tflops:7.1f}')

        ctx.destroy()
        del a_full, d, a_local, b, bt
        torch.cuda.empty_cache()

    if rank == 0:
        print('\n全部通过')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
