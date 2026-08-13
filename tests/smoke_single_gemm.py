"""最小 smoke test：验证 DeepGEMM 的单个 GEMM（FP8 与 BF16）在本机能跑通。

用法：
    ./setup_env.sh tests/smoke_single_gemm.py
"""

import subprocess

import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major

from generators import KernelType, MajorTypeAB, QuantConfig, get_ue8m0_usage, generate_normal


def gpu_clocks() -> str:
    """Current SM/mem clocks. WSL2 cannot lock them without Windows admin `nvidia-smi -lgc`."""
    try:
        return subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=clocks.sm,clocks.mem,clocks.max.sm',
             '--format=csv,noheader,nounits'],
            text=True).strip()
    except Exception:
        return 'n/a'


def bench(fn, num_warmup: int = 10, num_iters: int = 50) -> float:
    """返回单次调用的耗时（秒）。"""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters / 1e3


def test_fp8_1d1d(m: int, n: int, k: int) -> None:
    quant_config = QuantConfig()
    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=False)

    # SM120: raw FP8, accumulate=True, FP32 D (TMA reduce-add).
    arch = get_arch_major()
    accumulate = arch == 12
    out_dtype = torch.float if arch == 12 else torch.bfloat16

    a, b, c, d, ref_d = generate_normal(m, n, k,
                                        MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                                        accumulate, out_dtype, kernel_type,
                                        use_ue8m0=use_ue8m0, quant_config=quant_config)

    def run():
        deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=not use_ue8m0,
                                  recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

    if arch == 12:
        # Kernel ignores 1D1D scales. Check against cuBLASLt on the same FP8 tiles.
        c_init = d.clone()
        run()
        d_cublas = c_init.clone()
        deep_gemm.cublaslt_gemm_nt(a[0], b[0], d_cublas, c=d_cublas)
        diff = calc_diff(d, d_cublas)
    else:
        run()
        diff = calc_diff(d, ref_d)

    t = bench(run, num_warmup=20, num_iters=80)
    tflops = 2 * m * n * k / t / 1e12

    d_cublas_t = torch.empty_like(d)
    def run_cublas():
        deep_gemm.cublaslt_gemm_nt(a[0], b[0], d_cublas_t, c=None)
    t_cublas = bench(run_cublas, num_warmup=20, num_iters=80)
    tflops_cublas = 2 * m * n * k / t_cublas / 1e12
    tag = 'FP8 raw' if arch == 12 else 'FP8 1D1D'
    print(f'  [{tag:9}] m={m:6d} n={n:6d} k={k:6d} | diff={diff:.6f} | '
          f'DeepGEMM {t * 1e6:8.1f} us {tflops:7.1f} TFLOPS | '
          f'cuBLASLt {t_cublas * 1e6:8.1f} us {tflops_cublas:7.1f} TFLOPS | '
          f'{t_cublas / t:.2f}x cuBLASLt | clk {gpu_clocks()}')
    assert diff < 1e-3, f'FP8 GEMM 精度不达标: {diff}'


def test_bf16(m: int, n: int, k: int) -> None:
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = a @ b.t()

    def run():
        deep_gemm.bf16_gemm_nt(a, b, d)

    run()
    diff = calc_diff(d, ref_d)
    t = bench(run)
    tflops = 2 * m * n * k / t / 1e12
    print(f'  [BF16     ] m={m:6d} n={n:6d} k={k:6d} | diff={diff:.6f} | '
          f'{t * 1e6:8.1f} us | {tflops:7.1f} TFLOPS')
    assert diff < 1e-3, f'BF16 GEMM 精度不达标: {diff}'


if __name__ == '__main__':
    torch.manual_seed(0)
    print(f'device      : {torch.cuda.get_device_name(0)}')
    print(f'arch major  : {get_arch_major()}')
    print(f'num sms     : {deep_gemm.get_num_sms()}')
    print(f'gpu clocks  : {gpu_clocks()}  (sm, mem, max.sm MHz; lock from Windows admin: nvidia-smi -lgc)')

    print('== 单个 GEMM smoke test ==')
    if get_arch_major() in (9, 10):
        for shape in [(4096, 4096, 4096), (8192, 4096, 7168)]:
            test_bf16(*shape)
    fp8_shapes = [(128, 128, 128), (256, 256, 256), (512, 512, 512),
                  (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    if get_arch_major() != 12:
        fp8_shapes = [(4096, 4096, 4096), (8192, 4096, 7168)]
    for shape in fp8_shapes:
        test_fp8_1d1d(*shape)
    print('全部通过')
