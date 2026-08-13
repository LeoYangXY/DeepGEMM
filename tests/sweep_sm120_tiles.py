"""Sweep SM120 tiles on 4096^3 vs cuBLASLt. Pin with DG_SM120_BLOCK_M/N."""

import os
import subprocess
import sys

import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major

from generators import KernelType, MajorTypeAB, QuantConfig, get_ue8m0_usage, generate_normal
from smoke_single_gemm import bench, gpu_clocks


def one_shape(m, n, k):
    quant_config = QuantConfig()
    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=False)
    a, b, c, d, _ = generate_normal(
        m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        True, torch.float, kernel_type, use_ue8m0=use_ue8m0, quant_config=quant_config)

    def run_dg():
        deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=not use_ue8m0,
                                  recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

    c_init = d.clone()
    run_dg()
    d_cublas = c_init.clone()
    deep_gemm.cublaslt_gemm_nt(a[0], b[0], d_cublas, c=d_cublas)
    diff = calc_diff(d, d_cublas)

    d_cublas_t = torch.empty_like(d)

    def run_cublas():
        deep_gemm.cublaslt_gemm_nt(a[0], b[0], d_cublas_t, c=None)

    for _ in range(15):
        run_dg()
        run_cublas()
    torch.cuda.synchronize()

    t_dg = bench(run_dg, num_warmup=10, num_iters=40)
    clk_dg = gpu_clocks()
    t_cb = bench(run_cublas, num_warmup=10, num_iters=40)
    clk_cb = gpu_clocks()
    flops = 2 * m * n * k
    return diff, flops / t_dg / 1e12, flops / t_cb / 1e12, t_cb / t_dg, clk_dg, clk_cb


def main():
    m = n = k = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    tiles = [
        (128, 128), (128, 112), (128, 96), (128, 80),
        (128, 64), (64, 128), (64, 64), (64, 32),
    ]
    if os.environ.get('DG_SM120_BLOCK_M'):
        tiles = [(int(os.environ['DG_SM120_BLOCK_M']), int(os.environ['DG_SM120_BLOCK_N']))]

    cta = os.environ.get('DG_SM120_CTA_PER_SM', '1')
    print(f'device={torch.cuda.get_device_name(0)} sms={deep_gemm.get_num_sms()} '
          f'arch={get_arch_major()} shape={m} cta/sm={cta} clocks={gpu_clocks()}')
    print(f'{"tile":>10} {"diff":>8} {"DG":>8} {"cuBLAS":>8} {"ratio":>7}  clk_dg / clk_cb')
    for bm, bn in tiles:
        os.environ['DG_SM120_BLOCK_M'] = str(bm)
        os.environ['DG_SM120_BLOCK_N'] = str(bn)
        try:
            diff, tflops_dg, tflops_cb, ratio, clk_dg, clk_cb = one_shape(m, n, k)
        except Exception as e:
            print(f'{bm}x{bn:>3}  FAIL {e}')
            continue
        print(f'{bm}x{bn:<3} {diff:8.6f} {tflops_dg:8.1f} {tflops_cb:8.1f} {ratio:6.2f}x  {clk_dg} / {clk_cb}')


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
