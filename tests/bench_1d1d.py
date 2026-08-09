import numpy as np
import torch

import deep_gemm
from deep_gemm.testing import calc_diff, count_bytes, get_arch_major
from generators import (
    KernelType, MajorTypeAB, QuantConfig, get_ue8m0_usage, generate_normal
)


def time_ms(fn, num_warmup=30, num_iters=200):
    """Robust wall-clock timing of a GPU kernel via CUDA events."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters / 1000.0  # seconds


def main() -> None:
    print(f'arch major = {get_arch_major()}')
    torch.manual_seed(0)

    dtype = torch.float8_e4m3fn
    quant_config = QuantConfig()  # legacy (128, 128, fp32-a, fp32-b)
    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    disable_ue8m0_cast = not use_ue8m0
    recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=False)

    # Forward shapes that stress the 1D1D FP8 GEMM kernel across the m-range
    shapes = [
        (4096, 7168, 4096),     # typical LLM fwd
        (16384, 7168, 4096),    # large m
        (4096, 7168, 14336),    # large k
        (4096, 16384, 4096),    # large n
        (128, 7168, 4096),      # small m (decode-ish)
        (1, 7168, 4096),        # m=1 decode
    ]

    scores = []
    for (m, n, k) in shapes:
        a, b, c, d, ref_d = generate_normal(m, n, k,
                                            MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                                            False, torch.bfloat16, kernel_type,
                                            use_ue8m0=use_ue8m0, quant_config=quant_config)

        # Correctness (the 1D1D kernel applies scale_a*scale_b internally)
        deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast,
                                  recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
        diff = calc_diff(d, ref_d)

        # Benchmark DeepGEMM 1D1D kernel
        t = time_ms(lambda a=a, b=b, d=d, c=c:
                    deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast,
                                              recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b))

        # Benchmark cuBLASLt (raw fp8 tensors, timing only; no per-token scaling)
        cublas_t = time_ms(lambda a=a, b=b, d=d, c=c:
                           deep_gemm.cublaslt_gemm_nt(a[0], b[0], d, c=c))

        tflops = 2 * m * n * k / t / 1e12
        cublas_tflops = 2 * m * n * k / cublas_t / 1e12
        gbps = count_bytes(a, b, d) / 1e9 / t
        print(f'm={m:6} n={n:6} k={k:6} | diff={diff:.5f} | '
              f'DeepGEMM {t*1e6:7.1f}us {tflops:5.0f} TFLOPS {gbps:4.0f} GB/s | '
              f'cuBLASLt {cublas_t*1e6:7.1f}us {cublas_tflops:5.0f} TFLOPS | '
              f'{cublas_t/t:.2f}x cuBLASLt')
        scores.append(cublas_t / t)

    print(f"\nAverage FP8xFP8 1D1D GEMM speedup over cuBLASLt: "
          f"{float(np.prod(scores)) ** (1.0 / len(scores)):.3f}x")


if __name__ == '__main__':
    main()
