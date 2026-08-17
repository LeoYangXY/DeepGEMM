import torch
import time
import os
from torch.utils.cpp_extension import load
from functools import partial


def load_cuda(cuda_src: str, funcs: list[str], extra_cuda_cflags: list[str] = None, extra_include_paths: list[str] = None, extra_ldflags: list[str] = None, verbose: bool = False):
    """JIT compile and load a CUDA extension.
    
    Args:
        cuda_src: path to the .cu file
        funcs: list of function names exported via pybind11
        extra_cuda_cflags: additional nvcc flags
        extra_include_paths: additional include directories (e.g. for cutlass)
        extra_ldflags: additional linker flags (e.g. ["-lcuda"] for driver API)
        verbose: whether to print compilation output
    """
    cuda_cflags = [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]
    if extra_cuda_cflags:
        cuda_cflags.extend(extra_cuda_cflags)

    name = os.path.splitext(os.path.basename(cuda_src))[0]
    build_dir = os.path.join(os.path.dirname(cuda_src), "build")
    os.makedirs(build_dir, exist_ok=True)

    module = load(
        name=name,
        sources=[cuda_src],
        extra_cuda_cflags=cuda_cflags,
        extra_include_paths=extra_include_paths or [],
        extra_ldflags=extra_ldflags or [],
        build_directory=build_dir,
        verbose=verbose,
    )
    return {fn: getattr(module, fn) for fn in funcs}


def check(kernel_fn, ref_fn, *args, atol=1e-3, rtol=1e-3, desc="", **kwargs):
    """Check if kernel_fn output matches ref_fn output within tolerance.
    
    kernel_fn and ref_fn should accept the same args and return a tensor.
    Returns True if check passes.
    """
    out_kernel = kernel_fn(*args, **kwargs)
    out_ref = ref_fn(*args, **kwargs)

    if not isinstance(out_kernel, torch.Tensor):
        out_kernel = torch.tensor(out_kernel)
    if not isinstance(out_ref, torch.Tensor):
        out_ref = torch.tensor(out_ref)

    close = torch.allclose(out_kernel.float().cpu(), out_ref.float().cpu(), atol=atol, rtol=rtol)
    max_diff = (out_kernel.float().cpu() - out_ref.float().cpu()).abs().max().item()
    status = "✅ PASS" if close else "❌ FAIL"
    print(f"  [{desc}] {status} | max_diff={max_diff:.6e}")
    return close


def timed(fn, *args, warmup=10, rep=100, **kwargs):
    """Benchmark a function using CUDA events.
    
    Returns: (result, time_ms)
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(rep):
        result = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / rep
    return result, elapsed_ms


def benchmark_kernels(kernel_dict: dict, ref_fn=None, *args, 
                      warmup=10, rep=100, atol=1e-3, rtol=1e-3, **kwargs):
    """Benchmark multiple kernel variants and optionally compare with a reference.
    
    Args:
        kernel_dict: {"name": callable, ...}
        ref_fn: reference function (e.g. torch ops) to compare correctness & speed
        *args, **kwargs: arguments to pass to each kernel and ref_fn
        
    Returns:
        dict of {"name": time_ms, ...}
    """
    results = {}
    print("=" * 60)

    # Correctness check
    if ref_fn is not None:
        print("Correctness check:")
        for name, fn in kernel_dict.items():
            check(fn, ref_fn, *args, atol=atol, rtol=rtol, desc=name, **kwargs)
        print()

    # Benchmark
    print("Performance benchmark:")
    for name, fn in kernel_dict.items():
        _, ms = timed(fn, *args, warmup=warmup, rep=rep, **kwargs)
        results[name] = ms
        print(f"  [{name}] {ms:.4f} ms")

    if ref_fn is not None:
        _, ms_ref = timed(ref_fn, *args, warmup=warmup, rep=rep, **kwargs)
        results["pytorch_ref"] = ms_ref
        print(f"  [pytorch_ref] {ms_ref:.4f} ms")

    # Summary
    print()
    best_name = min(results, key=results.get)
    print(f"Best: {best_name} ({results[best_name]:.4f} ms)")
    print("=" * 60)
    return results
