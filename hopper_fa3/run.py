"""FA3 baseline：用同一份输入，对比你的 CUDA kernel 和官方 FA3 / torch SDPA。

用法:
  python3 run.py                 # 默认 dummy 模式（只跑通流程，不比正确性）
  python3 run.py --real          # 当你在 csrc 里实现了真 attention，开启正确性对比
  OFFICIAL=fa3 python3 run.py    # 强制用官方 FA3 作为对比基准
"""
import argparse
import glob
import importlib
import os
import shutil
import site
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

# WGMMA (.wgmma.mma_async) PTX 只能由 sm_90a 生成；但本机 torch 2.1.2 的
# cpp_extension 不认 "9.0a" 这种带 a 后缀的 arch 列表（会报 Unknown CUDA arch）。
# 因此这里只锁成 9.0（让 torch 生成 sm_90 的代码），真正的 sm_90a 代码生成
# 由下方 extra_cuda_cflags 里的 -gencode=arch=compute_90a,code=sm_90a 负责。
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")


def _ensure_nvidia_libs():
    """pip 版 torch 把 cudnn/cublas 等拆到 nvidia-* 包，其 .so 不在链接器默认
    搜索路径。必须在 python 启动前把它们的 lib 目录加入 LD_LIBRARY_PATH；
    若缺失则 re-exec 自身（新进程启动时链接器才能读到）。

    另外还要把构建 flash-attn 所用的 conda 环境 lib 目录加进来：用 gcc 11.4
    编出的扩展 .so 依赖 GLIBCXX_3.4.29，而本机默认 / IDE 自带的 libstdc++ 较旧，
    不加会导致 import 时 "GLIBCXX_3.4.29 not found"。"""
    dirs = []
    for sp in site.getsitepackages():
        for d in sorted(glob.glob(os.path.join(sp, "nvidia", "*"))):
            lib = os.path.join(d, "lib")
            if os.path.isdir(lib):
                dirs.append(lib)
    # conda env 的 lib（提供 GLIBCXX_3.4.29）
    here = Path(__file__).resolve().parent
    fa = os.path.realpath(os.path.join(here, "third_party", "flashattention"))
    env_lib = os.path.join(os.path.dirname(fa), "env", "lib")
    if os.path.isdir(env_lib):
        dirs.append(env_lib)
    if not dirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    cur_set = set(cur.split(":")) if cur else set()
    if all(d in cur_set for d in dirs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs) + (":" + cur if cur else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_nvidia_libs()


def _pick_cxx():
    """torch 头文件要求 GCC 9+；nvcc 用 -allow-unsupported-compiler 放行更新的 GCC。
    挑一个主版本落在 [9, 13] 区间内的编译器，写进 CXX/CC。-allow-unsupported-compiler
    已经在 extra_cuda_cflags 里，所以 13 也能用。"""
    def ver(cxx):
        try:
            out = subprocess.run([cxx, "-dumpversion"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            parts = [int(x) for x in out.split(".")[:2]]
            return tuple(parts + [0] * (2 - len(parts)))
        except Exception:
            return (0, 0)

    def ok(v):
        # 上限 12：gcc 13 与本项目所用的 CUTLASS 不兼容（signum 重载缺失），
        # 即使有 -allow-unsupported-compiler 也会编译失败。
        return (9, 0) <= v <= (12, 99)

    def set_if_ok(cxx):
        try:
            cxx = shutil.which(cxx) or cxx
        except Exception:
            pass
        if cxx and ok(ver(cxx)):
            os.environ["CXX"] = cxx
            os.environ["CC"] = cxx.replace("g++", "gcc").replace("c++", "gcc")
            return True
        return False

    cur = os.environ.get("CXX", "g++")
    if set_if_ok(cur):
        return
    # 1) gcc-toolset：选最高且主版本 <= 13 的（gcc-toolset-12/13 等）
    for cand in sorted(glob.glob("/opt/rh/gcc-toolset-*/root/usr/bin/c++"), reverse=True):
        if set_if_ok(cand):
            return
    # 2) 常见命名 g++-N
    for name in ("g++-13", "g++-12", "g++-11", "g++-10", "g++-9"):
        if set_if_ok(name):
            return
    # 3) conda / 第三方环境里的交叉 gcc（如构建 flash-attn 所用的 11.4）。
    #    从 third_party/flashattention 软链反推 flash-attn 目录，再到其附近的
    #    env/bin 下找 x86_64-conda-linux-gnu-g++（比硬编码路径更可移植）。
    here = Path(__file__).resolve().parent
    fa = os.path.realpath(os.path.join(here, "third_party", "flashattention"))
    for d in [fa, os.path.dirname(fa),
              os.path.dirname(os.path.dirname(fa)),
              os.path.dirname(os.path.dirname(os.path.dirname(fa)))]:
        cand = os.path.join(d, "env", "bin", "x86_64-conda-linux-gnu-g++")
        if set_if_ok(cand):
            return
    # 4) 系统默认 g++ 恰好落在区间内则直接用
    if ok(ver("g++")):
        return


_pick_cxx()

import torch
from torch.utils.cpp_extension import load

SRC = Path(__file__).resolve().parent / "csrc" / "my_fa3_kernel.cu"
BUILD_DIR = Path(__file__).resolve().parent / "build"
MODULE_NAME = "hopper_fa3_my_kernel"

# CUTLASS (for TMA descriptors, PipelineTmaAsync, GMMA op selectors) and the
# FA3 hopper headers (softmax.h / utils.h) we reuse for online softmax + WGMMA gemm.
_HERE = Path(__file__).resolve().parent
_EXTRA_INCLUDES = [
    str(_HERE / "third_party" / "cutlass" / "include"),
    str(_HERE / "third_party" / "flashattention" / "hopper"),
]

_MY_EXT = None
_MY_BACKEND = "python_sdpa_fallback"
_BUILD_TRIED = False


@lru_cache(maxsize=1)
def _get_extension():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    # 注意：torch 的 cpp_extension 会自动根据 CXX 环境变量给 nvcc 加 -ccbin，
    # 所以这里不需要（也不应）再手动加 -ccbin，否则会出现重复/格式错误的参数。
    return load(
        name=MODULE_NAME,
        sources=[str(SRC)],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math", "-DNDEBUG",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-allow-unsupported-compiler", "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        ] + [f"-I{p}" for p in _EXTRA_INCLUDES],
        extra_cflags=["-O3", "-std=c++17", "-DNDEBUG"] + [f"-I{p}" for p in _EXTRA_INCLUDES],
        build_directory=str(BUILD_DIR),
    )


def my_kernel(q, k, v, softmax_scale, causal, dummy=False):
    """调用你的 CUDA 扩展；正确性跑通前绝不回退 SDPA（会掩盖错误）。编译失败直接抛异常。"""
    global _MY_EXT, _MY_BACKEND, _BUILD_TRIED
    if _MY_EXT is None and not _BUILD_TRIED:
        _BUILD_TRIED = True
        _MY_EXT = _get_extension()
        _MY_BACKEND = "cuda_pybind_extension"
    return _MY_EXT.my_fa3_forward(q, k, v, float(softmax_scale), bool(causal))


def official_impl():
    """优先官方 flash-attn (FA3)；Hopper 上 flash_attn_func 自动 dispatch 到 FA3 内核。
    装不上时回退 torch SDPA。

    注意: flash_attn_func 期望 (B, N, H, D)，而本框架的 q/k/v 是 (B, H, N, D)。
    这里把输入 permute 成 (B, N, H, D) 再调用，输出再 permute 回 (B, H, N, D)
    以便和 my_kernel 的 (B, H, N, D) 输出做正确性对比。
    """
    try:
        from flash_attn import flash_attn_func
        def fn(q, k, v, s, c):
            q_t = q.permute(0, 2, 1, 3).contiguous()
            k_t = k.permute(0, 2, 1, 3).contiguous()
            v_t = v.permute(0, 2, 1, 3).contiguous()
            out = flash_attn_func(q_t, k_t, v_t, dropout_p=0.0, softmax_scale=s, causal=c)
            return out.permute(0, 2, 1, 3).contiguous()
        return "flash_attn_fa3", fn
    except Exception:
        pass
    return "torch_sdpa", lambda q, k, v, s, c: (
        torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=s, is_causal=c)
    )


def _bench(fn, q, k, v, scale, causal, warmup, iters):
    for _ in range(warmup):
        fn(q, k, v, scale, causal)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(q, k, v, scale, causal)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--official", default=os.environ.get("OFFICIAL", "auto"),
                    choices=["auto", "fa3", "sdpa"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--real", action="store_true",
                    help="csrc 里已实现真 attention 时，开启正确性对比")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    q = torch.randn(args.batch, args.heads, args.seqlen, args.head_dim, device="cuda", dtype=dtype).contiguous()
    k = torch.randn_like(q).contiguous()
    v = torch.randn_like(q).contiguous()
    scale = args.head_dim ** -0.5

    pref = "sdpa" if args.official == "sdpa" else "fa3"
    if pref == "sdpa":
        official_name, official_fn = "torch_sdpa", lambda q, k, v, s, c: (
            torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=s, is_causal=c)
        )
    else:
        official_name, official_fn = official_impl()

    out_off = official_fn(q, k, v, scale, args.causal)
    out_mine = my_kernel(q, k, v, scale, args.causal)

    max_abs = (out_mine - out_off).abs().max().item()
    ok = torch.allclose(out_mine, out_off, atol=2e-2, rtol=2e-2)

    flops = 4.0 * args.batch * args.heads * args.seqlen * args.seqlen * args.head_dim
    if args.causal:
        flops *= 0.5
    ms_off = _bench(official_fn, q, k, v, scale, args.causal, args.warmup, args.iters)
    ms_mine = _bench(lambda q, k, v, s, c: my_kernel(q, k, v, s, c), q, k, v, scale, args.causal, args.warmup, args.iters)

    print("=" * 72)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape=(B={args.batch}, H={args.heads}, N={args.seqlen}, D={args.head_dim}) "
          f"dtype={args.dtype} causal={args.causal}")
    print(f"official_backend={official_name}  |  my_kernel_backend={_MY_BACKEND}")
    print("-" * 72)
    if args.real:
        print(f"correctness: {'PASS' if ok else 'FAIL'} | max_abs_diff={max_abs:.6f}")
    else:
        print(f"correctness: SKIP (dummy) | max_abs_diff={max_abs:.6f}  "
              f"(写真 attention 后加 --real 开启对比)")
    print("-" * 72)
    print(f"{official_name:>16} | {ms_off:8.3f} ms | {flops / (ms_off / 1000) / 1e12:8.2f} TFLOPS")
    print(f"{'my_kernel':>16} | {ms_mine:8.3f} ms | {flops / (ms_mine / 1000) / 1e12:8.2f} TFLOPS")
    print("-" * 72)
    print(f"speedup(my_kernel vs {official_name}) = {ms_off / ms_mine:.3f}x")
    print("=" * 72)

    if args.real and not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
