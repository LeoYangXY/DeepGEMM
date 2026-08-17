"""
原生 CUDA HGEMM: TMA + mbarrier + Multi-Stage Pipeline + Warp Specialization
Target: SM120 (Blackwell consumer)

语义: C[M,N] = A[M,K] × B[N,K]^T  (B 转置存储)
"""
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, benchmark_kernels

print("Compiling HGEMM TMA warp-specialized kernel...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "hgemm_tma_warp_spec.cu"),
    funcs=["torch_hgemm_tma_warp_spec"],
    extra_cuda_cflags=["-arch=sm_120"],
    extra_ldflags=["-lcuda"],
)

# M, N, K must be multiples of BM=128, BN=128, BK=16
M, N, K = 4096, 4096, 4096
print(f"\n{'='*70}")
print(f"HGEMM TMA warp-spec fp16 (M={M}, N={N}, K={K})")
print(f"语义: C[M,N] = A[M,K] × B[N,K]^T")
print(f"{'='*70}")

A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(N, K, device="cuda", dtype=torch.float16)  # B 转置存储: (N, K)

def pytorch_matmul(A, B):
    return torch.matmul(A, B.T)

kernels = {
    "hgemm_tma_warp_spec (TMA + mbarrier + WMMA)": lambda A, B: lib["torch_hgemm_tma_warp_spec"](A, B),
}

benchmark_kernels(kernels, pytorch_matmul, A, B, atol=5.0, rtol=0.1)
