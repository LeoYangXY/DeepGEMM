#!/usr/bin/env bash
# DeepGEMM 一键运行脚本（H20 / SM90 + torch2.1.2 + nvcc12.4 + gcc9.2.1 环境）
#
# 原理：DeepGEMM 的 GEMM kernel 是 JIT 编译的，第一次运行会在
#   ~/.deep_gemm 下用 nvcc 自动编译出 kernel；C++ 扩展 _C.so 只需预构建一次
#   （已在 /root/DeepGEMM/deep_gemm/_C.cpython-39-x86_64-linux-gnu.so 构建好）。
#   因此「编译」是自动发生的，本脚本只负责把环境对齐后跑 benchmark。
#
# 用法：
#   ./bench.sh                 # 跑默认的 1D1D GEMM vs cuBLASLt 对比
#   ./bench.sh 你的脚本.py      # 跑任意脚本
set -euo pipefail

# ---- 环境对齐（这套组合是本机老工具链编译/运行 DeepGEMM 的必要条件）----
export CUDA_HOME=/usr/local/cuda-12.4
export DG_JIT_NVCC_COMPILER=/usr/local/cuda-12.4/bin/nvcc
export DG_JIT_CPP_STANDARD=17                       # JIT 用 C++17（CuTe 在 C++20+老 host gcc 下会 trait 报错）
export PATH=/tmp/gcc9bin:${PATH}                    # 优先用 gcc 9.2.1 的 host 编译器
export PYTHONPATH=/root/DeepGEMM
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

REPO=/root/DeepGEMM
SO="${REPO}/deep_gemm/_C.cpython-39-x86_64-linux-gnu.so"
BENCH="${1:-${REPO}/tests/bench_1d1d.py}"

if [ ! -f "${SO}" ]; then
    echo "[bench] 未找到预构建的 _C.so，请先构建 C++ 扩展（注意：在本机老工具链下需要本环境专用的构建补丁，"
    echo "        本脚本不含这些补丁。普通情况下 _C.so 应已预构建好，无需重建）。"
    exit 1
fi

cd "${REPO}/tests"
echo "[bench] 运行: ${BENCH}"
exec python3.9 "${BENCH}"
