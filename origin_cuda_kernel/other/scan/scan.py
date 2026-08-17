#!/usr/bin/env python3
"""Compile and run native CUDA prefix-sum (no PyTorch / CuTe)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "scan")
SRC = os.path.join(HERE, "scan.cu")
NVCC = os.environ.get("NVCC", "nvcc")
ARCH = os.environ.get("ARCH", "sm_120")

cmd = [NVCC, "-O3", "-std=c++17", f"-arch={ARCH}", "-o", BIN, SRC]
print(" ".join(cmd))
subprocess.check_call(cmd)
os.execv(BIN, [BIN] + sys.argv[1:])
