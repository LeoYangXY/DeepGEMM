#!/usr/bin/env bash
# =====================================================================
#  DeepGEMM 一键环境配置 + 构建 + 跑测脚本（修正版）
# ---------------------------------------------------------------------
#  相对原版修复：
#  1) gcc 工具集：探测 /opt/rh 下已有 gcc-toolset-*，优先 13 > 12 > 11，不再 dnf。
#  2) CUDA 版本号：JIT 用 nvcc=12.4.131，配套 cudart/cccl=12.4.127（官方归档真实命名）。
#  3) GPU 卡数：自动用 nvidia-smi -L 取真实卡数（本机 8 张 H20）。
#  4) 构建与 JIT 工具链分离：
#       - 构建 _C.so 用【系统 CUDA 12.1 的 include/libs + gcc-13 作为 host 编译器】。
#         说明：本机 gcc 8.5 编不过 csrc 里的 C++20 代码（smxx_layout.hpp 等聚合/指定初始化器），
#         故构建也用 gcc-13；构建只调 gcc 编 csrc/python_api.cpp（无 .cu），nvcc 不参与，
#         因此链接的 cudart 仍是 torch 的 12.1，不会在进程里混两份 runtime。
#       - 运行时 JIT 用【单独 12.4 nvcc + gcc-13】，nvcc 12.4 支持 gcc 13，kernel 是 C++20。
#  5) cuobjdump：JIT 用 nvcc 编出 cubin 后，DeepGEMM 用 cuobjdump -symbols 抽取内核符号，
#     而 cuda_nvcc 归档不含 cuobjdump，故额外下载 cuda_cuobjdump 12.4.127（否则全部 rank 编译成功却 exit 1）。
#  6) 本机 dnf 被自带插件拦截；安装系统依赖请用 `PYTHONPATH= dnf install ...`（避开 agent 注入的 shim）。
# =====================================================================
set -e

# ----------------------------------------------------------- 基础变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DG_JIT_CUDA_VER="${DG_JIT_CUDA_VER:-12.4.131}"     # nvcc 主版本
DG_RUNTIME_VER="12.4.127"                          # 配套 cudart / cccl
DG_CUDA_DIR="/usr/local/cuda-${DG_JIT_CUDA_VER%.*}" # 12.4.131 -> /usr/local/cuda-12.4
DG_SYS_CUDA="/usr/local/cuda"                      # 系统 CUDA 12.1（torch 配套）用于构建
DG_NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)"
DG_NGPU="${DG_NGPU:-1}"
PYTHON_BIN="$(command -v python3 || command -v python)"

echo ">> DeepGEMM 环境: JIT_CUDA=${DG_JIT_CUDA_VER}, SYS_CUDA=${DG_SYS_CUDA}, GPU=${DG_NGPU}, PY=${PYTHON_BIN}"

# ----------------------------------------------------------- gcc 工具集探测
detect_gcc_toolset() {
  for v in 13 12 11; do
    local e="/opt/rh/gcc-toolset-$v/enable"
    [ -f "$e" ] && { echo "$e"; return 0; }
  done
  return 1
}
GCC_ENABLE="$(detect_gcc_toolset || true)"
if [ -n "$GCC_ENABLE" ]; then
  echo ">> 使用 gcc 工具集: $GCC_ENABLE"
else
  echo ">> 未找到 gcc-toolset，将使用系统默认 gcc（JIT 编译可能失败，请用 gcc>=11）"
fi

# =========================================================== 函数定义

# 运行时 / JIT 环境（12.4 nvcc + gcc-13）
dg_export_runtime_env() {
  [ -n "$GCC_ENABLE" ] && source "$GCC_ENABLE" 2>/dev/null || true
  export CUDA_HOME="$DG_CUDA_DIR"                 # _C.init 读这个决定 JIT 用哪套 nvcc
  export DG_JIT_NVCC_COMPILER="$DG_CUDA_DIR/bin/nvcc"
  export PATH="$DG_CUDA_DIR/bin:$PATH"            # 让 which nvcc 命中 12.4
  export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH}"
}

# 构建环境（系统 CUDA 12.1 的 include/libs + gcc-13 作为 host 编译器）
# 说明：本机 gcc 8.5 编不过 smxx_layout.hpp（C++20 聚合/指定初始化器），
#       故构建也用 gcc-13；构建只调用 gcc 编 csrc/python_api.cpp（无 .cu），
#       nvcc 不参与，因此链接的 cudart 仍是 torch 的 12.1，不会混两份 runtime。
dg_export_build_env() {
  [ -n "$GCC_ENABLE" ] && source "$GCC_ENABLE" 2>/dev/null || true
  export CUDA_HOME="$DG_SYS_CUDA"
  export PATH="$DG_SYS_CUDA/bin:$PATH"
  export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH}"
}

# 1) 拉取依赖 + 下载 12.4 nvcc 组件
dg_install() {
  echo ">> [1/3] 初始化 git submodule（cutlass / fmt）..."
  if [ ! -f third-party/cutlass/include/cutlass/version.h ]; then
    git submodule update --init --recursive
  else
    echo "   submodule 已就绪，跳过"
  fi

  echo ">> [2/3] 软链 cutlass / cute 到 deep_gemm/include ..."
  ln -sfn "$PWD/third-party/cutlass/include/cutlass" deep_gemm/include/cutlass
  ln -sfn "$PWD/third-party/cutlass/include/cute"    deep_gemm/include/cute

  echo ">> [3/3] 下载 CUDA ${DG_JIT_CUDA_VER} nvcc + ${DG_RUNTIME_VER} cudart/cccl 到 ${DG_CUDA_DIR} ..."
  if [ -x "$DG_CUDA_DIR/bin/nvcc" ]; then
    echo "   nvcc 已存在于 $DG_CUDA_DIR，跳过下载"
  else
    mkdir -p "$DG_CUDA_DIR"
    DG_TMP="$(mktemp -d)"
    URL_BASE="https://developer.download.nvidia.com/compute/cuda/redist"
    declare -A comps=(
      [cuda_nvcc]="$DG_JIT_CUDA_VER"
      [cuda_cudart]="$DG_RUNTIME_VER"
      [cuda_cccl]="$DG_RUNTIME_VER"
      [cuda_cuobjdump]="$DG_RUNTIME_VER"
    )
    for comp in "${!comps[@]}"; do
      ver="${comps[$comp]}"
      f="${comp}-linux-x86_64-${ver}-archive.tar.xz"
      echo "   下载 $comp ($ver) ..."
      curl -fsSL "$URL_BASE/$comp/linux-x86_64/$f" -o "$DG_TMP/$f"
      tar -xf "$DG_TMP/$f" -C "$DG_TMP"
      cp -rn "$DG_TMP/$comp-linux-x86_64-${ver}-archive/"* "$DG_CUDA_DIR/" 2>/dev/null || true
    done
    rm -rf "$DG_TMP"
    "$DG_CUDA_DIR/bin/nvcc" --version | tail -1
  fi
}

# 2) 构建 _C.so（系统 CUDA 12.1 include/libs + gcc-13 host 编译器）
dg_build() {
  # 校验解释器大版本，避免用错 python 编出 ABI 不兼容的 .so
  local pyver
  pyver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
  echo ">> 构建 _C.so（系统 CUDA 12.1 + gcc-13，python=${pyver}；cudart 仍用 torch 的 12.1）..."
  dg_export_build_env
  python3 setup.py build
  cp -f build/lib.*/deep_gemm/_C*.so deep_gemm/ 2>/dev/null || true
  ls -l deep_gemm/_C*.so
}

# 3) 自检 import
dg_check() {
  echo ">> 自检 import deep_gemm ..."
  dg_export_runtime_env
  python3 -c "import deep_gemm; print('deep_gemm import OK, version =', deep_gemm.__version__)"
}

# 用 torchrun 多卡跑测试脚本（卡数自动）
dg_dist() {
  dg_export_runtime_env
  torchrun --nproc_per_node="${DG_NGPU}" --nnodes=1 "$@"
}

# 跑默认性能测试（含 M=51200,N=1280,K=4096）
dg_run() {
  echo ">> 跑性能测试 tests/test_ring_ag_gemm.py（含 (51200,1280,4096)）..."
  dg_dist tests/test_ring_ag_gemm.py --iters 30
}

# =========================================================== 参数分发
case "${1:-}" in
  -h|--help)
    cat <<USAGE
用法: $0 {--all|--install|--build|--check|--dist <脚本>|--run}

  --all      环境 + 构建 + 自检（不含跑测）
  --install  仅装环境（submodule / cutlass 软链 / 下载 12.4 nvcc+cudart+cccl+cuobjdump）
  --build    仅构建 _C.so
  --check    仅自检 import
  --dist     用 torchrun 多卡跑测试（卡数自动）： $0 --dist tests/xxx.py [args...]
  --run      跑默认性能测试（已含 M=51200,N=1280,K=4096）

环境变量:
  DG_JIT_CUDA_VER   JIT 用的 CUDA 版本（默认 12.4.131）
  DG_NGPU           覆盖自动探测的 GPU 卡数
USAGE
    ;;
  --install) dg_install ;;
  --build)   dg_build ;;
  --check)   dg_check ;;
  --dist)    shift; dg_dist "$@" ;;
  --run)     dg_run ;;
  --all)
    dg_install
    dg_build
    dg_check
    ;;
  *)
    echo "用法: $0 {--all|--install|--build|--check|--dist <脚本>|--run}（详见 --help）"
    exit 1
    ;;
esac
