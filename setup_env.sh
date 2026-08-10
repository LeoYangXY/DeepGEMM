#!/usr/bin/env bash
# =============================================================================
# DeepGEMM 一键环境配置 + 构建 + 运行脚本（Hopper / SM90，实测于 4x NVIDIA H20）
# =============================================================================
#
# 这个脚本把「装依赖 / 配环境变量 / 编 _C.so / 跑 test」四件事合在一起，
# 换一台类似的机器只要：
#
#     ./setup_env.sh --all          # 从零到能跑（装依赖 + 构建 + 冒烟测试）
#
# 也可以分步：
#     ./setup_env.sh --install      # 只装系统/Python 依赖（需要 root，会用 dnf/yum）
#     ./setup_env.sh --build        # 只构建 C++ 扩展 _C.so
#     ./setup_env.sh --check        # 打印环境自检信息
#     ./setup_env.sh                # 单卡：单个 GEMM 冒烟测试
#     ./setup_env.sh tests/bench_1d1d.py                    # 单卡：FP8 1D1D vs cuBLASLt
#     ./setup_env.sh --dist                                 # 4 卡：ring 通算融合 kernel 测试
#     ./setup_env.sh --dist tests/bench_ag_gemm_baseline.py # 4 卡：all_gather+GEMM baseline
#
# 在交互式 shell 里只想拿环境变量（不跑任何东西）：
#     source setup_env.sh
#
# -----------------------------------------------------------------------------
# 为什么需要这些步骤（踩过的坑）
# -----------------------------------------------------------------------------
#  1. DeepGEMM 的 GEMM kernel 是 JIT 的，要求 nvcc >= 12.3；很多机器上系统 CUDA
#     是 12.1（跟 torch 2.1.2+cu121 配套），所以额外装一份 CUDA 12.4 只给 JIT 用，
#     编 _C.so 仍然用系统 CUDA，避免 cudart 符号版本冲突。
#  2. JIT 出来的代码是 C++20，系统 gcc 8.5 编不过，需要 gcc-toolset-11 当 host 编译器。
#  3. torch < 2.2 的 pybind 没有 c10::ScalarType 的 caster，import _C 会报
#     "arg(): could not convert default argument into a Python object"，
#     仓库里的 csrc/utils/torch_compat.hpp 已经补上了。
#  4. cutlass/cute 头文件必须软链到 deep_gemm/include 下，JIT 编译时才找得到。
# =============================================================================

# --- 允许 `source setup_env.sh`：此时只导出环境变量，不执行任何动作 ------------
__DG_SOURCED=0
(return 0 2>/dev/null) && __DG_SOURCED=1

if [ "${__DG_SOURCED}" = "0" ]; then
    set -euo pipefail
fi

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    DG_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
else
    DG_REPO=$(pwd)
fi

# CUDA 12.4 只用于 JIT；想换版本改这两个变量即可
DG_JIT_CUDA_VER=${DG_JIT_CUDA_VER:-12.4.1}
DG_JIT_CUDA_HOME=${DG_JIT_CUDA_HOME:-/usr/local/cuda-12.4}

# =============================================================================
# 第一部分：环境变量（source 和执行都会走到这里）
# =============================================================================
dg_export_env() {
    # host 编译器：gcc 11。
    # NOTES: gcc-toolset 的 enable 脚本里引用了未定义变量 dynpath64，
    #        在 `set -u` 下会直接报错，所以临时关掉再恢复。
    if [ -f /opt/rh/gcc-toolset-11/enable ]; then
        local __old_opts
        __old_opts=$(set +o)
        set +u
        # shellcheck source=/dev/null
        source /opt/rh/gcc-toolset-11/enable
        eval "${__old_opts}"
    fi

    # 编 _C.so 用系统 CUDA（与 torch 的 cudart 版本一致）
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

    # JIT 用新版 nvcc
    if [ -x "${DG_JIT_CUDA_HOME}/bin/nvcc" ]; then
        export DG_JIT_NVCC_COMPILER=${DG_JIT_CUDA_HOME}/bin/nvcc
    fi
    export DG_JIT_CPP_STANDARD=${DG_JIT_CPP_STANDARD:-20}

    export PYTHONPATH=${DG_REPO}:${PYTHONPATH:-}
}

dg_export_env

if [ "${__DG_SOURCED}" = "1" ]; then
    echo "[deepgemm] 环境变量已导出：CUDA_HOME=${CUDA_HOME}, nvcc(JIT)=${DG_JIT_NVCC_COMPILER:-<未安装>}, gcc=$(gcc -dumpversion 2>/dev/null)"
    return 0
fi

# =============================================================================
# 第二部分：安装依赖
# =============================================================================
dg_log() { echo -e "\033[1;36m[deepgemm]\033[0m $*"; }

dg_install() {
    if [ "$(id -u)" != "0" ]; then
        dg_log "--install 需要 root 权限"; exit 1
    fi

    dg_log "1/6 安装系统包（gcc-toolset-11 / python3.9）"
    dnf install -y -q gcc-toolset-11 gcc-toolset-11-gcc-c++ \
                      python39 python39-devel python39-pip git make >/dev/null || {
        dg_log "dnf 安装失败，请手动确认 gcc-toolset-11 / python39-devel 是否可用"; }
    # 把 python3 指向 3.9（很多镜像默认是 3.6，torch 装不上）
    if command -v alternatives >/dev/null && [ -x /usr/bin/python3.9 ]; then
        alternatives --set python3 /usr/bin/python3.9 2>/dev/null || true
    fi

    dg_log "2/6 安装 CUDA ${DG_JIT_CUDA_VER}（仅 nvcc，供 JIT 使用）-> ${DG_JIT_CUDA_HOME}"
    if [ -x "${DG_JIT_CUDA_HOME}/bin/nvcc" ]; then
        dg_log "    已存在，跳过"
    else
        local url_base="https://developer.download.nvidia.com/compute/cuda/redist"
        local tmp; tmp=$(mktemp -d)
        mkdir -p "${DG_JIT_CUDA_HOME}"
        # JIT 只需要 nvcc 本体 + 头文件 + cudart 头
        for comp in cuda_nvcc cuda_cudart cuda_cccl libcublas; do
            local f="${comp}-linux-x86_64-${DG_JIT_CUDA_VER}-archive.tar.xz"
            dg_log "    下载 ${comp}"
            curl -fsSL "${url_base}/${comp}/linux-x86_64/${f}" -o "${tmp}/${f}"
            tar -xf "${tmp}/${f}" -C "${tmp}"
            cp -rn "${tmp}/${comp}-linux-x86_64-${DG_JIT_CUDA_VER}-archive/"* "${DG_JIT_CUDA_HOME}/" 2>/dev/null || true
        done
        rm -rf "${tmp}"
    fi

    dg_log "3/6 拉取 git submodule（cutlass / fmt）"
    cd "${DG_REPO}" && git submodule update --init --recursive

    dg_log "4/6 安装 Python 依赖"
    # NOTES: wheel >= 0.46 会移除 bdist_wheel 的老接口，torch 1.x/2.1 的 setup 走不通
    python3 -m pip install -q --upgrade pip
    python3 -m pip install -q 'wheel<0.46' setuptools ninja
    python3 -c "import torch" 2>/dev/null || \
        python3 -m pip install -q torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

    dg_log "5/6 软链 cutlass / cute 头文件"
    ln -sfn "${DG_REPO}/third-party/cutlass/include/cutlass" "${DG_REPO}/deep_gemm/include/cutlass"
    ln -sfn "${DG_REPO}/third-party/cutlass/include/cute" "${DG_REPO}/deep_gemm/include/cute"

    dg_log "6/6 依赖安装完成"
}

# =============================================================================
# 第三部分：构建 C++ 扩展
# =============================================================================
dg_so_path() {
    local py_tag
    py_tag=$(python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
    echo "${DG_REPO}/deep_gemm/_C${py_tag}"
}

dg_build() {
    cd "${DG_REPO}"
    ln -sfn "${DG_REPO}/third-party/cutlass/include/cutlass" "${DG_REPO}/deep_gemm/include/cutlass"
    ln -sfn "${DG_REPO}/third-party/cutlass/include/cute" "${DG_REPO}/deep_gemm/include/cute"
    dg_log "构建 _C.so（nvcc=${CUDA_HOME}/bin/nvcc, gcc=$(gcc -dumpversion)）"
    python3 setup.py build
    local so
    so=$(find build -name "_C*.so" -type f | head -n 1)
    [ -n "${so}" ] || { dg_log "构建失败：build 目录下没找到 _C*.so"; exit 1; }
    cp -f "${so}" "${DG_REPO}/deep_gemm/"
    dg_log "构建完成 -> deep_gemm/$(basename "${so}")"
}

# =============================================================================
# 第四部分：环境自检
# =============================================================================
dg_check() {
    dg_log "环境自检"
    echo "  OS          : $(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '\"')"
    echo "  gcc         : $(gcc --version | head -1)"
    echo "  CUDA_HOME   : ${CUDA_HOME}  ($(${CUDA_HOME}/bin/nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1))"
    echo "  nvcc (JIT)  : ${DG_JIT_NVCC_COMPILER:-<未安装>}  ($(${DG_JIT_NVCC_COMPILER:-false} --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1))"
    echo "  python3     : $(python3 -V 2>&1)"
    echo "  torch       : $(python3 -c 'import torch;print(torch.__version__)' 2>&1)"
    echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) x $(nvidia-smi -L 2>/dev/null | wc -l)"
    echo "  _C.so       : $([ -f "$(dg_so_path)" ] && echo OK || echo '缺失，请先 --build')"
    python3 -c "import deep_gemm; print('  deep_gemm   : import OK, num_sms =', deep_gemm.get_num_sms())" 2>&1 | tail -1
}

# =============================================================================
# 第五部分：命令分发
# =============================================================================
DO_INSTALL=0; DO_BUILD=0; DO_CHECK=0; DO_DIST=0; BENCH=""
case "${1:-}" in
    --all)     DO_INSTALL=1; DO_BUILD=1; DO_CHECK=1; shift ;;
    --install) DO_INSTALL=1; shift ;;
    --build)   DO_BUILD=1;   shift ;;
    --check)   DO_CHECK=1;   shift ;;
    --dist)    DO_DIST=1;    shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
esac
BENCH="${1:-}"

[ "${DO_INSTALL}" = "1" ] && { dg_install; dg_export_env; }
[ "${DO_BUILD}"   = "1" ] && dg_build
[ "${DO_CHECK}"   = "1" ] && dg_check

# --install / --build / --check 单独使用时不再跑 test
if [ "${DO_INSTALL}${DO_BUILD}${DO_CHECK}" != "000" ] && [ -z "${BENCH}" ]; then
    exit 0
fi

if [ ! -f "$(dg_so_path)" ]; then
    dg_log "未找到 _C.so，请先执行：./setup_env.sh --build"
    exit 1
fi

cd "${DG_REPO}"
if [ "${DO_DIST}" = "1" ]; then
    BENCH="${BENCH:-tests/test_ring_ag_gemm.py}"
    NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"
    dg_log "torchrun --nproc_per_node=${NPROC} ${BENCH}"
    exec torchrun --nproc_per_node="${NPROC}" "${BENCH}"
else
    BENCH="${BENCH:-tests/smoke_single_gemm.py}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    dg_log "python3 ${BENCH}"
    cd "${DG_REPO}/tests"
    exec python3 "${DG_REPO}/${BENCH}"
fi
