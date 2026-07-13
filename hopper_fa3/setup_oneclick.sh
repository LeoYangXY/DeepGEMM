#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python3.9
command -v "$PY" >/dev/null 2>&1 || PY=python3
echo "===== [hopper_fa3] 一键环境配置（装完即可 python3.9 run.py 对比 Tri Dao FA3）====="

echo "[1/8] 系统依赖 (gcc-toolset-12, python39-devel)..."
if command -v dnf >/dev/null 2>&1; then
  (sudo dnf install -y gcc-toolset-12-gcc-c++ python39-devel 2>/dev/null || dnf install -y gcc-toolset-12-gcc-c++ python39-devel)
elif command -v yum >/dev/null 2>&1; then
  (sudo yum install -y gcc-toolset-12-gcc-c++ python39-devel 2>/dev/null || yum install -y gcc-toolset-12-gcc-c++ python39-devel)
elif command -v apt-get >/dev/null 2>&1; then sudo apt-get install -y gcc-12 g++-12 python3-dev
fi

echo "[2/8] CUTLASS v3.5.1..."
TP="$PWD/third_party"
mkdir -p "$TP"
if [ ! -d "$TP/cutlass/include/cute" ]; then
  git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "$TP/cutlass"
else echo "    CUTLASS 已存在，跳过"; fi
echo "[3/8] flash-attention v2.7.4 (hopper 头文件)..."
if [ ! -f "$TP/flashattention/hopper/softmax.h" ]; then
  git clone --depth 1 --branch v2.7.4 https://github.com/Dao-AILab/flash-attention.git "$TP/flashattention"
else echo "    flashattention 已存在，跳过"; fi

echo "[4/8] 确保 torch>=2.2（官方 FA3 需要；低于则升级 2.2.2+cu121）..."
TV=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "0")
if "$PY" -c "import sys; v='$TV'.split('+')[0].split('.'); sys.exit(0 if (int(v[0])>=2 and int(v[1])>=2) else 1)" 2>/dev/null; then
  echo "    torch $TV 已满足，跳过"
else
  echo "    升级 torch -> 2.2.2+cu121 ..."
  "$PY" -m pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cu121
  for b in /usr/local/lib/python3.9/site-packages /usr/local/lib64/python3.9/site-packages; do
    [ -d "$b/nvidia" ] && find "$b/nvidia" -maxdepth 2 -type d -name lib
  done > /etc/ld.so.conf.d/nvidia-pip.conf
  ldconfig
  echo "    torch 升级完成，nvidia 运行库已写入 ld.so.conf"
fi
echo "[5/8] 官方 flash-attn (FA3) python 包..."
if "$PY" -c "import flash_attn" 2>/dev/null; then
  echo "    flash-attn 已安装，跳过"
else
  WHEEL="flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
  URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$WHEEL"
  ( cd /tmp && curl -sL -o "$WHEEL" "$URL" && "$PY" -m pip install --no-deps --force-reinstall "$WHEEL" )
  echo "    flash-attn 安装完成"
fi
echo "[6/8] 修补 torch pybind11 cast.h (GCC12 解析 bug)..."
CAST_H=$("$PY" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__),'include','pybind11','cast.h'))")
export CAST_H
if grep -q "using _cast_op_t" "$CAST_H" 2>/dev/null; then
  echo "    已修补，跳过"
elif grep -q "caster.operator typename make_caster" "$CAST_H" 2>/dev/null; then
  cp "$CAST_H" "${CAST_H}.bak.oneclick"
  "$PY" - <<'PY'
import os
p = os.environ["CAST_H"]
s = open(p).read()
old1 = "    return caster.operator typename make_caster<T>::template cast_op_type<T>();"
new1 = "    using _cast_op_t = typename make_caster<T>::template cast_op_type<T>;\n    return caster.operator _cast_op_t();"
old2 = "    return std::move(caster).operator typename make_caster<T>::\n        template cast_op_type<typename std::add_rvalue_reference<T>::type>();"
new2 = "    using _cast_op_t = typename make_caster<T>::template cast_op_type<typename std::add_rvalue_reference<T>::type>;\n    return std::move(caster).operator _cast_op_t();"
assert old1 in s and old2 in s, "pybind11 cast.h 模式不匹配，可能 torch 版本不同"
open(p, "w").write(s.replace(old1, new1).replace(old2, new2))
print("patched")
PY
  echo "    修补完成 (备份: ${CAST_H}.bak.oneclick)"
else
  echo "    [warn] 未找到预期模式，跳过修补"
fi
echo "[7/8] 预编译 CUDA 扩展 (rm -rf build 强制重建)..."
rm -rf "$PWD/build"
echo 'import run; run._get_extension()' > /tmp/_build_ext.py
"$PY" /tmp/_build_ext.py && echo "    扩展编译成功"
rm -f /tmp/_build_ext.py

echo "[8/8] 跑一次对比验证 (默认 causal + 对比 Tri Dao FA3)..."
"$PY" run.py

echo ""
echo "===== 完成 ====="
echo "直接对比 Tri Dao FA3（默认 causal）:  $PY run.py"
echo "non-causal 对比:                     $PY run.py --no-causal"
