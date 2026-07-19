#!/usr/bin/env bash
# ============================================================================
# run_profile.sh  ——  一键为 profile/ 下 3 个 .cu 生成 ncu + nsys 报告
#
# 前置：已用如下命令编译（带 -lineinfo 保留行号；不要 -G，否则优化被关掉、
#       warp stall 特征不真实）
#   nvcc -O3 -lineinfo -arch=sm_120 -o roofline_demo roofline_demo.cu
#   nvcc -O3 -lineinfo -arch=sm_120 -o stall_demo    stall_demo.cu
#   nvcc -O3 -lineinfo -arch=sm_120 -o stream_demo   stream_demo.cu
#
# 重要经验（本机已验证）：
#   1. ncu 不需要 sudo 也能跑（本机允许非 root 收集这些 section）。
#   2. --kernel-name 用的是【正则】不是 shell glob！"kernel_*" 在正则里是
#      "kernel_ 后跟 0 个或多个 _"，匹配不到 kernel_math_dep。要用精确名或
#      "kernel_.*"。最稳是逐个精确 kernel 名 + --launch-count 1。
#   3. --replay-mode kernel 配合 --kernel-name 多匹配会报 "No kernels were
#      profiled"，所以这里统一用默认 replay + 精确 kernel 名 + --launch-count。
#   4. 本机 nsys 版本不支持 "cudamemcpy" 这个 trace 名，用 "cuda" 即可覆盖
#      kernel + memcpy。
#   5. 下面 SECTIONS 已包含全部常用 section（详见同目录 README.md）：
#      - 基础：SourceCounters / SchedulerStats / ComputeWorkloadAnalysis /
#              MemoryWorkloadAnalysis / LaunchStats / SpeedOfLight
#      - warp 状态：WarpStateStats（含 Stall Long/Short Scoreboard 等）
#      - 指令与采样：InstructionStats / PmSampling / PmSampling_WarpStates
#      - CUTEDSL 专用：Tile（TMA/MMA tile 匹配）
#      - Roofline 分层图：Single/Half/Double/Tensor 四种精度
#      - 多卡互联：Nvlink / Nvlink_Tables / Nvlink_Topology（本机单卡也会
#        正常采集，只是数据为空；多卡时有用）
#
# 这些 section 也可一行替代：--set full（采集全部）。但 --set full 很慢且
# 单文件巨大，平时按需列 section 更实用。
#
# 行号对应 & warp stall reasons：
#   - 编译 -lineinfo；ncu 用 --import-source yes 把源码嵌进报告，GUI 直接看行号
。
#   - stall reasons 来自 WarpStateStats（GUI Details → Warp State Statistics）：
#       Stall Long Scoreboard / Short Scoreboard / Drain / Barrier / Wait ...
#   - 逐行 stall 来自 SourceCounters（GUI Source 页按 Warp Stall Sampling 着色）。
#
# 用法：
#   cd profile && ./run_profile.sh
# ============================================================================
set -e
NCU=/usr/local/cuda-13.2/bin/ncu
NSYS=/usr/local/cuda-13.2/bin/nsys
SECTIONS="--section SourceCounters \
--section SchedulerStats \
--section WarpStateStats \
--section ComputeWorkloadAnalysis \
--section MemoryWorkloadAnalysis \
--section MemoryWorkloadAnalysis_Tables \
--section LaunchStats \
--section InstructionStats \
--section PmSampling \
--section PmSampling_WarpStates \
--section Tile \
--section SpeedOfLight \
--section SpeedOfLight_RooflineChart \
--section SpeedOfLight_HierarchicalSingleRooflineChart \
--section SpeedOfLight_HierarchicalHalfRooflineChart \
--section SpeedOfLight_HierarchicalDoubleRooflineChart \
--section SpeedOfLight_HierarchicalTensorRooflineChart \
--section Nvlink \
--section Nvlink_Tables \
--section Nvlink_Topology"
echo ">>> [1/3] ncu: stall_demo (全部 8 个 stall kernel 合并抓 1 份)"
# 注意：本机 ncu 2026.1.0 的 --kernel-name 只接受单一精确全名（kernel_.* / 裸前
# 缀 / 逗号分隔都会报 No kernels were profiled）。因此这里【不指定 --kernel-name】，
# 让 ncu 默认抓取程序里全部 kernel，写进同一个 stall_demo_all.ncu-rep。
# stall_demo 的 main 里 8 个 kernel 各 launch 一次、无 warmup，正好全部采到。
$NCU --apply-rules on $SECTIONS \
     --import-source yes -f -o stall_demo_all ./stall_demo

echo ">>> [2/3] ncu: roofline_demo (3 个 bound kernel 合并抓 1 份, 跳过 3 次 warmup)"
# 合并成单个 roofline_demo_all.ncu-rep，GUI 总览表可直接对比 3 个 kernel。
# 前面 main() 跑了 3 次 warmup，所以用 --launch-skip 3 --launch-count 3 跳过。
$NCU --apply-rules on --launch-skip 3 --launch-count 3 $SECTIONS \
     --import-source yes -f -o roofline_demo_all ./roofline_demo

echo ">>> [3/3] ncu: stream_demo (slow_kernel, 抓第 2 次 launch)"
$NCU --apply-rules on --kernel-name "slow_kernel" --launch-skip 1 --launch-count 1 \
     $SECTIONS --import-source yes -f -o stream_slow_kernel ./stream_demo

echo ">>> [4/4] nsys: stream_demo (timeline + overlap)"
$NSYS profile --trace=cuda,osrt --force-overwrite=true \
     -o stream_demo_nsys ./stream_demo

echo "=== ALL DONE ==="
echo "ncu UI:  ncu-ui roofline_demo_all.ncu-rep        (总览表对比 3 个 bound kernel)"
echo "ncu UI:  ncu-ui stall_demo_all.ncu-rep           (8 个 stall kernel, Details 页看 Warp State Statistics)"
echo "ncu UI:  ncu-ui stream_slow_kernel.ncu-rep       (看 stream 重叠 / 同步点)"
echo "nsys UI: nsys-ui stream_demo_nsys.nsys-rep      (看 H2D/Kernel/D2H 重叠)"
