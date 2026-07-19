# GEMM TMA 流水线性能调优历程

## 1) 实验范围与目标

- 使用的 Kernel 代码文件：`cutedsl_tma_pipeline_tuning_demo.py`
- 实验目标：实现并验证 **TMA 异步数据搬运 + 软件流水线（software pipeline）+ Tensor Core 矩阵乘加（GEMM）**，然后使用 Nsight Compute 对访存行为做细粒度调优。
- 调优阶段使用的矩阵规模：`M=N=K=4096`

## 2) Profiling（性能分析）环境配置

### NCU 采集命令方式

本次使用 Nsight Compute（简称 NCU）进行性能剖析，重点采集了以下 section（分析模块）：

- `SpeedOfLight`：看各计算/访存单元距离理论峰值的利用率。
- `MemoryWorkloadAnalysis`：分析整体访存工作负载，包括 L1/L2/DRAM 的命中与吞吐。
- `MemoryWorkloadAnalysis_Tables`：以表格形式展示更细的访存拆解数据。
- `LaunchStats`：查看 kernel 发射参数，如寄存器数、block 大小、动态共享内存等。
- `Tile`：查看 TMA / MMA 的 tile 形状匹配情况（CUTEDSL 专用分析项）。

同时，在 Python 代码里使用了 profile range 控制，通过 `cudaProfilerStart/Stop` 来精确圈定需要采集的 kernel 区间，并配合 `ncu --profile-from-start off` 参数，避免把无关阶段的启动开销也采进去。

### 性能报告文件

- Baseline（基准版本）报告：`cutedsl_tma_baseline.ncu-rep`
- Tuned（调优版本）报告：`cutedsl_tma_tuned.ncu-rep`

## 3) 我们观察到了什么

通过 `MemoryWorkloadAnalysis` 以及 kernel 发射参数 / tile 相关明细，我们发现：

- 在这个 GEMM 的访存流量模式下，基准版本表现出了 **低于预期（lower-than-expected）的 L2 行为**。
- 关键证据 —— 基准版本的 L2 命中率（L2 hit rate）只有：**69.11%**。
- 基准版本的 grid 布局为：`(32, 32, 1)`，即没有做任何 CTA swizzle 分组（no CTA swizzle grouping）。

## 4) 我们做了哪些优化

### A. Tile（分块）形状调优

把调优版本的 tile 形状改为：

- 基准版本 tile：`(128, 128, 64)`
- 调优版本 tile：`(128, 64, 64)`

调整理由：在当前矩阵形状下，这样的改动可以改善 N 维度上的访问局部性（access locality），并降低单个 block 上的资源压力。

### B. CTA swizzle 调优

启用 CTA swizzle 分组机制：

- 基准版本 swizzle group：`1`（即不做分组）
- 调优版本 swizzle group：`4`

kernel 的发射映射方式也相应改成 3D 的 swizzle 风格 grid：

- 调优版本的 grid 布局：`(16, 32, 4)`

## 5) 调优结果

### 性能（bench 基准测试）

- 基准版本延迟：约 `5479.63 us`
- 调优版本延迟：约 `5217.42 us`
- 加速比：约 `1.05x`（即大约提升 5%）

### 访存行为（NCU 指标）

- L2 命中率：**69.11% -> 94.71%**
- 每线程寄存器数（Registers/thread）：`178 -> 116`
- 每个 block 的动态共享内存（Dynamic shared memory/block）：`66.56 KB -> 50.18 KB`

结论：通过调优 `Tile + CTA swizzle`，我们改善了缓存局部性和访存效率，并最终转化为可测量的端到端 GEMM 加速效果。

## 6) 最终文件结构

- `profile/gemm/cutedsl_tma_pipeline_tuning_demo.py`
- `profile/gemm/cutedsl_tma_baseline.ncu-rep`
- `profile/gemm/cutedsl_tma_tuned.ncu-rep`
- `profile/gemm/tuning_journey.md`

另外，study 相关的 CUDA demo 也一并移动到了：

- `profile/study/stall_demo.cu`
- `profile/study/roofline_demo.cu`
- `profile/study/stream_demo.cu`
