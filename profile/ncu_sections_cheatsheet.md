# ncu 常用 Sections 速查表

> 核心心智模型：**基础 section 用来"判断 bound 类型"（结果），Warp 状态 section 用来"定位 stall 原因"（原因）。SOL 告诉你慢到什么程度，stall 告诉你为什么慢、卡在哪行代码。**

## 基础（判断 bound）
| Section | 用来干嘛 |
|---------|----------|
| `SpeedOfLight` | 看 Compute% / Memory% / L1 占用，判断 bound 类型的总入口 |
| `ComputeWorkloadAnalysis` | 看 SM 各 pipe 占用，尤其 Tensor Pipe（WGMMA/MMA）是否打满 |
| `MemoryWorkloadAnalysis`（+ `_Tables` / `_Chart`） | 看 TMA 是否饱和、L2 命中率、未合并/多余 sector |
| `LaunchStats` | 看 grid / block / 寄存器 / shared memory / wave 数，确认启动配置 |
| `Occupancy` | — 看 achieved vs theoretical 占用率，以及被什么资源卡住。占用率 = 同时驻留 SM 的 active warp 数 / SM 最大可容纳 warp 数，受三个资源约束（木桶效应，三者取 min）|

- **Registers / thread**：32 以内满占用；64 掉到 50%；256 只剩 5%
- **Block size**：调度单位，越大能放的 block 越少
- **Shared mem / block**：下降最陡，是占用率头号杀手

> 关键：占用率**不是越高越好**。共享内存加一点占用率就暴跌，所以大 tile kernel 常故意低占用（靠数据复用 + ILP 补偿）；寄存器超 32 后开始掉，飙到 64+ 还可能触发 register spill 更慢。

## Warp 状态（看 stall 原因）
| Section | 用来干嘛 |
|---------|----------|
| `SchedulerStats` | 看有没有 warp 可发（No Eligible %、Issue Slot Utilization） |
| `WarpStateStats` | warp stall 原因分解（Long/Short Scoreboard、Barrier、Wait...） |
| `SourceCounters` | 逐行定位 stall 在哪行（GUI Source 页，Navigate By: Warp Stall Sampling） |

## 典型下钻路径
```
SpeedOfLight（Compute% vs Memory% 定性 bound）
   ↓
ComputeWorkloadAnalysis / MemoryWorkloadAnalysis（确认哪个 pipe / 哪层内存是瓶颈）
   ↓
SchedulerStats（有没有 warp 可发？占用率够不够？）
   ↓
WarpStateStats（停顿在等什么：long scoreboard=等内存 / short scoreboard=等 SMEM / wait=指令依赖 / barrier=同步）
   ↓
SourceCounters（精确到源码/SASS 某一行的具体指令）
```
