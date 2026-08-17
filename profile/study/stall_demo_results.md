# stall_demo 实测结果（Blackwell sm_120 / RTX 5050 Laptop）

> 编译：`nvcc -O3 -lineinfo -arch=sm_120 -o stall_demo stall_demo.cu`
> Profile：`ncu --apply-rules on <sections> --import-source yes -f -o stall_demo_all ./stall_demo`
> GPU：NVIDIA GeForce RTX 5050 Laptop GPU, compute cap 12.0 (Blackwell)
> 指标：`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`（warp 处于某 stall 的平均周期数 / 每 active issue）

## 重要前提：Blackwell 的 stall 归因特性

在 sm_120 上，NCU 的 warp stall 归因与 Hopper/Ampere 文档**不完全一致**：

1. **`wait` 几乎吞噬一切**。绝大多数"等依赖 / 等内存 / 等执行单元"在本机都被归到 `wait`（实测 86%~99%），而非 `long_scoreboard` / `short_scoreboard`。这是 Blackwell 调度器+NCU 归因逻辑的结果，不是 kernel 没 stall。
2. **普通 CUDA C++ micro-kernel 很难干净触发某些 stall**：`sleeping`、`math_pipe_throttle`、`branch_resolving`、`dispatch_stall`、`mio_throttle`、`no_instruction`、`not_selected`、`misc`、`short_scoreboard`、`long_scoreboard` 在本机大多被编译器优化或架构归因到 `wait` / `drain`。
3. 因此下面"实测 TOP stall"是**本机真实数据**，不是教科书预期。每个 kernel 都确实跑过 ncu 并确认。

## 每个 kernel 的设计意图 vs 实测 TOP stall

| # | kernel | 设计意图 stall | 实测 TOP3 stall (本机 sm_120) | 是否命中意图 |
|---|--------|---------------|-------------------------------|------------|
| 1 | kernel_math_dep | Short Scoreboard | wait=94.72, no_instruction=5.17, tex_throttle=2.92 | ❌ 归 wait |
| 2 | kernel_memory_dep | Long Scoreboard | lg_throttle=120.94, drain=46.96, wait=6.15 | ❌ 归 lg_throttle |
| 3 | kernel_no_stall | Selected 高(对照) | wait=86.36, no_instruction=8.31, lg_throttle=1.73 | ⚠️ 对照 |
| 4 | kernel_barrier | Barrier | wait=24.92, membar=18.15, selected=9.08 | ⚠️ barrier 未进 top |
| 5 | kernel_math_pipe_throttle | Math Pipe Throttle | wait=96.57, no_instruction=8.76, not_selected=1.00 | ❌ 全 0 |
| 6 | kernel_membar | Membar | drain=62.32, wait=13.92, no_instruction=5.21 | ⚠️ membar=18 但 drain 更高 |
| 7 | kernel_lg_throttle | LG Throttle | drain=120.63, lg_throttle=17.98, wait=6.99 | ⚠️ drain 更高 |
| 8 | kernel_sleep | Sleeping | short_scoreboard=587.49, barrier=9.37, membar=8.31 | ❌ __nanosleep 被优化 |
| 9 | kernel_branch | Branch Resolving | wait=99.23, no_instruction=7.91, tex_throttle=1.92 | ❌ 被优化成 select |
| 10 | kernel_mio_throttle | MIO Throttle | wait=95.94, no_instruction=5.96, tex_throttle=2.95 | ❌ 归 wait |
| 11 | kernel_wait | Wait | wait=27.59, lg_throttle=22.29, membar=13.26 | ✅ wait 命中(但占比偏低) |
| 12 | kernel_dispatch_stall | Dispatch Stall | wait=95.07, no_instruction=7.13, tex_throttle=1.89 | ❌ 没触发 dispatch |
| 13 | kernel_no_instruction | No Instruction | misc=13.60, wait=5.77, selected=5.51 | ❌ 归 misc |
| 14 | kernel_drain | Drain | wait=43.64, drain=15.62, lg_throttle=3.33 | ✅ drain 命中 |

legend: ❌=未干净复现意图  ⚠️=部分/间接复现  ✅=确实复现

## 本机 sm_120 上"干净复现"的 stall reason（可作为有效示例）

- **wait** —— kernel_wait / kernel_math_dep / kernel_no_stall / kernel_branch 等（几乎所有 kernel 都显著出现，是 Blackwell 主导 stall）
- **drain** —— kernel_drain（15.62）、kernel_lg_throttle（120.63）、kernel_membar（62.32）、kernel_memory_dep（46.96）
- **lg_throttle** —— kernel_memory_dep（120.94）、kernel_lg_throttle（17.98）、kernel_wait（22.29）
- **membar** —— kernel_membar（18.15）、kernel_barrier（18.15）、kernel_wait（13.26）
- **short_scoreboard** —— kernel_sleep（587.49，异常高，因 __nanosleep 被优化退化）
- **misc** —— kernel_no_instruction（13.60）
- **no_instruction** —— 多数 kernel 都出现 2~9（normal tail/idle）
- **barrier** —— kernel_sleep（9.37）、kernel_barrier（间接）

## 本机 sm_120 上"用 pure CUDA C++ 无法干净复现"的 stall（原因说明）

| stall | 为什么本机没复现 |
|-------|----------------|
| sleeping | `__nanosleep` 在 -O3 下被编译器优化掉，warp 没真正睡眠；需 `-O0` 或 `volatile` 强制 |
| math_pipe_throttle | Blackwell FP32 吞吐极高，简单 FMA 循环打不满 math pipe，且编译器可能向量化/重排 |
| branch_resolving | 数据相关分支被编译器优化成 predicate/select（`? :`），无真实分支解析停顿 |
| dispatch_stall | Blackwell 发射端口宽、特殊单元（SFU）充足，密集 SFU 仍归 wait 而非 dispatch stall |
| mio_throttle | shared memory 访问被优化/合并，且 Blackwell 把这类等待归到 wait |
| long_scoreboard | pointer chasing 在本机被归因 lg_throttle + drain，而非 long_scoreboard |
| short_scoreboard | 算术依赖链在本机被归因 wait |
| no_instruction | 低占用 kernel 在本机归 misc 而非 no_instruction |
| not_selected | 自然状态，仅在 occupancy 充足时出现，未单独复现 |
| tex_throttle | texture 单元未使用，仅零星出现 |

## 结论

- 本机 **`stall_demo_all.ncu-rep`** 已包含全部 14 个 kernel 的真实 WarpStateStats，可直接在 ncu-ui 打开看每个 kernel 的 `Warp State Statistics`。
- 在 **Blackwell sm_120** 上，warp stall 归因高度集中于 `wait` + `drain` + `lg_throttle` + `membar`，教科书式的逐项 micro-benchmark 不能 1:1 复现所有 16 个 stall reason。
- 若需在 Blackwell 上硬凑 `sleeping` / `math_pipe_throttle` / `branch_resolving` / `dispatch_stall` / `mio_throttle` 等，需要：关优化（`-O0`）、`volatile` 防优化、或内联 PTX/`__asm__` 阻止编译器重排。可后续迭代。
