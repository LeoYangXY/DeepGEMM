# CUDA 微架构微基准手册 (H20 / sm_90)

全部数字均为本仓库代码在真机实测所得，非文档抄录。

## 测试平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA H20 (GH100 裁剪版), sm_90 |
| SM 数 | 78 |
| SM 时钟 | 1980 MHz (满载全程不降频) |
| 显存 | 97 GB HBM3 |
| L2 | 60 MB |
| Shared/SM | 227 KB (opt-in) |
| 互联 | 2 卡 NV18 NVLink |
| 编译 | `nvcc -O3 -arch=sm_90` |

## 时间单位换算

SM 时钟 1980 MHz，所以：

```
1 cycle = 0.505 ns = 5.05e-4 us
1 us    = 1980 cycles  (记成 ~2000 好算)
```

| 事件 | cycles | 时间 |
|---|---|---|
| FFMA 依赖延迟 | 4.45 | 2.2 ns |
| `__syncwarp()` | 1-9 | 0.5-4.5 ns |
| shared 访问 | 29 | 15 ns |
| `__syncthreads()` (256 thr) | 29 | 15 ns |
| DSMEM (邻居 block) | ~190 | 96 ns |
| L2 命中 | 295 | 149 ns |
| HBM | 335 | 169 ns |
| `__threadfence()` | 474 | 239 ns |
| NVLink 远端读 | 1583 | **0.8 us** |
| kernel launch 地板 | ~3560 | **1.8 us** |
| launch + sync | ~13760 | **6.9 us** |

> **1 us ≈ 2000 cycles ≈ 6 次 HBM 往返 ≈ 450 条 FFMA 依赖链。** 单次访存只有 0.17 us，所以性能问题从来不是"某一次访存慢"，而是"几万次访存 × 模式很烂"累积出来的。
>
> 注意该换算只对 **SM 时钟域**成立（`clock64()` 读的就是它）。显存控制器、NVLink、copy engine 各有独立时钟域，那些地方的 GB/s 不要用 SM 时钟反推 cycle。

## 快速索引

| 文件 | 主题 |
|---|---|
| `m_gmem` `m_l2sweep` `m_latency` `m_tlb` | 全局访存 / 缓存层级 / TLB |
| `m_occ` `m_ilp` `m_issue` `m_ops` `m_div` | 发射、占用率、ILP/TLP、指令代价 |
| `m_lds` `m_swizzle` `m_cpasync` `m_pipeline` `m_dsmem` | Shared / cp.async / TMA / 集群 |
| `m_mma` `m_barrier` `m_warpprim` `m_atomic2` | Tensor Core / 同步 / warp 原语 / 原子 |
| `m_launch` `m_grid` `m_stream` `m_p2p` `m_h2d` `m_clock` | 系统级：启动、波次、并发、多卡、时钟 |

---

# A. 全局访存与缓存层级

## A1 `m_gmem` — 访存模式决定一切

| 模式 | 带宽 |
|---|---|
| 连续 scalar | 2977 GB/s |
| stride=32 (每 warp 摸 32 条 line) | 238 GB/s |
| 随机置换 | 163 GB/s |
| 连续 `float4` 向量化 | **3813 GB/s** |

> 同样的数据量，只改访问模式，带宽差 **23 倍**。向量化 `float4` 比 scalar 再快 28%——因为一条 LDG.128 替掉四条 LDG.32，省的是**发射槽**而不是带宽。

## A2 `m_l2sweep` — L2 容量悬崖

固定总访问量，扫 working set：

| WS | GB/s | 所在层级 |
|---|---|---|
| 16 KB – 512 KB | ~7620 | L1/L2 命中 |
| 1 MB – 16 MB | 7112 → 6724 | L2 |
| 32 MB | 6132 | L2 接近满 |
| **64 MB** | **3568** | **掉出 60MB L2 → HBM** |
| 128 MB – 512 MB | 3542 → 3338 | HBM |

> L2 带宽是 HBM 的 **2.0 倍**。悬崖精确出现在 60 MB L2 容量处。这就是为什么 GEMM 要做 L2-aware 的 tile 调度（swizzle block 顺序让同时在跑的 block 复用同一片 L2）。

## A3 `m_latency` — 延迟数字

| 层级 | cycles | ns |
|---|---|---|
| L2 命中 | 295 | 149 |
| HBM | 335 | 169 |
| Shared (本地) | 29 | 15 |
| DSMEM (邻居 block) | 181–198 | 91–100 |
| 远端 GPU (NVLink) | 1583 | 799 |

依赖链上的指令延迟：FFMA 11.7 ns / FADD 12.6 / IMAD 15.1 / DFMA 33.3。

> **HBM 只比 L2 慢 40 cycles**。Hopper 的 L2 优势主要在**带宽**不在延迟。要掩盖 335 cycle，单 warp 的 ILP 根本不够，必须靠 warp 数量。

## A4 `m_tlb` — TLB 几乎摸不到

pointer-chase，stride 从 4 KB 扫到 128 MB：

| stride | cyc/hop |
|---|---|
| 4 KB – 4 MB | 284 |
| 8 MB | 287 |
| 16 – 32 MB | 295 |
| 64 MB | 303 |
| 128 MB | 311 |

> 跨 128 MB 的随机跳转只比密集访问贵 **9.5%**。Hopper 用大页 + 巨大 TLB，CPU 上那种灾难性的 TLB miss 在 GPU 上基本不存在。**别为 TLB 做优化，去优化 coalescing。**

---

# B. 发射、占用率与指令代价

## B1 `m_occ` — 延迟隐藏需要多少 warp

访存受限 kernel，用 shared 用量卡住每 SM 的 block 数：

| warps/SM | GB/s | 占峰值 |
|---|---|---|
| 4 | 405 | 12% |
| 8 | 800 | 23% |
| 16 | 1529 | 44% |
| 24 | 2164 | 63% |
| 32 | 2635 | 77% |
| 48 | 3164 | 92% |
| 64 | 3443 | 100% |

> 带宽和 warp 数**几乎线性**直到 48 warps。要吃满 HBM 至少需要 **48 warps/SM (75% occupancy)**。这条曲线是"提高 occupancy 到底值不值"的直接答案：对访存 kernel，值。

## B2 `m_ilp` — ILP 能替代 occupancy

单 warp 内独立 FFMA 依赖链数量扫描：

| 链数 (ILP) | cyc/FFMA | 单 warp FFMA/cyc |
|---|---|---|
| 1 | 4.45 | 0.22 |
| 2 | 2.57 | 0.39 |
| 4 | **1.32** | 0.76 |
| 8 | 1.56 | 0.64 |
| 16 | 1.28 | 0.78 |

warp 数扫描（每 SM IPC，上限 4 = 4 个 SMSP）：

| 配置 | IPC/SM |
|---|---|
| ILP=1, 1 warp | 0.22 |
| ILP=1, 32 warps | 3.34 |
| **ILP=4, 4 warps** | **3.04** |
| ILP=4, 16 warps | 3.62 |

> **FFMA 依赖延迟 ≈ 4.45 cycles**，正好等于"要 4 条独立链才能填满流水线"。**ILP=4 配 4 个 warp，就能达到 32 个 warp 的 91% 性能**——这就是为什么高性能 GEMM 用大 tile + 大量寄存器累加器（低 occupancy 但高 ILP）反而更快。

## B3 `m_ops` — 指令代价换算表（以 FFMA 为 1.0）

浮点：

| 指令 | cyc/op (单 warp) | IPC/SM | 相对 FFMA |
|---|---|---|---|
| FFMA / FADD / FMUL | 1.2–1.3 | 3.6–4.2 | **1.0** |
| `__frcp_rn` | 76.7 | 0.295 | 12.3x |
| `__fdividef` | 15.3 | 0.488 | 7.4x |
| `/` 精确除法 | 57.8 | 0.233 | **15.6x** |
| `rsqrtf` | 15.1 | 0.494 | 7.4x |
| `sqrtf` | 56.7 | 0.320 | 11.3x |
| `__sinf` (MUFU) | 12.8 | 0.497 | 7.3x |
| `sinf` (精确) | 76.5 | 0.146 | **24.9x** |
| `__expf` | 16.1 | 0.477 | 7.6x |
| `expf` | 18.3 | 0.377 | 9.6x |
| `__logf` | 15.5 | 0.492 | 7.4x |
| `tanhf` | 101.0 | 0.205 | 17.7x |
| `powf` | 96.4 | 0.227 | 16.0x |
| **FP64 FMA** | 64.0 | 0.062 | **58.2x** |
| `half2` HFMA2 | — | 1.84 | 2.0x/指令 (1.0x/FLOP) |

整数：

| 指令 | 相对 FFMA |
|---|---|
| IADD | 0.29x (最便宜) |
| IMAD / IMUL 32b | 1.8x |
| 64-bit ADD | 1.9x |
| 64-bit MUL | 7.7x |
| `funnelshift` | 2.1x |
| `max` | 3.9x |
| `popc`/`clz`/`brev` | 7.3x (走 MUFU) |
| 整数 `%` | 14.3x |
| 整数 `/` | **20.1x** |

> 三条黄金规则：**(1)** 用 `-use_fast_math` 或手写 `__xxxf`，精确 `sinf` 比快速版贵 3.4 倍；**(2)** H20 上 FP64 是 FP32 的 1/58，任何 `double` 都是灾难；**(3)** 整数除法/取模比 FFMA 贵 14–20 倍，换成位运算或预计算倒数。

## B4 `m_div` — 分支代价

| 模式 | 耗时 | 相对 |
|---|---|---|
| 无分支 | 123.9 ms | 1.00x |
| **一致分支**（整 warp 同走向） | 169.4 ms | **1.37x** |
| 发散分支（奇偶线程分叉） | 174.4 ms | 1.41x |

> 反直觉结论：**分支指令本身的开销 (1.37x) 远大于发散的额外开销 (1.03x)**。也就是说，让 warp "不发散"只省了 3%，而"干脆别写分支"能省 27%。优先用 `fmaf`/`min`/`max`/select 消灭分支，而不是费劲去对齐 warp 边界。

## B5 `m_issue` — warp scheduler (SMSP) 争用

同 block 内一个 warp 狂做 ld/st、另一个做计算：

- 两 warp 落在**同一个 SMSP**：计算 warp 损失 **≤ 8.8%**
- 两 warp 落在**不同 SMSP**：损失 **0%**
- 单 warp IPC ≈ 0.5，同 SMSP 双 warp IPC ≈ 1.0（发射槽被填满）

> 每 SM 有 4 个 SMSP，warp i 固定去 `i % 4` 号 SMSP。**warp specialization 时把 producer 和 consumer 的 warp id 错开 4 的余数，就能完全避开发射槽争用。** ncu 的 `not_selected` stall 在同 SMSP 时会非常高，但那是两个 warp 合并统计的结果，会高估实际损失。

---

# C. Shared Memory / cp.async / TMA / 集群

## C1 `m_lds` — Shared 带宽与位宽

| 位宽 | cyc/inst (8 warp) | B/clk/SM |
|---|---|---|
| LDS.32 | 1.00 | 128 |
| LDS.64 | 2.00 | 128 |
| LDS.128 | 4.00 | 128 |
| LDS.128 (32 warp) | 1.00 | **512** |

单 warp 延迟：LDS.32 = 4.16 cyc，LDS.128 = 8.19 cyc。

> **每个 SMSP 的 shared 带宽固定 128 B/clk**，与位宽无关。用 `float4` 不是为了更多带宽，而是**用 1/4 的指令数拿到同样的数据**，把发射槽让给计算。四个 SMSP 齐开时全 SM 可达 512 B/clk。

## C2 `m_lds` bank conflict — 完美线性惩罚

| 冲突度 | cyc/inst | 相对无冲突 |
|---|---|---|
| broadcast (全 warp 同地址) | 1.00 | 1x (免费) |
| 无冲突 | 1.00 | 1x |
| 2-way | 2.00 | 2x |
| 4-way | 4.00 | 4x |
| 8-way | 7.99 | 8x |
| 16-way | 16.00 | 16x |
| 32-way | 32.00 | **32x** |

> 冲突惩罚**严格线性**，没有任何硬件缓解。广播完全免费。32-way 冲突让 shared 慢到比 L2 还差。

## C3 `m_swizzle` — padding vs XOR swizzle

对 `half[64][64]` tile 做列方向读取（MMA 喂数的典型模式）：

| 布局 | cyc/LDS.u16 | shared 用量 |
|---|---|---|
| 裸 `[64][64]` | **32.00** | 8192 B |
| padding `[64][66]` | **1.00** | 8448 B (+3%) |
| XOR swizzle 8×8 | 4.00 | 8192 B (+0%) |

> 裸列读是 32-way 冲突，慢 32 倍。**padding 用 3% 的 shared 换来 32 倍加速，是最划算的优化**。XOR swizzle 在这个 8×8 粒度下只消到 4-way，但它零额外空间，且是 TMA/`ldmatrix` 唯一支持的布局——用 TMA 时没得选，手写 kernel 时优先 padding。

## C4 `m_cpasync` — 三种搬运方式

| 方式 | 带宽 |
|---|---|
| 手写 `ld.global` + `st.shared` | 2348 GB/s |
| `cp.async` 16B | 2371 GB/s |
| **`cp.async.bulk` (TMA)** | **5423 GB/s** |

> `cp.async` 单看带宽和手写几乎一样（它的价值是**释放寄存器和发射槽**，让计算和搬运真正重叠）。而 **TMA 快 2.3 倍**——因为它是单线程发起的整块描述符拷贝，彻底绕开了 per-thread 的地址计算和发射开销。Hopper 上想吃满带宽，TMA 不是可选项。

## C5 `m_pipeline` — cp.async 流水级数

| stages | GB/s |
|---|---|
| 1 (无流水) | 1768 |
| 2 | 2360 |
| 3 | 2742 |
| **4** | **2846** |
| 6 | 2872 (峰值) |
| 8 | 2859 |
| 12 | 2657 (shared 挤占 occupancy) |

朴素同步 global 读作为对照：2574 GB/s。

> **收益在 4 级饱和，6 级见顶，之后因为 shared 用量拖低 occupancy 反而倒退。** 常见的 cutlass 3~4 stage 默认值是有实测依据的。单级 cp.async 比朴素读还慢——没有流水就只有开销。

## C6 `m_dsmem` — Hopper 集群分布式 Shared Memory

`cudaOccupancyMaxPotentialClusterSize` = **8**。

| 访问目标 | cyc/LDS | B/clk/SM | 延迟 |
|---|---|---|---|
| 本地 shared | 1.00 | 127 | **29 cyc** |
| DSMEM (cluster=2) | 7.50 | 17.1 | 181 cyc |
| DSMEM (cluster=4) | 7.50 | 17.1 | 198 cyc |
| DSMEM (cluster=8) | 7.50 | 17.1 | 197 cyc |

> DSMEM 延迟是本地 shared 的 **6.5 倍**，带宽只有 **1/7.4**。但对比 L2 (295 cyc) 它仍快 **1.5 倍**，且**完全不占 L2 带宽**。定位很清楚：不是用来当大 shared 用的，而是替代"经 global 做 block 间交换"。集群规模对性能无影响（2 和 8 一样）。

---

# D. Tensor Core / 同步 / warp 原语 / 原子

## D1 `m_mma` — WMMA 实测吞吐

| 数据类型 | 形状 | 吞吐 |
|---|---|---|
| fp16, f32 累加 | m16n16k16 | 94.7 TFLOPS |
| fp16, f16 累加 | m16n16k16 | 95.0 TFLOPS |
| bf16, f32 累加 | m16n16k16 | 94.6 TFLOPS |
| tf32, f32 累加 | m16n16k8 | 35.6 TFLOPS |
| int8, s32 累加 | m16n16k16 | **140.9 TOPS** |

对照：FP32 CUDA core 满载 = 23.8 TFLOPS。

> **Tensor Core 是 CUDA core 的 4.0 倍**（fp16 vs fp32）。f16 累加相比 f32 累加**没有任何速度优势**（Hopper 上累加器精度免费），所以永远用 f32 累加。tf32 只有 fp16 的 37%，int8 是 fp16 的 1.5 倍。94.7 是 WMMA 同步路径成绩，`wgmma` 异步路径还能更高。

## D2 `m_barrier` — 同步开销（cycles/次）

| 原语 | 32 thr | 256 thr | 1024 thr |
|---|---|---|---|
| `__syncwarp()` | 1 | 2 | 9 |
| `__syncthreads()` | 15 | 29 | **77** |
| `bar.sync` 具名 barrier | 27 | 41 | 89 |
| `__threadfence_block()` | 12 | 19 | 120 |
| `mbarrier` (Hopper) | 71 | 90 | 218 |
| `__threadfence()` | 277 | 287 | **474** |

> 换算：一次 `__syncthreads()` (256 线程) ≈ **24 条 FFMA**；一次 `__threadfence()` ≈ **240 条 FFMA**，比一次 HBM 访问还贵。`__syncwarp()` 基本免费。`mbarrier` 比 `__syncthreads()` 贵 3 倍——它的价值在于**异步**（配合 TMA 做 producer/consumer），单纯当 barrier 用是亏的。

## D3 `m_warpprim` — warp 级原语

| 原语 | cyc/op (单 warp) | IPC/SM | 相对 IADD |
|---|---|---|---|
| IADD (基线) | 0.43 | 20.6 | 1.0x |
| `__shfl_xor_sync` | 6.0 | 1.00 | 20.6x |
| `__shfl_down_sync` | 9.1 | 1.00 | 20.7x |
| `__shfl_sync` (idx) | 10.5 | 0.90 | 22.8x |
| `__ballot_sync` | 8.6 | 0.63 | 32.5x |
| **shared 内存交换** | 8.1 | 0.50 | **41.2x** |
| `__any_sync` | 10.9 | 0.48 | 42.8x |
| `__match_any_sync` | 13.6 | 0.50 | 41.4x |
| `__reduce_add_sync` | 40.4 | 0.50 | 41.2x |

> **shuffle 比走 shared 做同样的 warp 内交换快 2 倍**（IPC 1.00 vs 0.50），且不占 shared、不需要 barrier。`__shfl_xor` 是最快的变体，蝶形 reduce 应该用它。`redux.add` 硬件指令延迟高达 40 cyc，只在需要跨整 warp 归约时才比 5 次 `shfl_xor` 划算。

## D4 `m_atomic2` — 原子操作冲突度曲线

79872 个线程同时打 N 个不同地址：

| 唯一地址数 | global (Gop/s) | shared (Gop/s) | shared/global |
|---|---|---|---|
| 1 (全部同址) | **1.37** | **147.6** | 108x |
| 2 | 2.61 | 282 | 108x |
| 4 | 5.22 | 520 | 100x |
| 8 | 12.7 | 893 | 71x |
| 16 | 13.1 | 1398 | 107x |
| 32 | 12.4 | 1913 | 154x |
| 256 | 26.6 | 1920 | 72x |
| 1024 | 102 | — | — |
| 65536 | **402.6** | — | — |

带返回值 vs 不带返回值：**几乎无差异**（1.37 vs 1.37, 402 vs 363）。

> **全局原子的同址冲突是灾难：1.37 Gop/s，比无冲突慢 294 倍。** 而 shared 原子在同址下仍有 147 Gop/s，比全局原子快 **108 倍**。所以直方图/归约的标准范式就是：**先在 shared 里聚合，每 block 只往 global 打一次**。另外 shared 原子在 32 地址后饱和（正好是 32 个 bank）。

---

# E. 系统级：启动、波次、并发、多卡

## E1 `m_launch` — kernel 启动开销

| 场景 | 耗时 |
|---|---|
| CPU 侧 launch API（异步不同步） | 1.787 us |
| 背靠背连续 launch 摊薄 | 1.790 us |
| **launch + sync 端到端延迟** | **6.949 us** |
| 4 KB kernel 参数 | 2.001 us (+12%) |
| CUDA Graph（100 kernel 打包） | **0.714 us/kernel** |
| 逐个 stream launch | 1.782 us/kernel |

> **launch 开销 1.8 us 是硬地板**——GPU 端执行不到 1.8 us 的 kernel 纯粹在浪费。**CUDA Graph 把它压到 0.714 us，加速 2.50 倍**，对大量小 kernel 的推理场景是必选项。`launch+sync` 要 6.9 us，同步本身就占 5 us，千万别在循环里同步。

## E2 `m_grid` — 波次量化与尾效应

SM=78，每 SM 只放 1 个 block：

| blocks | 波次 | 耗时 | 有效效率 |
|---|---|---|---|
| 78 | 1.00 | 0.034 ms | 100% |
| **79** | 1.01 | **0.065 ms** | **52%** |
| 156 | 2.00 | 0.061 ms | 111% |
| **157** | 2.01 | 0.088 ms | 77% |
| 312 | 4.00 | 0.116 ms | 116% |
| 313 | 4.01 | 0.143 ms | 94% |
| 624 | 8.00 | 0.227 ms | 118% |
| 625 | 8.01 | 0.254 ms | 106% |
| 1248 | 16.00 | 0.449 ms | 120% |

> **多 1 个 block 就多一整波：79 个 block 比 78 个慢 91%，效率腰斩到 52%。** 波次越多尾巴的相对代价越小（8 波时只掉 12%）。两个对策：**(1)** 让 grid 恰好是 SM 数的整数倍；**(2)** 用 persistent kernel（block 数 = SM 数 + grid-stride loop）彻底消除量化。

## E3 `m_stream` — 并发与优先级

优先级范围：lo=0, hi=-5。

| 场景 | 耗时 |
|---|---|
| 1 个 kernel (78 blocks) | 0.698 ms |
| 2 个 kernel 分别在 2 个 stream | 1.367 ms (≈2x) |
| 小 kernel 单独跑 | 137848 cyc |
| 小 kernel 排在满 GPU 大 kernel 之后 | 137874 cyc |

> **多 stream 并发不产生任何加速**（1.367 ≈ 2×0.698）——SM 已被占满时，并发只是把时间片切开。**GPU 没有真正的抢占**：高优先级小 kernel 一旦拿到 SM，执行速度和独占时完全一样（137874 vs 137848 cyc，差 0.02%），但它必须等到大 kernel 释放出 SM。stream 优先级影响的是**排队顺序**，不是运行时抢占。

## E4 `m_p2p` — 双卡 NVLink (NV18)

`cudaDeviceCanAccessPeer` = 1（双向）。

| 消息大小 | `cudaMemcpyPeer` |
|---|---|
| 4 KB | 0.5 GB/s |
| 256 KB | 29.4 GB/s |
| 1 MB | 88.0 GB/s |
| 16 MB | 318 GB/s |
| 256 MB | **391 GB/s** |

kernel 直接访问：

| | 带宽 | 延迟 |
|---|---|---|
| 本地 HBM | 3219 GB/s | 284 cyc / 143 ns |
| **远端 (NVLink)** | **371 GB/s** | **1583 cyc / 799 ns** |

> NVLink 带宽是本地 HBM 的 **1/8.7**，延迟是 **5.6 倍**。小消息（<1 MB）效率极低——4 KB 只有峰值的 0.13%，**必须攒批**。kernel 直读远端显存能拿到 371 GB/s，和 `cudaMemcpyPeer` 峰值 (391) 基本持平，所以在算法里直接用 peer 指针是可行的，不必显式拷贝。

## E5 `m_h2d` — 主机传输

`asyncEngineCount` = 3。

| 大小 | pageable H2D | pinned H2D | pageable D2H | pinned D2H |
|---|---|---|---|---|
| 4 KB | 0.65 | 0.50 | 0.41 | 0.52 |
| 256 KB | 11.5 | 21.0 | 10.6 | 21.0 |
| 2 MB | 20.1 | 47.1 | 19.2 | 44.4 |
| 16 MB | 25.4 | 56.0 | 21.4 | 52.2 |
| 1 GB | 26.1 | **57.6** | 21.6 | **53.7** |

其他：

| 场景 | 带宽 |
|---|---|
| H2D + D2H 同时 | **91.1 GB/s** 聚合 |
| Unified Memory 首次触碰 | 9.72 GB/s |
| UM `cudaMemPrefetchAsync` 之后 | **1430.8 GB/s** |

> **pinned 内存是 pageable 的 2.2 倍**，几乎是免费的加速。有 3 个 copy engine，H2D 和 D2H 能真正并行（91 GB/s > 单向 57.6）。**UM 的 page fault 代价惊人：不 prefetch 只有 9.7 GB/s，prefetch 后 1431 GB/s，差 147 倍**——用 UM 必须配 `cudaMemPrefetchAsync`。

## E6 `m_clock` — 时钟与功耗（25 秒满载 FP32）

| 时刻 | SM 时钟 | 功耗 | 温度 | throttle |
|---|---|---|---|---|
| 空闲 | 345 MHz | 74.8 W | 33 °C | idle |
| 第 1 秒 | 1980 MHz | 145 W | 36 °C | 0x0 |
| 第 5 秒 | 1980 MHz | 204 W | 38 °C | 0x0 |
| 第 25 秒 | **1980 MHz** | 206 W | 40 °C | **0x0** |

FP32 吞吐全程稳定在 **23.8 TFLOPS**（10.72–10.76 ms/iter，抖动 < 0.4%）。

> **H20 在 FP32 满载下完全不降频**：205 W 远低于 400 W TDP，温度只到 40 °C，throttle 位始终为 0。这是 H20 作为裁剪版的特点——算力被砍了，但功耗余量极大。**好消息是：这台机器上的 benchmark 数字高度可复现，不需要担心"前几次迭代偏高"的降频伪影。**（在 H100/A100 上则必须做 warmup 后再测。）

---

# 硬件直觉速查表

按"能省多少"排序的优化优先级：

| # | 直觉 | 实测依据 |
|---|---|---|
| 1 | **访存模式 > 一切**，合并访问 + `float4` | 3813 vs 163 GB/s，**23x** |
| 2 | **shared 原子替代 global 原子** | 147 vs 1.37 Gop/s，**108x** |
| 3 | **列读 shared 必须 padding** | 1.0 vs 32.0 cyc，**32x** |
| 4 | **别用 FP64** | FP64 FMA = 58 条 FFMA |
| 5 | **整数 `/` `%` 换位运算** | 20x / 14x |
| 6 | **grid 对齐 SM 整数倍** | 79 blk 比 78 blk 慢 91% |
| 7 | **TMA 搬数据** | 5423 vs 2348 GB/s，**2.3x** |
| 8 | **Tensor Core 替代 CUDA core** | 94.7 vs 23.8 TFLOPS，**4x** |
| 9 | **精确数学函数换快速版** | `sinf` 24.9x vs `__sinf` 7.3x |
| 10 | **UM 必须 prefetch** | 1431 vs 9.7 GB/s，**147x** |
| 11 | **小 kernel 用 CUDA Graph** | 0.714 vs 1.782 us，**2.5x** |
| 12 | **ILP=4 顶得上 8 倍 occupancy** | ILP4×4warp = ILP1×32warp 的 91% |
| 13 | **shuffle 替代 shared 交换** | IPC 1.00 vs 0.50，**2x** |
| 14 | **pinned 内存** | 57.6 vs 26.1 GB/s，**2.2x** |
| 15 | **消灭分支本身，而非消灭发散** | 分支 1.37x，发散仅额外 1.03x |
| 16 | **访存 kernel 要 48+ warps/SM** | 48 warp 才到 92% 带宽 |
| 17 | **cp.async 流水 4 级足够** | 4 级 2846，12 级反降到 2657 |
| 18 | **warp specialization 错开 SMSP** | 同 SMSP 损失 8.8%，跨 SMSP 为 0 |
| 19 | **NVLink 要攒批** | 4 KB 仅 0.5 GB/s，256 MB 才 391 |
| 20 | **别为 TLB 优化** | 跨 128 MB 只贵 9.5% |

## 关键常数速记（H20）

```
FFMA 依赖延迟   4.45 cyc        shared 延迟      29 cyc
shared 带宽     128 B/clk/SMSP  (512 B/clk/SM)
L2 容量         60 MB           L2 延迟         295 cyc
L2 带宽         ~6.8 TB/s       HBM 延迟        335 cyc
HBM 带宽        ~3.4 TB/s       DSMEM 延迟     ~190 cyc
NVLink 带宽     391 GB/s        NVLink 延迟    1583 cyc
kernel launch   1.8 us          syncthreads     29 cyc
FP32 峰值       23.8 TFLOPS     TC fp16        94.7 TFLOPS
SM 数           78              SMSP/SM           4
```

---

# 复现方法

```bash
nvcc -O3 -arch=sm_90 -o m_xxx m_xxx.cu && ./m_xxx
```

`m_clock` 接受参数：`./m_clock <mode: 0=FP32, 1=mem> <秒数>`。

## 写微基准的四个坑（本仓库全部踩过）

1. **死代码消除**：累加结果不写回 global 就会被整段删掉，表现为耗时 ≈ 0 或数字离谱大。务必写进 `out[]`，必要时加 `asm volatile("":::"memory")`。
2. **ptxas 会把循环不变的 shared load 提出去**：测 shared 必须用 `ld.volatile.shared`，否则 32-way bank conflict 也会显示成 1 cycle。
3. **测吞吐要打断依赖链，测延迟要构造依赖链**：同一段代码 ILP=1 和 ILP=4 差 3.4 倍，搞错了测出来的是另一个量。
4. **窄化截断**：`o[tid] = (int)(x0+x1+x2+x3)` 会让编译器发现 64 位高位无用，把 64 位乘法降级成 32 位（i64mul 从 1.9x 的假象变成真实的 7.7x）。必须按原类型写回。

---

# 附录 A：`__threadfence` 与 CUDA 内存模型

`__threadfence()` 是**内存栅栏**，不是同步 barrier —— 最容易混淆的一点。

| | `__syncthreads()` | `__threadfence()` |
|---|---|---|
| 语义 | **等所有线程到齐**（执行同步） | **不等任何人**，只约束本线程的写**顺序** |
| 作用域 | block 内 | 整个 GPU (device scope) |
| 保证 | 到齐 + 隐含 block 内 fence | 栅栏前的写，对他人一定先于栅栏后的写可见 |

`__threadfence()` 执行完线程立刻继续跑，它只承诺：**别人看到我栅栏后的写时，一定也能看到我栅栏前的写**。

## 三个作用域级别（本仓库 `m_barrier` 实测）

| 级别 | 一致性点 | 64 thr | 1024 thr |
|---|---|---|---|
| `__threadfence_block()` | block 内 / L1 | 12 cyc | 120 cyc |
| `__threadfence()` | 整个 GPU / **L2** | **277 cyc** | **474 cyc** |
| `__threadfence_system()` | host + peer GPU | 更贵（未测） |

**为什么 device scope 贵 23 倍**：L2 是 GPU 上所有 SM 的一致性点。`__threadfence()` 必须把本线程所有 pending 的写从 L1 / 写合并缓冲**刷到 L2 并等确认**；`__threadfence_block()` 只需刷到 L1/shared 层级。

### 典型用法：device-scope 生产者 / 消费者

```cpp
// 生产者线程
data[i] = result;     // 1) 先把数据写好
__threadfence();      // 2) 保证 data 落到 L2 之后，flag 才可能被别人看到
flag = 1;             // 3) 再置位“完成”标志

// 消费者线程
while (flag == 0);    // 自旋等待
// 读到 flag==1 的瞬间，data[i] 一定已经是最终值（fence 保证了写顺序）
```

⚠️ 两个坑：

- **它不保证原子性**：多个线程对同一个地址写，照样互相覆盖，`__threadfence()` 管不了竞争，要用 `atomicAdd` / `atomic_ref` 这类原子操作。
- **它不保证对方“看到”**：fence 只定**顺序**，不负责通知、不负责轮询。别人没看到你的写，纯粹是你没去等 / 没去读，不是 fence 的锅。要做可靠的跨线程信号，得自己配 `flag + 自旋`（如上），或用 `cuda::atomic_ref` 的 acquire/release。

**现代写法**：直接用 `cuda::atomic_ref<T, cuda::thread_scope_device>` 配 `memory_order_release`（生产者）/ `memory_order_acquire`（消费者），把“写顺序”和“读可见”一次性交给原子语义，比手写 `__threadfence() + flag` 安全且不易写错。

### 换算成直觉

一次 `__threadfence()`（device scope，277 cyc）≈ **62 条依赖链 FFMA**（按 4.45 cyc/条）或 277 条独立 FFMA；它比一次 HBM 访问（335 cyc）便宜一点点，但**同量级**——也就是说，一次设备级栅栏的开销已经接近一次远内存访问，能不用就不用，跨 block 通信优先走 shared / cluster DSMEM / `__threadfence_block()`（12~120 cyc）这种便宜得多的层级。

`__threadfence_block()`（12~120 cyc）才是“block 内生产者/消费者”该用的；只有确认真的要和别的 block / 别的 SM 协调全局可见性时，才上 device scope 的 `__threadfence()`。
