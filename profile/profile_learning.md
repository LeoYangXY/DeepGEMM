# GPU Kernel 性能瓶颈分析：工业级实践学习计划

> 目标：系统掌握 GPU kernel 性能分析的完整知识体系，
> 能独立完成从 90% → 100% 峰值性能的最后一程优化。

---


### 1.2 核心指标速查表

| 你想知道 | ncu 指标 | 怎么解读 |
|----------|----------|----------|
| 计算 vs 访存瓶颈 | `sm__throughput.avg.pct` vs `gpu__dram_throughput.avg.pct` | 哪个高 = 哪个是瓶颈；两个都低 = latency bound |
| warp 为什么 stall | `smsp__warps_issue_stalled_*` | `long_scoreboard` = 等 global memory / L2；`wait` = 等 barrier；`mio_throttle` = shared memory bank conflict；`short_scoreboard` = 等 L1/SMEM 结果；`not_selected` = warp ready 但 issue slot 不够 |
| HBM 带宽利用率 | `dram__throughput.avg.pct_of_peak_sustained` | >80% 基本打满 |
| L2 带宽利用率 | `lts__throughput.avg.pct_of_peak_sustained` | 同上 |
| L1 带宽利用率 | `l1tex__throughput.avg.pct_of_peak_sustained` | 同上 |
| occupancy | `sm__warps_active.avg.per_cycle_active` / 理论最大 | 低不一定是问题，但 latency bound 时需要提高 |
| coalescing 效率 | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / 理论最少 sector 数 | 比值越大 = 越多冗余传输 |
| shared memory bank conflict | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | >0 有 bank conflict；数值越大越严重 |
| Tensor Core 利用率 | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained` | 对比理论峰值 |
| 指令 mix 分布 | `sm__inst_executed_pipe_*` | 看 FMA、LSU、Tensor、SFU 各占多少，非计算指令占比越低越好 |
| L1 / L2 cache hit rate | `l1tex__t_sector_hit_rate.pct` / `lts__t_sector_hit_rate.pct` | 命中率越高越好 |
| register 使用量 | `launch__registers_per_thread` | 过高导致 occupancy 下降，过低说明 ILP 不够 |
| register spill | `lmem__throughput` 或 `local memory` usage | 有 local memory 流量 = register spill 了，严重影响性能 |
| achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained` | 实际活跃 warp 比例 |
| eligible warps | `smsp__warps_eligible.avg.per_cycle_active` | 每 cycle 有几个 warp 准备好被调度 |


### 1.4 常见 90% → 100% 的瓶颈及优化方向

| ncu 看到的症状 | 可能原因 | 优化方向 |
|----------------|----------|----------|
| SOL Memory 高，SOL SM 低 | memory-bound | 增大 tile size 提高数据复用；调整数据布局减少传输量 |
| Warp Stall: Long Scoreboard | 等 global memory / L2 回来 | 多级 software pipeline（prefetch 下一轮数据到 SMEM）；cp.async / TMA 异步搬运 |
| Warp Stall: Wait | 等 barrier / mbarrier | 调整流水线深度；减少同步点；用 mbarrier 替代 __syncthreads 做更细粒度同步 |
| Warp Stall: MIO Throttle | shared memory bank conflict | swizzle layout；padding 一列；调整访问 stride |
| Warp Stall: Not Selected | warp ready 但 issue slot 抢不到 | 降低 occupancy 换取更多 register/ILP；或减少非计算指令 |
| Warp Stall: Short Scoreboard | 等 shared memory / L1 / Tensor Core 结果 | 减少 bank conflict；增大向量化宽度；增加 MMA 和 SMEM load 的交错 |
| Warp Stall: Math Pipe Throttle | 某个计算 pipe 饱和 | 换用更快的 pipe（如用 Tensor Core 替代 FMA）；减少冗余计算 |
| L2 hit rate 很低 | tiling 遍历顺序不好 | L2-friendly tile traversal（Swizzle / Hilbert 曲线遍历 / stream-k） |
| Occupancy 低但性能没上去 | register pressure 过高 | 减少 register（`__launch_bounds__`、手动减少局部变量）；或反过来用更大 tile 提高 ILP |
| FMA pipe 利用率 < 期望 | 非计算指令太多 | 减少 address calculation、branch；用向量化 load（LDG.128）；展开循环 |
| Tensor Core 利用率低 | 数据搬运跟不上 | SMEM → register 的搬运需要和 MMA 计算 overlap；用 TMA 异步加载 |
| local memory 有流量 | register spill | 降低 register 用量；调整 `maxrregcount`；简化逻辑减少活跃变量 |
| kernel 之间有 gap（nsys 看到的） | CPU-GPU 同步 / launch overhead | 用 CUDA Graph 打包多次 launch；用 persistent kernel；减少 cudaDeviceSynchronize |
| 末尾 SM 空闲（wave quantization） | block 数不能整除 SM 数 | 调整 grid size 使 block 数是 SM 数的整数倍；或用 stream-k 分解 |

---
### 2.12 SASS 级分析

看sass的指令格式，了解固定延迟，long short scoreboard！！！！

SASS = GPU 的真实机器码。最后 5% 的优化经常需要看 SASS。

#### 2.12.1 `smsp__warp_issue_stalled_*` 指标到底输出什么？

**这是最容易误解的点，先讲清楚：**

- 这些指标**不是**「某个特定 warp 因为 XX 而 stall 了」。它是**整个 SM 上所有 warp 的汇总统计**。
- 它统计的是：**在一个 SM 周期内，某个 warp 因为 XX 原因而无法被发射（issue）** 的 **cycle 数累加**。单位是 **cycle**，不是 warp 个数。
- 所以 `short_scoreboard = 39.2%` 的意思是：在所有 SM 活跃周期内，平均有 39.2% 的「可发射机会」被 short_scoreboard 这个原因堵住了。
- **一个 warp 在一个 cycle 可能同时被多个 stall 原因标记**（ncu 会按优先级归因到一个主要原因）。因此**所有 stall 类百分比加起来可以 >100%**，这是正常的，不是 bug。
- 看这类指标的正确姿势：找 **top-1 stall reason**，而不是看绝对值是否「加起来=100%」。

抓法（Blackwell 上裸 `--metrics` 容易 n/a，用 section 更稳）：

```bash
# 方法 1：Section（本机 sm_120 最稳）
ncu --section WarpStateStats --launch-skip 3 --launch-count 1 ./app

# 方法 2：裸指标（若返回 n/a 就退回方法 1）
ncu --metrics smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained,\
smsp__warp_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained,\
smsp__warp_issue_stalled_mio_throttle.avg.pct_of_peak_sustained,\
smsp__warp_issue_stalled_not_selected.avg.pct_of_peak_sustained \
--launch-skip 3 --launch-count 1 ./app
```

#### 2.12.2 Warp Stall 六大类对照表（本机实测验证）

| 你写的名 | ncu 指标名 | 准确吗 | 代表什么 stall | 本机实测 |
|----------|------------|--------|----------------|----------|
| `short_scoreboard` | `smsp__warp_issue_stalled_short_scoreboard` | ✅ | 等**短固定延迟结果**就绪：SFU(Tensor Core/MUFU) **和数学结果(FMA)**。⚠️ Blackwell 上**纯 FMA 的等待也被归到这一类** | `fma_chain` 实测 short_sb=**39.2%** 而 long_sb=0% |
| `long_scoreboard` | `smsp__warp_issue_stalled_long_scoreboard` | ✅ | 等**全局/L2/HBM 加载**回来，可变高延迟 | memory-bound kernel 里 top-1 |
| `barrier` / `membar` | `..._barrier` / `..._membar` | ✅ | `barrier`=等 `__syncthreads`(执行屏障)；`membar`=等显式内存屏障(`__threadfence`/`MEMBAR.CTA`) | 4 个 kernel 里都是 **0%**（没用同步） |
| `mio_throttle` | `smsp__warp_issue_stalled_mio_throttle` | ✅ | Shared memory bank conflict（以及 SMEM 其他 MIO 端口争用） | 有 SMEM 冲突的 kernel 里出现 |
| `not_selected` | `smsp__warp_issue_stalled_not_selected` | ✅ | warp 已就绪(eligible)但 issue slot 抢不到（warp 太多/指令太密） | `mixed_chain` 里高达 **53%** |
| `wait` | `smsp__warp_issue_stalled_wait` | ✅ | 等 barrier / mbarrier 完成、pipeline 同步（TMA 常见） | TMA pipeline kernel 里出现 |

> 经验法则（直接对应 1.2 速查表）：`long_scoreboard`=等 global memory/L2；`wait`=等 barrier；`mio_throttle`=shared memory bank conflict；`short_scoreboard`=等 L1/SMEM/TC 结果；`not_selected`=warp ready 但 issue slot 不够；`barrier`/`membar`=同步屏障。

#### 2.12.3 还有哪些 ncu 真实存在但常被忽略的 stall 类

上面 6 类是日常 top-1 常客，但 `smsp__warp_issue_stalled_*` 家族里还有这些，看 ncu 表格时会遇到：

- **`tex_throttle`** — 纹理单元吞吐受限（用 surface/texture 时才出现）
- **`lg_throttle`** — 逻辑/特殊指令（`__popc`、`__brev` 等）的 issue 节流
- **`no_instruction`** — 该 SM 上没有可发的合法指令（kernel 边界、warp 已退出）
- **`sleeping`** — warp 因 `nanosleep` / `vote` 等被挂起
- **`dispatch_stall` / `imc_miss` / `tm_miss`** — 指令缓存/SM 管理类，一般不用管

一句话总结（补进分析流程）：Blackwell/sm_120 的 warp stall 类 = **long_scoreboard**(访存) + **short_scoreboard**(SFU/数学结果) + **barrier/membar**(执行/内存屏障) + **wait**(pipeline 同步) + **mio_throttle**(SMEM bank conflict) + **not_selected**(issue slot 争用)，外加 `tex_throttle`/`lg_throttle`/`no_instruction`/`sleeping` 等次要类。日常分析 top-1 几乎都在前 6 类里。

**关键 SASS 概念：**

```bash
# 查看 kernel 的 SASS 汇编
cuobjdump -sass ./your_app | less

# 配合 ncu Source 页面看 SASS ↔ CUDA C++ 的对应
# ncu GUI: Source → 选 "Correlate with Source"
```

**关键 SASS 概念：**
- **Scoreboard / DEPBAR**：硬件依赖追踪。`DEPBAR` 指令等待之前的 load 完成
- **Predication (`@P0`, `@!P0`)**：条件执行，替代分支的方式
- **Stall counts**：每条 SASS 指令后的 stall cycle 数字（如 `/* 0x000fc800 */`），表示 issue 前需等待的 cycle
- **Register bank conflict**：SASS 中如果连续指令的 source operand 在同一 register bank，可能造成 1-cycle penalty
- **Dual issue**：某些架构支持同 cycle 发射两条指令（如 FMA + 非 FMA），看 SASS 是否做了

---

## 三、工具使用手册

### 3.1 Nsight Compute (ncu) — 微观分析

```bash
# 基础 profile
ncu --set full -o profile ./your_app

# 只 profile 特定 kernel
ncu --set full --kernel-name "your_kernel_name" -o profile ./your_app

# 只 profile 第 N 次 launch（避免 warmup 干扰）
ncu --set full --launch-skip 2 --launch-count 1 -o profile ./your_app

# 对比优化前后
ncu --set full -o before ./your_app_v1
ncu --set full -o after  ./your_app_v2
# GUI 中 File → Open → 选两个文件 Baseline Compare

# 命令行查看 roofline
ncu --section SpeedOfLight_RooflineChart ./your_app

# 导出指标为 CSV（自动化分析）
ncu --csv --metrics sm__throughput.avg.pct_of_peak_sustained,dram__throughput.avg.pct_of_peak_sustained ./your_app

# 查看 SMEM/register 使用量
ncu --metrics launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic ./your_app
```

**核心 Section 阅读顺序：**
1. **Speed of Light** → 大方向：compute vs memory bound
2. **Roofline Chart** → 可视化看 kernel 在 roofline 的位置
3. **Memory Workload Analysis** → 各级内存 throughput 和 hit rate
4. **Compute Workload Analysis** → 各 pipe 利用率
5. **Warp State Statistics** → stall reason 分布（最重要！）
6. **Occupancy** → 资源限制因素（register? SMEM? block size?）
7. **Scheduler Statistics** → eligible warps per cycle
8. **Source Counters** → 定位到具体代码行的热点

### 3.2 Nsight Systems (nsys) — 宏观分析

```bash
# 全局 timeline 分析
nsys profile --stats=true -o timeline ./your_app

# 含内存分配追踪
nsys profile --cuda-memory-usage=true -o timeline ./your_app

# 只看 CUDA 活动
nsys profile --trace=cuda,nvtx -o timeline ./your_app
```

nsys 回答的问题：
- kernel 之间是否有 idle gap（CPU 端瓶颈）
- 多个 stream 是否真的在并行执行
- memcpy 和 kernel 是否 overlap
- CUDA Graph 是否正确 replay

### 3.3 cuobjdump + SASS

```bash
# 查看 SASS
cuobjdump -sass ./your_app | less

# 查看 PTX
cuobjdump -ptx ./your_app | less

# 查看资源使用
cuobjdump -res-usage ./your_app
```

### 3.4 NCU A/B Testing 方法论

```bash
# 1. Profile baseline
ncu --set full -o baseline ./app_v1

# 2. 做一个改动（只改一个变量！）

# 3. Profile 优化版
ncu --set full -o optimized ./app_v2

# 4. 在 GUI 中 Baseline Compare，逐指标对比

# 5. 确认：
#    - 目标指标是否改善了？（如 bank conflict 减少）
#    - 有没有意外的 regression？（如 occupancy 降低）
#    - 整体 kernel 时间是否减少？
```

---

## 四、学习计划（实战驱动版）

> **核心理念**：Profile 是实践学科，不是阅读学科。
> 以 **FlashAttention 3 的完整分析** 为主线项目，热身阶段快速建立工具直觉，后续拓展覆盖更多 kernel 类型。
> 理论知识直接问 AI 讲解，只有需要反复查阅的参考资料才列出链接。

---

### Phase 1: 热身 — 建立 ncu 直觉 + 优化闭环体验

**目标**：用你已有的 kernel 快速熟悉 ncu 工作流，练一次完整的"profile → 改 → 验证"闭环。

**预计时间**：3-5 天

#### 1.1 三类 Kernel 对比（1 天）

用你已有的代码，不需要额外写：

| Kernel | 用什么 | 预期瓶颈 |
|--------|--------|----------|
| Memory-bound | `vectorAdd` 或 `elementwise` | SOL Memory 高，SOL SM 低 |
| Compute-bound | 你的 `hgemm_wmma.cu` | SOL SM 高，Tensor Core 利用率高 |
| Latency-bound | 写一个简单的 `gather`（random index 访问） | 两个 SOL 都低，stall: long_scoreboard |

每个 kernel 做：
```bash
ncu --set full --kernel-name "kernel_name" --launch-skip 1 --launch-count 1 -o xxx ./app
```

**重点练习**：
- 看 Speed of Light 面板，3 秒判断瓶颈类型
- 看 Warp State Statistics，找 top-1 stall reason
- 手算理论上限：`bytes / HBM_BW` vs `FLOPs / peak_FLOPS`，取 max = 理论时间

#### 1.2 在你的 HGEMM 上做一次优化闭环（2-4 天）

你已经有 `hgemm_wmma.cu`，不需要从 naive 重写。目标是**练一次完整闭环**：

- [ ] **Profile baseline**：跑 ncu，记录 kernel time、SOL%、top stall reason
- [ ] **找到 top-1 瓶颈**：是 bank conflict？是 pipeline 不够深？是 occupancy 太低？
- [ ] **做一个针对性改动**（只改一个东西！）：
  - 如果 bank conflict 高 → 加 padding 或 swizzle
  - 如果 long_scoreboard 高 → 尝试 double buffering
  - 如果 occupancy 低 → 调整 `__launch_bounds__` 或减少 register
- [ ] **Profile 优化版**：ncu A/B compare，确认目标指标改善了
- [ ] **记录**：改了什么、哪个指标变了、kernel time 变化多少

#### 需要掌握的理论（问 AI 即可，不需要读论文）

- Roofline 模型：`AI = FLOPs / Bytes`，拐点 = peak_FLOPS / peak_BW
- Hierarchical Roofline：L1/L2/HBM 每层都有自己的 ceiling
- 你的 GPU 的关键数字（查 spec sheet 或跑 `deviceQuery`）

#### 过关标准

✅ 能看 ncu 的 SOL 面板，3 秒内判断瓶颈类型
✅ 能手算任意 kernel 的理论时间
✅ 完成了一次完整的 profile → 优化 → 验证闭环

---

### Phase 2: 主线项目 — FlashAttention 3 深度 Profile 实战

**目标**：对 FA3 做工业级的完整性能分析，理解每个设计决策背后的性能原因。这是面试时能讲的核心"故事"。

**预计时间**：3-4 周

#### 为什么选 FA3 作为唯一主线

1. 它是 Hopper 上最有代表性的 kernel——TMA、WGMMA、software pipeline、swizzle、online softmax 全用上了
2. 一个项目覆盖所有进阶技术，不需要分散精力去做多个小项目
3. 面试时说"我深度 profile 过 FA3 并理解了它的每个设计决策"是极强的信号
4. 它的优化已经接近极限，分析它能学到"最后 10%"在哪里

#### Step 1: 环境搭建 + Baseline 数据（2-3 天）

- [ ] **Clone 并编译 flash-attn**
  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  # 按 README 安装，确保 Hopper GPU 可用
  pip install -e .
  ```

- [ ] **跑 benchmark，建立性能基线**
  ```python
  # 测试不同配置下的 TFLOPS
  # batch=4, heads=32, head_dim=128, dtype=fp16
  # seq_len = [512, 1024, 2048, 4096, 8192, 16384]
  ```
  记录每个配置的：
  - 实测 TFLOPS
  - 理论峰值 TFLOPS（H100: 989 TFLOPS for FP16 Tensor Core）
  - efficiency% = 实测 / 理论

- [ ] **理解 FA3 的 FLOP 计算**（问 AI 推导）
  ```
  Forward:
    Q×K^T: 2 * batch * heads * seq_len * seq_len * head_dim  (GEMM)
    P×V:   2 * batch * heads * seq_len * seq_len * head_dim  (GEMM)
    总 FLOP ≈ 4 * B * H * N^2 * d  (忽略 softmax 的 elementwise 开销)

  Arithmetic Intensity:
    FLOP = 4*B*H*N^2*d
    Bytes = 2 * B*H*N*d * 3 (Q,K,V 读) + 2 * B*H*N*d (O 写)  [FP16 = 2 bytes]
    AI = 4*N^2*d / (8*N*d) = N/2
    → seq_len 越长，AI 越高，越 compute-bound
  ```

- [ ] **画出 FA3 在不同 seq_len 下的 roofline 位置**
  - seq_len=512: AI=256, 可能还在 memory-bound 区域
  - seq_len=4096: AI=2048, 深入 compute-bound 区域
  - 这解释了为什么 FA3 在长序列时效率更高

#### Step 2: ncu Profile Forward Kernel（3-5 天）

- [ ] **Profile 命令**
  ```bash
  # FA3 的 forward kernel 名字通常包含 "flash_fwd" 或 "compute_attn"
  # 先用 nsys 找到 kernel 名字：
  nsys profile --stats=true python your_benchmark.py

  # 然后用 ncu profile 具体 kernel：
  ncu --set full \
      --kernel-name "regex:flash_fwd.*" \
      --launch-skip 2 --launch-count 1 \
      -o fa3_fwd_seqlen4096 \
      python your_benchmark.py --seq-len 4096
  ```

- [ ] **分析 SOL 面板**
  - 预期（seq_len=4096, head_dim=128）：
    - SM SOL: 60-80%（compute-bound 方向）
    - Memory SOL: 30-50%
    - Tensor Core 利用率: 50-70%（不会到 90%+，因为有 softmax 等非 MMA 操作）

- [ ] **分析 Warp Stall 分布**
  - 预期 top stall reasons：
    - `wait` — 等 mbarrier（pipeline 同步）
    - `long_scoreboard` — 等 TMA 搬运完成
    - `math_pipe_throttle` — Tensor Core 饱和（好事！）
    - `short_scoreboard` — 等 SMEM load 到 register
  - **关键洞察**：如果 `wait` 占比很高，说明 pipeline 深度不够或 TMA 搬运太慢

- [ ] **分析 Memory 层级**
  - HBM throughput: 接近峰值说明搬运效率高
  - L2 hit rate: FA3 的 K/V 可能有 L2 复用（取决于 tile 遍历顺序）
  - SMEM throughput: 看 bank conflict 数量
  - **关键指标**：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`

- [ ] **对比不同 seq_len 的 profile 数据**
  ```
  seq_len=512  → 预期更 memory-bound（AI 低）
  seq_len=4096 → 预期更 compute-bound（AI 高）
  ```
  做一个表格：

  | seq_len | kernel time | SM SOL% | Mem SOL% | TC util% | top stall |
  |---------|-------------|---------|----------|----------|-----------|
  | 512     | ?           | ?       | ?        | ?        | ?         |
  | 1024    | ?           | ?       | ?        | ?        | ?         |
  | 4096    | ?           | ?       | ?        | ?        | ?         |
  | 8192    | ?           | ?       | ?        | ?        | ?         |

#### Step 3: 理解 FA3 源码架构（1 周）

这一步是**边读源码边问 AI**，目标是理解每个设计决策的性能原因。

- [ ] **定位核心源码文件**
  ```
  flash-attention/
  ├── hopper/                          # Hopper 专用实现
  │   ├── flash_fwd_kernel.h           # Forward kernel 主体
  │   ├── flash_bwd_kernel.h           # Backward kernel 主体
  │   ├── mainloop_fwd_sm90_tma_gmma_*.hpp  # Pipeline mainloop
  │   ├── tile_scheduler.h             # Tile 遍历策略
  │   └── softmax.h                    # Online softmax 实现
  └── cute/                            # CuTe layout 定义
  ```

- [ ] **理解 Pipeline 结构**（核心中的核心）

  FA3 的 forward pipeline 大致是：
  ```
  Stage 0: TMA load K_tile[i+1] → SMEM buffer A
  Stage 1: TMA load V_tile[i]   → SMEM buffer B
  Compute: WGMMA(Q_reg, K_smem[i]) → S_reg
           softmax(S_reg) → P_reg
           WGMMA(P_reg, V_smem[i]) → O_reg

  关键：K 的 load 比 V 提前一步（因为要先算 S=QK^T 才能算 PV）
  ```

  问 AI 搞清楚：
  - 为什么是 2-stage 而不是 3-stage？（SMEM 容量：每个 tile 128×128×2bytes = 32KB，2 个 buffer 就 64KB，加上 Q 和 O 的空间，接近 SMEM 上限）
  - mbarrier 怎么协调 TMA 和 compute？（TMA 完成后 arrive barrier，compute 前 wait barrier）
  - Warp specialization：哪些 warp 负责搬运，哪些负责计算？

- [ ] **理解 Tile Size 选择**

  FA3 典型配置：`tile_M=128, tile_N=128, head_dim=128`

  问 AI 分析：
  - 为什么 tile_M=128？（一个 tile 正好占满一个 SM 的 warp 数）
  - 为什么不用 tile_M=256？（SMEM 不够 + wave quantization 变差）
  - 为什么不用 tile_M=64？（Tensor Core 利用率下降，每次 WGMMA 的数据复用变少）

- [ ] **理解 WGMMA 的使用方式**

  FA3 用 WGMMA 做两个 GEMM：
  ```
  S = Q × K^T    (Q 在 register, K 在 SMEM)
  O = P × V      (P 在 register, V 在 SMEM)
  ```

  问 AI 搞清楚：
  - 为什么 Q 放 register 而 K 放 SMEM？（WGMMA 的 A operand 可以来自 register，B 必须来自 SMEM）
  - WGMMA 的 MNK shape 是什么？（通常 M64_N128_K16 或类似）
  - 一个 tile 内需要多少次 WGMMA 调用？

- [ ] **理解 Online Softmax**

  FA3 不需要先算完整个 S 矩阵再做 softmax，而是 tile-by-tile 地做：
  ```
  for each K_tile:
      S_tile = Q × K_tile^T
      m_new = max(m_old, rowmax(S_tile))
      correction = exp(m_old - m_new)
      O = O * correction + exp(S_tile - m_new) × V_tile
  ```

  问 AI 搞清楚：
  - 这个算法的数值稳定性为什么 OK？
  - correction factor 的计算开销有多大？（很小，是 elementwise 操作）
  - 为什么这比"两遍扫描"（先算 max，再算 softmax）快？（省了一次 global memory 读写）

- [ ] **理解 Shared Memory Swizzle**

  FA3 的 SMEM layout 不是简单的行优先，而是用了 swizzle 消除 bank conflict：
  ```
  // CuTe 的 Swizzle<3, 3, 3> 表示：
  // 用地址的 bit[6:8] XOR bit[3:5] 来重排 SMEM 地址
  // 效果：连续 warp 的线程访问不同 bank
  ```

  问 AI 搞清楚：
  - 为什么 naive 行优先 layout 会有 bank conflict？
  - Swizzle 的 XOR 操作具体怎么消除 conflict？
  - 在 ncu 中怎么验证 swizzle 是否生效？（看 bank conflict 计数）

#### Step 4: 实验性修改 + 性能对比（3-5 天）

这一步的目标不是"优化 FA3"（它已经很优了），而是**通过故意改差来验证你对设计决策的理解**。

- [ ] **实验 1: 去掉 swizzle，看 bank conflict 变化**
  - 修改 SMEM layout 为 naive 行优先
  - ncu 对比：`l1tex__data_bank_conflicts` 应该大幅增加
  - 性能下降多少？→ 量化 swizzle 的价值

- [ ] **实验 2: 改 tile size，看性能曲线**
  - tile_M = 64 / 128 / 256（如果 SMEM 够的话）
  - 记录每个配置的 kernel time、occupancy、TC utilization
  - 画图：tile_M vs performance，理解最优点在哪里

- [ ] **实验 3: 改 pipeline depth，看 stall 变化**
  - 如果能改成 1-stage（无 pipeline），看 `long_scoreboard` 暴增多少
  - 如果能改成 3-stage，看是否有提升（可能 SMEM 不够）

- [ ] **实验 4: 对比不同 head_dim 的行为**
  - head_dim=64 vs 128 vs 256
  - 预期：head_dim 越大，每个 tile 的计算量越大，TC 利用率越高
  - 但 SMEM 用量也越大，可能需要减小 tile_M

#### Step 5: Backward Kernel 分析（3-5 天）

- [ ] **Profile backward kernel**
  ```bash
  ncu --set full --kernel-name "regex:flash_bwd.*" \
      --launch-skip 2 --launch-count 1 \
      -o fa3_bwd python your_benchmark.py
  ```

- [ ] **理解 backward 为什么比 forward 慢**
  - Forward: 2 个 GEMM（QK^T 和 PV）
  - Backward: 5 个 GEMM（dO×V^T, dO×P^T, dS×K, dS×Q, 加上 recompute S=QK^T）
  - 更多的 SMEM 需求 → 更小的 tile 或更低的 occupancy
  - 更复杂的 pipeline → 更多的 barrier 等待

- [ ] **对比 forward vs backward 的 ncu 数据**

  | 指标 | Forward | Backward | 分析 |
  |------|---------|----------|------|
  | kernel time | ? | ? | bwd 通常 1.5-2x fwd |
  | SM SOL% | ? | ? | bwd 可能更低（pipeline 更复杂） |
  | TC util% | ? | ? | bwd 有更多非 MMA 操作 |
  | top stall | ? | ? | bwd 可能 `wait` 更多 |
  | registers/thread | ? | ? | bwd 通常更高 |

#### Step 6: 写分析报告（2-3 天）

写一份可以在面试中讲 10-15 分钟的报告，结构如下：

```markdown
# FlashAttention 3 性能分析报告

## 1. 理论分析
- FLOP 计算公式和推导
- Arithmetic Intensity vs seq_len 的关系
- 理论峰值 TFLOPS 和 roofline 位置

## 2. 实测数据
- 不同 seq_len 下的 TFLOPS 和 efficiency%
- ncu 关键指标表格
- 瓶颈类型随 seq_len 的变化

## 3. 设计决策分析
- Pipeline 结构：为什么 2-stage + warp specialization
- Tile size：为什么 128×128
- Memory layout：swizzle 的必要性和效果
- Online softmax：算法选择的性能原因

## 4. 实验验证
- 去掉 swizzle 的性能下降
- 不同 tile size 的性能曲线
- Forward vs Backward 的效率差异分析

## 5. 进一步优化的可能方向
- 当前瓶颈在哪里（具体 stall reason + 数值）
- 可能的改进（如果有的话）
- 为什么已经接近极限（量化论证）
```

#### 需要在此阶段掌握的技术（边做边学）

| 技术 | 什么时候学 | 怎么学 |
|------|-----------|--------|
| TMA (Tensor Memory Accelerator) | Step 2 看到 TMA 相关指标时 | 问 AI 讲原理；深入看 [Colfax TMA 教程](https://research.colfax-intl.com/tutorial-hopper-tma/) |
| WGMMA 指令 | Step 3 分析 Tensor Core 利用率时 | 问 AI 讲 WGMMA vs HMMA 的区别和限制 |
| Software Pipeline + mbarrier | Step 3 理解 pipeline 结构时 | 问 AI 讲 + 看 CUTLASS `MainloopSm90` 的结构 |
| Shared Memory Swizzle | Step 4 做 swizzle 实验时 | 问 AI 讲 CuTe `Swizzle<B,M,S>` 原理 |
| Online Softmax 算法 | Step 3 理解 softmax 实现时 | 问 AI 推导数学公式 |
| CuTe Layout 系统 | 读源码遇到 `Layout<Shape, Stride>` 时 | 问 AI 讲 CuTe 的 layout algebra |

#### 过关标准

✅ 能完整解释 FA3 forward kernel 的 pipeline 结构（画图 + 文字）
✅ 能说清楚每个设计决策（tile size、pipeline depth、swizzle、warp specialization）的性能原因
✅ 有实验数据证明你的理解（改差 → 性能下降 → 验证假设）
✅ 有一份结构化的分析报告，能在面试中讲 10 分钟
✅ 能回答"如果要进一步优化 FA3，你会怎么做？"这个问题

---

### Phase 3: 拓展实战 — 更多 Kernel 类型覆盖

**目标**：覆盖更多 kernel 类型，建立"看到任何 kernel 都能分析"的能力，积累更多面试素材。

**预计时间**：持续进行，每个子项目 3-7 天

#### 实战项目菜单（选 2-3 个做）

| 项目 | 学到什么 | 为什么值得做 | 难度 |
|------|----------|-------------|------|
| **Fused Softmax** | memory-bound 极致优化、online algorithm、warp shuffle | 面试高频题，且和 FA 直接相关 | ⭐⭐ |
| **RMSNorm** | reduction kernel、向量化、occupancy 调优 | LLM 推理必备算子，简单但能练完整闭环 | ⭐⭐ |
| **Quantized GEMM (INT8/FP8)** | 混合精度 Tensor Core、dequant 开销分析 | 量化推理是当前热点 | ⭐⭐⭐ |
| **Custom Triton kernel** profile | Triton 生成代码质量分析、和手写 CUDA 对比 | 很多公司用 Triton，能分析其瓶颈很加分 | ⭐⭐ |
| **CUTLASS 3.x GEMM 配置调优** | tile shape / stage / cluster 选择的影响 | 理解 CUTLASS 设计哲学 | ⭐⭐⭐⭐ |

#### 每个项目的标准流程

```
1. 写/找到 baseline 实现
2. 手算理论上限（FLOP、bytes、AI、理论时间）
3. ncu --set full profile
4. 计算 efficiency% = 理论时间 / 实际时间
5. 识别 top-1 瓶颈 + root cause
6. 实施一个优化
7. ncu A/B compare 验证
8. 重复 5-7 直到满意或瓶颈转移
9. 写 3-5 句话总结：做了什么、瓶颈是什么、优化了多少
```

#### 微架构知识（按需深入，不需要专门花时间系统学）

以下内容在你遇到"最后 5% 怎么也优化不动"时再深入：

- **Warp Scheduler Issue 策略**：eligible warps < 1 时才需要关心
- **Register Bank Conflict**：当 SASS 中出现不合理的 stall 时才需要看
- **SASS 指令调度**：当你怀疑编译器生成了次优代码时才需要看
- **CuAssembler / maxas**：99% 的情况不需要，了解存在即可

#### 推荐的深入阅读（遇到具体问题时查阅）

| 场景 | 去看什么 |
|------|----------|
| 想理解你 GPU 的精确 latency/bandwidth 数字 | [Hopper Microbenchmark 论文](https://arxiv.org/abs/2402.13499) 的对应表格 |
| 想理解 CUTLASS 的 pipeline 设计 | CUTLASS 源码 `include/cutlass/gemm/collective/sm90_mma_tma_gmma_*.hpp` |
| 想了解 Blackwell 新特性 | [Blackwell Microbenchmark](https://arxiv.org/abs/2507.10789) + CUTLASS SM100 examples |
| 想看工业级优化演讲 | [GTC 2025 Memory BW](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/) |
| ncu 某个指标看不懂 | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) |

#### 过关标准

✅ 至少完成 2 个不同类型 kernel 的完整优化闭环
✅ 能在 30 分钟内对一个陌生 kernel 完成 profile + 瓶颈定位 + 优化方向建议
✅ 面试时能讲 2-3 个"我优化了 XX kernel，从 Y% 提升到 Z%"的完整故事

---

### 总结：学习路径一览

```
Phase 1 (3-5天)          Phase 2 (3-4周)              Phase 3 (持续)
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ 热身：3类kernel │     │ 主线：FA3 深度分析    │     │ 拓展：更多kernel  │
│ 对比 + HGEMM  │ ──→ │ 源码理解 + 实验验证   │ ──→ │ 类型覆盖          │
│ 优化闭环体验   │     │ + 分析报告           │     │ + 面试故事积累     │
└──────────────┘     └─────────────────────┘     └──────────────────┘
       │                        │                          │
       ▼                        ▼                          ▼
  能判断瓶颈类型          能深度分析工业级kernel       能应对任何kernel
  能跑完一次闭环          能讲清楚设计决策原因        有多个优化故事
```

**核心原则**：
1. **先动手，遇到不懂的再学理论**（而不是先读完所有论文再动手）
2. **每次只改一个变量**，用 ncu 数据验证你的假设
3. **记录每次优化的数据**，这就是你面试时的素材
4. **理论知识优先问 AI**，只有需要反复查阅的参考资料才去读原文
5. **FA3 是主线**，其他项目是补充覆盖面用的

---

## 五、核心思维模型

### 5.1 三个必须回答的问题

对每一个你 profile 的 kernel，都应该能清晰回答：

1. **理论上限是多少？**
   - 这个 kernel 搬了多少 byte（从 HBM），做了多少 FLOP
   - Arithmetic Intensity = FLOP / Byte
   - 理论执行时间 = max(Byte / HBM_BW, FLOP / Peak_FLOPS)
   - 如果中间层也是瓶颈，用 Hierarchical Roofline 逐层算

2. **实际达到了多少？**
   - ncu 报告的 kernel time
   - 实际 throughput（memory 和 compute）各是峰值的百分之几
   - 实际 / 理论 = efficiency%

3. **差距的 top-1 原因是什么？**
   - warp stall reason 分布的 top-1
   - 哪一层内存的 throughput 最先成为瓶颈
   - 有多少指令是非计算的（address calc、type convert、branch）

### 5.2 性能优化的优先级原则

```
1. 算法层面（最高优先级）
   - 减少总的 FLOP 和 memory traffic
   - 选择更好的 tiling / 分解策略

2. 数据搬运层面
   - coalesced access
   - 消除 bank conflict
   - software pipelining（compute-memory overlap）
   - L2 tile traversal 优化

3. 计算层面
   - 用 Tensor Core 替代标量 FMA
   - 向量化 load/store
   - 减少非计算指令
   - 充分展开循环

4. 指令调度层面（最低优先级，最后手段）
   - ILP vs occupancy 调优
   - register bank conflict
   - SASS 级指令重排
```

**原则：先改回报最大的，高层优化的收益远大于底层微调。**

### 5.3 常见误区

| 误区 | 事实 |
|------|------|
| occupancy 越高越好 | 不一定。大 tile + 低 occupancy 常常更快（更好的数据复用和 ILP） |
| L1 cache 能帮到 global load | Ampere+ 默认 global load 走 L2 不走 L1（除非用 `__ldg` 或 `const __restrict__`） |
| 减少指令数 = 更快 | 不一定。关键是减少在**关键路径**上的指令延迟 |
| bank conflict 只影响 shared memory | register file 也有 bank conflict，只是更隐蔽 |
| 用 `#pragma unroll` 就够了 | 展开太多反而增加 register pressure 导致 spill；需要找到平衡点 |
| kernel 越大越好 | kernel 太大可能导致 register spill 或 SMEM 不够分；有时拆成多个 kernel 更快 |
| 峰值 FLOPS 就是实际上限 | 还需考虑指令 mix（非 FMA 指令也占 issue slot）、pipeline bubble 等 |