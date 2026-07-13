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