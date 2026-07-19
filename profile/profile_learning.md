
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