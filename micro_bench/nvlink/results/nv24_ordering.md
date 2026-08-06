### nv24_ordering — 1.2 M × 8 组合共 960 万次 MP 观测，**零违例**；但 relaxed 写侧只要 13.7 cyc/it 而 fence/release 要 1900 cyc/it，正确性代价是 **139×**

**假设**

经典 message-passing litmus：GPU0 写 `data=42` 再写 `flag=1`，GPU1 自旋等 `flag==1` 后读 `data`。若读到 `data != 42`，说明观察到了 store-store 重排。跨 NVLink 的两次写如果走不同路径（不同 link、不同 L2 slice），到达顺序完全可能颠倒。想量化：(1) 无 fence 时违例率有多高；(2) 加 fence / release-acquire 能否消除；(3) 正确性要付多少性能代价。

**方法**

- `data` 和 `flag` 分处**不同的 128 B cache line**（`struct Slot` 中间垫 31 个 uint），避免同 line 顺带可见造成假阴性。
- **每次迭代用一个全新的 slot**，且每轮开始把整个 slot 数组 memset 清零。这是关键 —— 如果复用同一个 slot，上一轮残留的 `data=42` 会让违例永远观察不到。
- slot 分配在 **GPU1 显存**（reader 本地，writer 远端写）。这是最容易暴露重排的方向：writer 的两次写都要过 NVLink，可能走不同 link。
- 写侧 4 种 × 读侧 2 种 = 8 组合，每组合 N=20000 slots × ROUNDS=60 = **1.2 M 次观测**。
- 死锁防护：自旋 `SPIN_MAX = 2e6` 上限，超时单独统计为 timeout 而**不计入违例**（超时 ≠ 重排），且不中断测试继续下一轮。实测 timeout 全为 0。
- 另做 store-store 观察：GPU0 连续写 A、B 到不同 128 B 块，GPU1 见 B 时查 A。

**实测数据**

A) MP litmus，每组合 1.2 M 次观测

| 写侧 / 读侧 | 总迭代 | 违例数 | 超时数 | 写 cyc/it | 读 cyc/it |
|---|---|---|---|---|---|
| relaxed st / relaxed ld | 1200000 | **0** | 0 | **13.7** | 864.7 |
| relaxed st / acquire ld | 1200000 | **0** | 0 | 13.6 | 872.3 |
| fence.sys / relaxed ld | 1200000 | **0** | 0 | **1902.9** | 1903.7 |
| fence.sys / acquire ld | 1200000 | **0** | 0 | 1903.3 | 1903.9 |
| release.sys / relaxed ld | 1200000 | **0** | 0 | **1898.0** | 1898.8 |
| release.sys / acquire ld | 1200000 | **0** | 0 | 1899.2 | 1899.9 |
| volatile / relaxed ld | 1200000 | **0** | 0 | 13.6 | 864.4 |
| volatile / acquire ld | 1200000 | **0** | 0 | 13.6 | 872.3 |

B) store-store 乱序观察

```
总迭代 1200000, 乱序(见B时A未到) 0 次, 超时 0 次, 乱序率 0.000e+00
```

**结论**

1. **在 960 万次跨 GPU MP 观测中未观察到任何 store-store 重排违例（8 组合 × 1.2 M，含 240 万次完全无同步原语的 relaxed/relaxed 与 volatile 组合）。** 依据 A 段全表「违例数」列均为 0，且「超时数」也全为 0（说明每次 flag 都真的等到了，不存在「因为超时跳过所以没看到违例」的假阴性）。B 段独立的 store-store 观察同样 0 次，乱序率上界 < 1/1.2e6 = **8.3e-7**。

2. **零违例的真正原因不是硬件保证有序，而是读侧太慢掩盖了窗口。** 这是本实验最重要的解读，依据 A 段第 1 行的两个数字对比：**写侧 relaxed 只要 13.7 cyc/it，读侧却要 864.7 cyc/it**。读侧比写侧慢 63×，意味着 writer 早就把 data 和 flag 都发出去了，reader 才刚开始轮询。**重排窗口（data 与 flag 到达时刻之差）远小于 reader 的检测粒度 864.7 cyc**，所以观察不到。这不能证明硬件不会重排，只能说在这种「writer 快、reader 慢」的自然节奏下重排不可观测。要真正逼出违例需要让两次写的传播路径差异最大化并把 reader 轮询粒度压到几十 cycle，本实验没做到。

3. **`fence.sys` 和 `release.sys` 的代价是把写侧从 13.7 cyc 拉到 ~1900 cyc，慢 139×。** 依据 A 段：relaxed 13.7 → fence.sys 1902.9 / release.sys 1898.0 cyc/it。这 1900 cyc 与 nv20 测到的远端完成延迟（`red + fence.sys` 远端 1758.9 cyc、`atom` 远端 1705 cyc）高度吻合 —— **fence 的作用就是把「posted 的写」变成「必须等确认的写」，代价正好是一个完整的跨卡往返**。nv20 已证明裸写是 posted 的（远端 `st.global` 发射只要 11.5 cyc），fence 把这个优势全部抵消掉。

4. **`fence.sys` 与 `st.release.sys` 代价几乎相同（1902.9 vs 1898.0，差 0.26%），应优先选 release。** 两者都要等一个往返，但 `release` 语义更精确（只约束这一个 flag 的写），在有多个独立 flag 时编译器和硬件有更多优化空间。既然价格一样，没有理由用更粗的 `__threadfence_system()`。

5. **读侧 acquire 相对 relaxed 只贵 7.6 cyc（872.3 vs 864.7，+0.9%），几乎免费。** 依据 A 段第 1、2 行。与 nv20 的结论一致（`.sys` scope 修饰符在已跨卡路径上零额外开销）。**读侧应无条件用 `ld.acquire.sys`** —— 花 0.9% 买正确性保证。

6. **`volatile` 与 relaxed 在性能和违例率上完全等价（13.6 vs 13.7 cyc，都是 0 违例），它不提供任何跨 GPU 序保证。** 依据 A 段第 7、8 行与第 1、2 行逐项对比。`volatile` 只阻止编译器优化掉访存，不生成任何 fence 指令，**不能用它来做跨 GPU 同步**。这是一个常见误用，数据在这里给了明确否定。

7. **性能建议（可直接落地）**：跨 GPU 生产者-消费者应采用「写侧 `st.release.sys` + 读侧 `ld.acquire.sys`」组合（A 段第 6 行，1899.2 / 1899.9 cyc）。虽然写侧付 139× 代价，但这是每个 chunk 只付一次的固定开销 —— 只要 chunk 足够大，摊薄后可忽略（这正是 nv25 要量化的）。

**不确定性**

- **「零违例」是弱结论，不能宣称硬件保证有序。** 如结论 2 所述，读侧 864.7 cyc 的轮询粒度可能完全掩盖了重排窗口。更强的实验设计需要：(a) 让 writer 的两次写落到明确不同的 NVLink link 或不同 L2 slice（利用 nv32 测出的 256 B 条带跨度）；(b) 用多个 reader 线程以更细粒度轮询；(c) 在 writer 两次写之间插入可变延迟扫描，找出重排窗口的时间尺度。这些都没做，时间不够。
- 只测了「slot 在 GPU1（writer 远端写、reader 本地读）」一个方向。反方向（slot 在 GPU0，writer 本地写、reader 远端读）没测，那个方向 reader 每次轮询要付 826 ns 远端读，窗口更容易被掩盖，但 writer 侧的两次本地写顺序性质不同，值得单独做。
- 1.2 M 次给出的违例率上界是 8.3e-7。若真实违例率在 1e-8 量级，本实验的样本量不足以发现。要把上界压到 1e-9 需要跑 10 亿次迭代，按当前 864.7 cyc/it 约需 7 分钟/组合，时间预算内做不完全部 8 个组合。
- 写侧 13.7 cyc/it 说明 writer kernel 几乎是纯发射，没有背压。这意味着 1.2 M 次写可能在链路上大量堆积，实际的两次写间隔比代码顺序更紧凑，反而**降低**了重排概率。用带背压的 writer（每 N 次插一个 fence）重测可能得到不同结果。
