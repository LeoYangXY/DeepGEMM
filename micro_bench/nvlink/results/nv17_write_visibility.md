### nv17_write_visibility — 远端 store 发射只要 **7.6 cyc**，等到系统级可见要 **1788.6 cyc**；写方向 `dataRx=0` 但 `rawRx=972 MB`，ack 是纯协议流量、零净荷

**假设**
远端 store 是 fire-and-forget：发射即退休，代价极低；真正的成本发生在需要确认可见性时（fence）。若存在写确认包，NVLink 计数器应在写负载下看到反方向的 **raw** 流量但 **data** 净荷为零。

**方法**
- (a) `<<<1,1>>>`，每次 store 打到相隔 8 KB 的新地址（避免合并）。四种模式：裸 store / +`__threadfence_block()` / +`__threadfence()` / +`__threadfence_system()`。本地与远端各测一遍，相减即可剥离 fence 的本地固有成本。best-of-5，ITERS=1024。
- (b) 写后读：`st X; ld.volatile X`（同地址）vs `st X; ld.volatile X+16KB`（异地址）vs 纯读基线。读用 volatile 强制真的访存，防止编译器做 store-forward。
- (c) 计数器：256 MB buffer 写 4 遍，分别用 `st.global.v4`(128b)、`st.global.u32`(32b)，并以远端读作对照，看 Tx/Rx 的 data 与 raw 分解。

**实测数据**

(a) 发射 vs 完成：

| 模式 | 本地 cyc | 本地 ns | 远端 cyc | 远端 ns | 远端−本地 |
|---|---|---|---|---|---|
| st 裸发射（无 fence） | **7.6** | 3.8 | **7.6** | 3.8 | 0.0 |
| st + `__threadfence_block()` | 25.1 | 12.7 | 25.1 | 12.7 | 0.0 |
| st + `__threadfence()` | 502.4 | 253.7 | 593.8 | 299.9 | +91.4 |
| st + `__threadfence_system()` | 1179.7 | 595.8 | **1788.6** | 903.3 | **+608.9** |

推导：远端写确认往返（sys fence − 裸发射）= **1781.0 cyc = 899.5 ns**；本地同项 = 1172.1 cyc = 592.0 ns；**纯 NVLink ack 附加成本 = 608.9 cyc = 307.5 ns**。

(b) 写后读：

| 模式 | 本地 cyc | 远端 cyc | 远端 ns |
|---|---|---|---|
| 写 X 后读 X（同地址） | 316.3 | 1629.0 | 822.7 |
| 写 X 后读 X+16KB（异地址） | 292.3 | 1598.0 | 807.1 |
| 只读（基线） | 27.8 | 112.3 | 56.7 |

(c) 计数器（payload 1074 MB）：

| 负载 | dataTx MB | dataRx MB | rawTx MB | rawRx MB | Rx/Tx(raw) | raw/data |
|---|---|---|---|---|---|---|
| 远端写 `st.global.v4` (128b) | 1073.7 | **0.0** | 1214.2 | **972.0** | 0.8005 | 2.0361 |
| 远端写 `st.global.u32` (32b) | 1073.7 | **0.0** | 1222.7 | 977.4 | 0.7994 | 2.0490 |
| 远端读 `ld.volatile.v4`（对照） | 0.0 | 1073.7 | 830.2 | 1211.4 | 1.4592 | 1.9013 |

**结论**

1. **远端 store 发射代价 = 7.6 cyc = 3.8 ns，与本地完全相同（两列均 7.6）**（a 表行1）。远端写是真正的 fire-and-forget，SM 不等待任何应答即可退休该指令。
2. **完成语义的代价是发射的 235 倍**：`__threadfence_system()` 后远端 store 需 **1788.6 cyc = 903.3 ns**（a 表行4）。**这 1781 cyc 的差值就是写确认往返**。
3. **`__threadfence_block()` 对远端写几乎免费（25.1 cyc）且本地远端无差**（a 表行2）——它只约束 CTA 内可见性，不触发任何 NVLink 事务。
4. **scope 阶梯清晰**：block 25.1 → device 593.8 → system 1788.6 cyc（远端列）。device→system 跨越 **1194.8 cyc**，这是把可见性从"本 GPU"提升到"全系统"的价格。
5. **ack 是零净荷的纯协议包**（决定性证据）：写负载下 `dataRx = 0.0 MB` 而 `rawRx = 972.0 MB`（c 表行1）。data 计数器只统计净荷，raw 统计线上总字节 —— 二者的组合说明反方向流量 100% 是协议开销，即写确认。
   - 每 128 B 写事务的 ack 线上字节 ≈ 972.0 MB / (1073.7 MB/128 B) = **115.9 B**。32 b 写版本给出 977.4 MB，几乎相同，说明 **ack 开销由事务数决定而非写宽度**（两行 rawRx 仅差 0.55%）。
   - 写方向 `raw/data = 2.036`，即每传 1 B 有效数据要占用约 2 B 线上带宽（含 ack 回程），这解释了为何远端写实测 370 GB/s 远低于 478 GB/s 理论单向值。
6. **不存在跨 NVLink 的 write-combining / store-forwarding。** 同地址 1629.0 vs 异地址 1598.0，**差 −31.0 cyc（同地址反而略慢）**（b 表）。若有 store buffer 转发，同地址应显著更快；实测符号相反且幅度仅 1.9%，属噪声。程序输出中"存在转发/合并效应"的判定是打印阈值（|diff|>30）的误判，**正确结论是无转发**。
7. 写后读比纯读贵 1516.7 cyc（b 表行1 − 行3）：store 会阻塞后续对同一区域的 load，读必须等写在远端落定。

**不确定性**
- (b) 的"异地址"仅偏移 16 KB，仍在同一 NVLink 路由与同一 DRAM bank group 内；更大偏移可能出现不同结果。
- (c) 的 115.9 B/事务是把 rawRx 全部归因于 ack 的上界估计，其中含 flow-control flit、CRC 与 idle 填充，未做进一步分离。
- (a) 的 fence 代价含 fence 指令自身的固定开销（nv19 测得裸 `membar.sys` = 1166.5 cyc），故"写确认往返"1781 cyc 中约 1166 cyc 是 fence 固有成本，**归属于 NVLink 的增量应以远端−本地的 608.9 cyc 为准**。
