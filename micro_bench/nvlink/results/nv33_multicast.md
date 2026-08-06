### nv33_multicast — NVSwitch 里的组播与在网归约 (NVLink SHARP)

**假设**：Hopper + NVSwitch 提供 multicast object，`multimem.st` 的复制、`multimem.ld_reduce` 的求和据说发生在**交换机内部**。官方只给了 API，没有任何行为描述。要验证「复制/求和到底在哪做」，只有一个办法：数线上字节。

**方法**：用 driver API 建 multicast object（`cuMulticastCreate` + `cuMulticastAddDevice` + `cuMulticastBindMem`），4 张卡全加进组，每张卡再各自 map 一个 unicast 视图用于校验。
决定性的一步是 C 段：由 GPU0 发起 `multimem.st`，然后读**四张卡各自**的 NVML Data Tx/Rx 计数器。
- 若复制在交换机内做 → GPU0 的 Tx 只有 1 份 payload，而 GPU1/2/3 的 Rx 各有 1 份。
- 若复制在 GPU0 做 → GPU0 的 Tx 应该是 3 份。
对照组是手写的 unicast 三连发，两者逻辑 payload 完全相同。

**实测数据**

A) 能力与粒度：

| 项 | 值 |
|---|---|
| MULTICAST_SUPPORTED | 1（4 张卡全部） |
| multicast granularity minimum | 2097152 B = **2 MB** |
| multicast granularity recommended | 2097152 B = 2 MB |
| VMM allocation granularity | 2097152 B = 2 MB |

B) 正确性：GPU0 发一条 `multimem.st.global.v4.f32`，四张卡本地内存首元素都读到 `1 2 3 4` → 组播真的落到 4 张卡。

C) 线上字节数（逻辑 payload 均为 2147 MB）：

| 负载 | GPU0 Tx | GPU0 Rx | GPU1 Rx | GPU2 Rx | GPU3 Rx |
|---|---|---|---|---|---|
| `multimem.st` 组播给 4 卡 | **2147 MB** | 2147 MB | 2147 MB | 2147 MB | 2147 MB |
| unicast 手动广播给 3 卡 | **6442 MB** | 0 MB | 2147 MB | 2147 MB | 2147 MB |

E) `multimem.ld_reduce` 的线上字节（逻辑读回 2147 MB 归约结果）：

| GPU | Tx | Rx |
|---|---|---|
| GPU0（发起方） | 2147 MB | **2147 MB** |
| GPU1 | 2147 MB | 0 MB |
| GPU2 | 2147 MB | 0 MB |
| GPU3 | 2147 MB | 0 MB |

D) 带宽：

| 操作 | 逻辑 GB/s | 线上 GB/s |
|---|---|---|
| unicast 写 1 个 peer | 370.5 | 370.5 |
| unicast 广播给 3 个 peer | **120.2** | 360.6 |
| `multimem.st` 组播给 4 张卡 | **277.4** | 277.4 |
| `multimem.ld_reduce` 4 卡在网求和 | 151.8 | 151.8 |
| `multimem.red` 4 卡在网累加（无返回） | 277.3 | 277.3 |

**结论**

1. **复制确实发生在 NVSwitch 内部，这是直接证据**。C 段：`multimem.st` 时 GPU0 只发出 2147 MB，而 GPU1/2/3 各收到 2147 MB。总交付量 3×2147 MB 但发起方只付了 1 份的发送带宽。对照组 unicast 广播 GPU0 要发 6442 MB。**发送侧线上流量降到 1/3**。

2. **发起方自己那一份也走了交换机绕回来**：`multimem.st` 时 GPU0 的 Rx 也是 2147 MB。GPU0 明明是组成员、数据就在本地，但它自己的副本仍然是从 switch loopback 回来写进本地显存的，没有走本地捷径。这解释了为什么组播只到 277 GB/s 而不是 370：GPU0 的链路要同时扛 277 Tx + 277 Rx。

3. **4 路广播的实际加速比 = 277.4 / 120.2 = 2.31×**。理论上限是 3×（省掉 3 份发送里的 2 份），拿不到 3× 的差额正好是上一条的 loopback 开销。

4. **在网归约是真的在交换机里做加法**。E 段：4 张卡各 Tx 2147 MB 进 switch，只有发起方 Rx 2147 MB 出来。若归约在 GPU0 做，GPU0 就得 Rx 3×2147 = 6442 MB。**接收侧线上流量降到 1/3**。注意 GPU0 自己也 Tx 了 2147 MB —— 它本地的那份数据同样要送进 switch 参与求和。

5. **`ld_reduce`（151.8 GB/s）比 `red`（277.3 GB/s）慢 1.83×**，而 `red` 与 `multimem.st`（277.4）几乎完全相同。说明交换机里的加法器不是瓶颈（否则 `red` 也会慢），慢的是 `ld_reduce` 需要等归约结果返回的那个往返。**要 all-reduce 就别用 ld_reduce 做 gather 侧，用 red 做 scatter 侧。**

6. **组播的最小粒度是 2 MB**，比一般 VMM 分配粒度大一个量级。小于 2 MB 的 buffer 根本用不了组播，这是 NCCL 为什么只在大消息上启用 NVLS 算法的硬件原因。

**不确定性**
- 只有 4 张卡，无法验证组播加速比是否随组大小线性增长（8 卡时理论加速比应该是 7×）。
- `multimem.ld_reduce` 只测了 `.add.v4.f32`。其它 op（min/max）和精度（bf16/f16）没测，交换机 ALU 对不同类型的吞吐可能不同。
- Data 计数器不区分「组播复制出的字节」和「普通字节」，结论依赖于 payload 与计数器的精确对应关系（这一点在 nv32 里已独立验证过）。
