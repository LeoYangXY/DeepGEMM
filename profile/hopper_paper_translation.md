# Benchmarking and Dissecting the NVIDIA Hopper GPU Architecture
## 逐段详细翻译 + 讲解

> 原文：arXiv:2402.13499（2024-02）
> 作者：Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du（港科大广州）, Qiang Wang（哈工大深圳）, Xiaowen Chu（港科大）
> 本文件为原文的**逐段翻译**，并附带讲解（括号内为译者注/解释）。

---

## 封面 / 摘要（Abstract）

**英文**
> Graphics processing units (GPUs) are continually evolving to cater to the computational demands of contemporary general-purpose workloads, particularly those driven by artificial intelligence (AI) utilizing deep learning techniques. A substantial body of studies have been dedicated to dissecting the microarchitectural metrics characterizing diverse GPU generations, which helps researchers understand the hardware details and leverage them to optimize the GPU programs.
> However, the latest Hopper GPUs present a set of novel attributes, including new tensor cores supporting FP8, DPX, and distributed shared memory. Their details still remain mysterious in terms of performance and operational characteristics. In this research, we propose an extensive benchmarking study focused on the Hopper GPU. The objective is to unveil its microarchitectural intricacies through an examination of the new instruction-set architecture (ISA) of Nvidia GPUs and the utilization of new CUDA APIs. Our approach involves two main aspects. Firstly, we conduct conventional latency and throughput comparison benchmarks across the three most recent GPU architectures, namely Hopper, Ada, and Ampere.
> Secondly, we delve into a comprehensive discussion and benchmarking of the latest Hopper features, encompassing the Hopper DPX dynamic programming (DP) instruction set, distributed shared memory, and the availability of FP8 tensor cores. The microbenchmarking results we present offer a deeper understanding of the novel GPU AI function units and programming features introduced by the Hopper architecture. This newfound understanding is expected to greatly facilitate software optimization and modeling efforts for GPU architectures. To the best of our knowledge, this study makes the first attempt to demystify the tensor core performance and programming instruction sets unique to Hopper GPUs.

**中文翻译**
GPU 持续演进，以满足当代通用计算（尤其是基于深度学习技术的 AI）的算力需求。已有大量研究致力于"解剖"不同代 GPU 的微架构指标，帮助研究者理解硬件细节并利用它们优化 GPU 程序。然而，最新的 Hopper GPU 带来一组新特性——支持 FP8 的新张量核、DPX、以及分布式共享内存——它们在性能和运行特征上仍然很不透明。
本研究对 Hopper GPU 做了一次大规模的基准测试，目标是通过审视 NVIDIA GPU 的新指令集架构（ISA）和利用新的 CUDA API，揭开其微架构的奥秘。我们的方法包含两方面：第一，对最近三代 GPU 架构（Hopper、Ada、Ampere）做传统的延迟与吞吐对比基准；第二，深入讨论并对 Hopper 最新特性做基准测试，包括 Hopper 的 DPX 动态规划（DP）指令集、分布式共享内存，以及 FP8 张量核的可用性。我们给出的微基准结果，增进了对 Hopper 架构引入的新型 GPU AI 计算单元和编程特性的理解，预期将极大助力 GPU 架构的软件优化与建模工作。据我们所知，本研究是**首次尝试揭开 Hopper 专属张量核的性能与编程指令集之谜**。

**关键词（Index Terms）**：指令延迟、张量核、PTX、Hopper、DPX、异步执行、分布式共享内存

---
## I 引言（Introduction）

**中文翻译**
GPU 在加速各类应用（从神经网络到科学计算）的能力上实现了巨大飞跃，这一增长尤其被大语言模型（LLM）所推动——例如拥有超过 1500 亿参数的 GPT-3 就是典型代表。现代 GPU 架构（Ampere、Ada、Hopper）都集成了张量核、高带宽显存等前沿特性，已成为高性能计算集群的基石。

NVIDIA 每两年推出新架构并加入先进特性，但这些特性的**微架构细节往往有限、难以精确量化**，深入理解它们对应用性能的影响愈发必要。

张量核（TC）最初随 Volta 架构推出，专注用 FP16/FP32 加速深度神经网络；Ampere 扩展到稀疏化及 INT8/INT4/FP64/BF16/TF32 等更广精度；**Hopper 进一步加入 FP8 支持**，大幅加速 LLM 训练与推理。
然而，虽有研究 [2] 讨论过 Hopper 的张量核可编程性，但其汇编分析与微基准仍是跑在 Ampere/Turing 上，**专门针对 Hopper 张量核的研究仍缺位**。

除新张量核，Hopper 还引入了创新特性：
- **DPX 指令**（Dynamic Programming X）：加速大量动态规划算法，这些算法通常需要大量 min/max 操作来比较先前已算出的解；
- **分布式共享内存（DSM）**：支持 SM 之间的直接通信，包括跨多个 SM 的共享内存块的 load、store 和 atomic 操作；
- **增强的异步执行机制（TMA）**：支持 cluster 内 thread block 间的异步拷贝，提升效率。

但这些特性的**具体实现与性能细节在现有文献中仍未公开**。揭开这些技术细节，对程序员有效优化 AI 应用、利用现代 GPU 新特性至关重要。

---

## II 相关工作（Related Work）
**中文翻译（综述要点）**
- 从 Volta 时代起，就有大量研究"解剖"张量核：SASS 汇编级分析、CUBLAS/CUTLASS 库基准、数值行为（舍入模式与亚规范数）研究。
- 老的 `wmma` API 在算子形状上有限制，无法充分利用 Ampere/Hopper 的稀疏矩阵乘法；新一代 `mma` 自 Turing 起引入，Ampere 以上支持 `mma.sp` 稀疏。
- Sun 等人 [2] 对 Turing/Ampere 的 `mma`/`ldmatrix`/`mma.sp` 做了指令级微基准，揭示了张量核完整性能与新稀疏特性。
- **Hopper 的 `wgmma`/`mma` 的 SASS 代码更加多样，其用法与性能此前完全未被揭示**。
- 除性能外，能效也是常被讨论的因素（DVFS、跨厂商加速器对比）。
- 结论：随着 DPX、异步操作、分布式共享内存成为趋势，**对现代 GPU 架构做基准测试与解剖非常必要且紧迫**。

---

## III 方法论（Methodology）

### III-A 内存单元的访问延迟与吞吐
**中文翻译**
本小节关注两个内存性能指标：延迟（latency）与吞吐（throughput）。内存测试方法类似于 P-chase 微基准（最早见于 [28][29]）。

- **III-A1 L1 Cache**：延迟测试先用 `ca` 修饰符把数据从全局内存载入 L1，再用一个线程访问 L1 测延迟。吞吐测试同样用 `ca` 预热；因 L1 是 SM 独占的，只用一个含 1024 线程的 block 反复访问 L1，按访问数据量/耗时算带宽。
- **III-A2 Shared Memory**：测法与 L1 类似，只是无需用修饰符显式预热，直接声明共享内存即可测。共享内存只能在 block 内访问（本小节不考 DSM），同样单线程测延迟、1024 线程测带宽。
- **III-A3 L2 Cache**：延迟测试同 L1，但改用 `cg` 修饰符确保载入的是 L2。吞吐测试因 L2 被所有 SM 共享，用大量 block 访问 L2 来测带宽。
- **III-A4 Global Memory**：延迟测试先分配超过 L2 大小的内存以避免 L2 预取并初始化（固定步长 + 预热 TLB）；测试时启动 4 个连续线程各读 8 字节，凑成 32 字节内存读事务，再算每线程访存延迟。
### III-B 张量核的延迟与吞吐

**III-B1 张量核的演进**（表 I）

**中文翻译**
表 I 展示了张量核在精度、操作数形状、编程模式、执行模式上的演进：
- **精度**：Volta 仅支持 FP16 输入；Ampere/Ada/Hopper 扩展到 BF16、TF32、FP64、INT8、INT4、Binary 等。
- **编程接口**：Ampere 和 Ada 既可用老式 C 级 `wmma` API，也可用 PTX 级 `mma` 指令。但 `wmma` 无法充分利用张量核能力，而 `mma` 能用上 Ampere 起的稀疏矩阵乘。`wmma`/`mma` 在 Hopper 仍受支持，但我们**发现只有 `wgmma` 才能释放 Hopper 张量核的全部潜力**。

图 2 给出 `mma` 与 `wgmma` 示例：
- `mma` 计算 `D=A×B+C`，**由 1 个 CUDA warp（32 线程）同步执行**；形状 `m16n8k16` 或 `m16n8k8`。
- `wgmma`（Hopper）计算 `D=A×B+D`，**由 1 个 warp group（4 个 warp）异步执行**；形状 `m64nNk16`，N 可取 16/32/64/128/256。
- 关键优势：**`wgmma` 能直接从共享内存加载 A、B**，而 `mma` 必须先把矩阵放进寄存器。A、B 都从共享内存加载称 **"SS" 模式**；A 从寄存器加载称 **"RS" 模式**。
表 I 关键内容：
| 架构 | 精度 | 可编程性 | 模式 |
|---|---|---|---|
| Ampere | FP16,BF16,TF32,FP64,INT8,INT4,Binary | C: wmma / PTX: mma, mma.sp | Sync |
| Ada | 同上 + FP8 | C: wmma / PTX: mma, mma.sp | Sync |
| Hopper | FP16,BF16,FP8,TF32,FP64,INT8,Binary | C: wmma / PTX: mma, mma.sp (Sync) + wgmma, wgmma.sp (**Async**) | Sync + **Async** |

**III-B2 基准层级与性能指标**
**中文翻译**
我们在 **PTX 级**做张量核微基准，并把 PTX 反汇编成 **SASS** 以深入理解。
- **延迟**：从指令发出到结果可用所经过的时钟周期，称"完成延迟"。`mma` 由每 SM 一个 warp 发出；`wgmma` 由每 SM 一个 warp group（4 warp）发出；kernel 内执行 1024 次。
- **吞吐**：`吞吐 = 总OPS / 耗时`。我们**不用总时钟周期算吞吐**（GPU 频率可能变化），与 [2] 不同。
### III-C Transformer Engine（TE）
**中文翻译**
TE [31] 是 Hopper 后专为加速 Transformer [32] 设计的库，可利用 FP8，并提供 PyTorch [33] 下优化模块。
- **III-C1 Linear Layer**：TE 提供 `te.Linear` 以更高吞吐做 FP8 矩阵乘。做 FP8 时把输入和权重转 FP8：取输入最大绝对值作缩放因子 `inp_fp8 = inp_fp16 / scale`，做 `out_fp8 = inp_fp8 × w_fp8`，再 `out_fp16 = out_fp8 × scale`。**这步转换引入额外开销**；矩阵较小时转换开销占比大于 GEMM 本身。
- **III-C2 TransformerLayer**：TE 用算子融合利用 FP8，例如 `te.LayerNormMLP` 把 layernorm 与 MLP 合并，中间用 FP8 传输，省去格式转换开销。
- **III-C3 LLM Generation**：当前 TE 对 decode-only 模型支持不全。作者把 Llama 的 `nn.Linear`/`RMSNorm` 换成 `te` 版；用 ShareGPT 数据集，输入/生成长度各 128，batch 8，用每秒吞吐 `(input_len+output_len)/time` 评估。
### III-D 新 CUDA 编程特性

**III-D1 DPX**
**中文翻译**
NVIDIA 从 CUDA 12 起提供 DPX 函数加速动态规划；**Hopper 上这些函数是硬件加速的**。测试聚焦延迟与吞吐：单线程反复发指令测延迟；一个 block 反复发测每 SM 吞吐；**改变 block 数**观察吞吐与 block 数关系以**定位 DPX 加速硬件位置**。

**III-D2 Asynchronous Data Movement（异步数据搬运）**
**中文翻译**
异步执行是 Ampere 亮点：用 `cuda::memcpy_async` 实现全局↔共享内存非阻塞传输，让计算与传输重叠。Hopper 用 **TMA（Tensor Memory Accelerator）** 增强复杂异步拷贝。
作者对比两种实现（官方样例 `globalToShmemAsyncCopy`，矩阵 A 宽/B 高固定 2048）：
- `SyncShare`：传统分块 + **同步**拷贝；
- `AsyncPipe`：**异步**搬运 + 双缓冲两级流水线（共享内存缓冲加倍），计算与拷贝重叠。
block 大小 8×8 到 32×32 变化以评估对 warp 并发影响。
**III-D3 Distributed Shared Memory（DSM）**
**中文翻译**
Hopper 在 cluster 内有 **SM-to-SM 直连网络**，使一个 block 的线程能访问另一 block 的共享内存（DSM）。官方称可降低跨 SM 传输开销最多 7×；还能在 cluster 内分片数据缓解每 block 共享内存压力。
编程接口 `cluster.map_shared_rank(SMEM, DST_BLOCK_RANK)`（编译成 PTX `mapa`）。
三个基准：
1. **延迟**：两 block 各 1 线程，测跨 SM 加寄存器延迟；用 `mov.u32 %0, %%smid` 确保跑在不同 SM。
2. **RBC 环形拷贝**：每 SM 一 block 聚成 cluster，按 rank R 把值加到 `(R+1)%CS`，用 ILP 拉满带宽。
3. **直方图**：用 DSM 把 bin 分散到同 cluster 不同 block，做 atomic 自增。

---

## IV 实验结果（Experimental Results）
### IV-A 实验配置（表 III）
**中文翻译**
三款卡：A100 PCIe（Ampere）、RTX4090（Ada）、H800 PCIe（Hopper）。
| 设备 | A100 | RTX4090 | H800 |
|---|---|---|---|
| SM×核心/SM | 108×64 | 128×128 | 114×128 |
| 显存 | 40GB HBM2e | 24GB GDDR6X | 80GB HBM2e |
| 显存带宽 | 1555 GB/s | 1008 GB/s | **2039 GB/s** |
| 张量核 | 432(三代) | 512(四代) | 456(四代) |
| DPX 硬件 | **无** | **无** | **有** |

### IV-B 内存延迟与吞吐
**中文翻译**
- **延迟**：三代各级延迟接近；HBM2e 的 A100/H800 全局内存延迟略低于 4090。L2 平均延迟是 L1 的 6.5 倍，全局内存是 L2 的 1.9 倍。
- **吞吐**：向量化 `float4` 总是更好。L2 吞吐 H800 是 4090 的 2.6 倍、A100 的 2.2 倍。实测达理论值 92%/90%/91%；L2/全局吞吐比 4.67/2.01/4.23。
### IV-C 张量核结果

**`mma` 结果（表 VII）**
**中文翻译**
- A100/H800 上较大形状 `mma` 吞吐更高；4090 上该规律消失。
- 稀疏 `mma`：4090 上可达 dense 两倍；A100 仅大形状达理论加速；**H800 上稀疏平均只快 1.42 倍**（未充分利用稀疏张量核）。
- **A100 的 `mma` 超理论峰值 95%；4090 略超；但 H800 上 `mma` 仅达平均 62.9%**。开发 Hopper 高性能应用应谨慎使用 `mma`。

**`wgmma` 结果（表 VIII/IX）**
**中文翻译**
`wgmma` 是 Hopper 专属、warp-group 级、**首个异步**张量核指令。
- 矩阵初始化为 0 时**吞吐超理论峰值 95%**；随机初始化时下降（功耗逼近 H800-PCIe **350W 上限**频率被拉低）。**提醒：用张量核要考虑功耗墙**。
- 稠密 `wgmma`（N=128）各类型延迟均 128.0；RS/SS 接近。
- 稀疏 `wgmma`：RS 延迟 128、SS 144；**SS 吞吐低于 RS**（SS 从共享内存取 m×k 并做 2:4 剪枝，访问量翻倍藏不住延迟）。
**不同 N 值的 `wgmma`（表 X）**
**中文翻译**
以 `wgmma.m64nNk16.f32.f16.f16` 为例：
- **N ≥ 64 时所有 `wgmma` 都接近峰值**；
- N < 64 时吞吐下降，SS 延迟高于 RS、吞吐低于 RS（计算密度低，藏不住共享内存延迟）。
- **建议：用 `wgmma` 尽量取 N ≥ 64**。
（FP8/INT8 理论峰值 1513 TOPS；零矩阵实测 1448 TOPS。）

**能效（表 XI）**
**中文翻译**
用 `mma` 最大形状测能耗：稠密 H800 能效是 A100 的 **1.60 倍**、4090 的 **1.69 倍**；稀疏 **1.33 / 1.39 倍**——H800 能效显著更高。
### IV-D Transformer Engine / LLM 推理
**中文翻译**
- `te.Linear` 矩阵乘：较大矩阵（N=16384）FP8 接近 FP16 两倍；小矩阵受量化开销拖累。
- `te.TransformerLayer`：计算密度增大后 H800 优势明显，FP16 近 FP32 两倍；**FP8 在 hidden_size>4096 时优于 FP16 但未达两倍**（部分模块仍非 FP8）。
- **LLM 推理（表 XII）**：decode-only 推理是**内存受限**，FP8 优势不明显；且 TE 支持不全、模块间仍 FP16/FP32 传输未融合。
| GPU | 模型 | FP32 | BF16 | FP8 |
|---|---|---|---|---|
| 4090 | llama-3B | 414.08 | 425.19 | 429.31 |
| H800 | llama-2-7B | 568.91 | 502.65 | 474.42 |
（Llama 上 FP8 未占优，甚至略低于 BF16。）
### IV-E Hopper 新特性

**DPX（图 6/7）**
**中文翻译**
- 4090/A100 的 DPX 是软件模拟性能相近；**H800 在 relu 类指令更快，16-bit 操作加速最高 13 倍**。
- 但**并非所有函数都加速**：如 `__viaddmax_s32`（返回 max(s1+s2,s3)）三卡接近——Hopper 用新 `VIMNMX` 但相比老 `IMNMX` 提升不大。
- **定位硬件**：block 数<SM 数时吞吐正比 block 数；刚超 SM 整数倍时骤降再回升；**最大吞吐在 block 数为 SM 整数倍 → DPX 加速单元位于 SM 级**。

**异步数据搬运（表 XIII/XIV）**
**中文翻译**
- 小 block（8×8）`AsyncPipe` 更优：H800 **+39.5%**、A100 **+19.6%**（小 block warp 数不足藏不住同步拷贝延迟）。
- block 增大到 32×32 优势消失；H800 上反而 **-1.8%**（高 warp 并发已隐藏延迟，异步反成开销）。
**分布式共享内存（DSM）**
**中文翻译**
- SM-to-SM 网络延迟 **180 周期，比 L2 低 32%**。
- 吞吐：cluster size=2 峰值 **3.27 TB/s**，size=4 降至 **2.65 TB/s**（更多 block 争带宽）。需权衡 block/cluster 大小。
- 直方图：最优 cluster 大小随 block 变化；Nbins 增到 2048（CS=1）性能骤降，用 cluster 分片 Nbins 可缓解；合适 cluster 大小还能借 SM-to-SM 网络减少片上共享内存流量。

---

## V 结论（Conclusion）
**中文翻译**
本文用指令级基准研究三代 GPU 的内存层级与张量核性能。Hopper 在内存带宽与张量核上符合官方宣称。**关键：必须用最新 `wgmma` 指令才能发挥第四代张量核全部性能**。库级/应用级分析显示**运算规模大时低精度优势明显**。我们探索了 Hopper 三大特性：DPX、异步数据搬运、分布式共享内存，增进了对新架构的理解，助力算法优化。

> 说明：本翻译基于 ar5iv（2024-03-05）正文提取，冗长数据表保留关键结论。原文以 arXiv:2402.13499 为准。
