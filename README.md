# learn-cuda-cute-triton

练 GPU kernel：裸 CUDA → PTX → CuTeDSL / Triton → Hopper FA3 → 微架构实测。

原生 CUDA 参考 [leet-cuda](https://github.com/xlite-dev/leetcuda)。环境：`source setup_env.sh`。博客：[reading/blogs.md](reading/blogs.md)。

| 目录 | 干什么 |
|:---|:---|
| [`origin_cuda_kernel/`](#origin_cuda_kernel) | 手写 CUDA 算子 |
| [`ptx_asm/`](#ptx_asm) | 内联 PTX，拼到 HGEMM |
| [`cutedsl_tutorial/`](#cutedsl_tutorial) | CuTeDSL 入门教程 |
| [`cutedsl_ref/`](#cutedsl_ref) | 用 CuTe 对照重写裸 CUDA 算子 |
| [`triton/`](#triton) | 同一批算子的 Triton 版 |
| [`hopper_fa3/`](#hopper_fa3) | Hopper 上写 FA3，和官方比 |
| [`micro_bench/`](#micro_bench) | H20 微架构实测数字 |
| [`profile/`](#profile) | nsys / ncu 看慢在哪 |
| [`reading/`](#reading) | 博客清单 |

---

## origin_cuda_kernel

裸 CUDA 写算子。单文件、以 fp32 / fp16 为主，向量化用 `float4` / `half2` / 128-bit pack。

- `add/`：逐元素加，1 → 4 → 8 元素/线程
- `embedding/` `transpose/` `reduce_max/` `softmax/` `layer_norm/` `sgemv/`：查表、转置、归约、softmax、LN、GEMV
- `sgemm/`：naive → tiling / double buffer / `cp.async` → WMMA / TMA + warp spec
- `other/`：scan、topk、sort、conv、pooling、loss、量化、图、FA、fused add+rmsnorm 等
- `plan.md`：Tensara 题目覆盖

练的是：合并访存、shared memory、warp shuffle、流水线。

---

## ptx_asm

不用 CUTLASS，内联 PTX 控指令。按编号看：

1. `01_basics` — `asm volatile`、访存、barrier、atomic、shuffle
2. `02_cp_async` — GMEM → SMEM 异步拷、double buffer
3. `03_ldmatrix` — SMEM 按 TC layout 装寄存器
4. `04_mma` — `mma.sync` m16n8k16
5. `05_hgemm_mma` — 上面串成带流水的 HGEMM
6. `06_hopper_wgmma_tma` — wgmma / TMA / mbarrier / cluster
7. `07` `08` — 类型转换、cache / prefetch

后面还有 TMA 争用、syncwarp、SFU 等小实验。

---

## cutedsl_tutorial

可跑的 Python 教程，从 hello 到 Flash Attention。

- 01–02：kernel / Layout（shape + stride）
- 03–04：tiling、TiledMMA、smem、异步流水
- 05–07：WMMA；Hopper TMA+WGMMA；Blackwell tcgen05（后两篇要对应卡）
- 08+：SDPA / FA V1 V2、TiledCopy、swizzle、persistent kernel

---

## cutedsl_ref

用 CuTeDSL 重写 `origin_cuda_kernel/` 里的 add / embedding / transpose / reduce / GEMM / LN。对照看 Layout、copy atom、MMA 数据流。

- `sgemm_explained.md`：GMEM → SMEM → 寄存器 → TC → 写回（CuTe vs 裸 CUDA）
- `CUTE_API_CHEATSHEET.md`：API 速查

---

## triton

同一批算子：add、reduce、softmax、LN、GEMM，以及 `flash_attention/` 的 V1 / V2。看 Triton 帮你藏了什么、还得自己管什么。

---

## hopper_fa3

写 FA3 的最小实验台。只改 `csrc/my_fa3_kernel.cu`，`run.py` 和官方 FA3（或 SDPA）比正确性和速度。

- `01_hopper_features.cu`：cluster、WGMMA、DSMEM、warp spec、TMA multicast
- 改完：`python3 run.py`；自己的 attention 写完加 `--real`

---

## micro_bench

H20（sm_90）真机数字，不是文档抄的。每个 `m_*.cu` 测一件事：访存、L2、延迟、occupancy / ILP、smem、swizzle、cp.async、TC、同步、launch、NVLink。

结论在本目录 `README.md`。`nvlink/` 另有 30+ 个实验（粒度、条带、原子、fence、组播）。

---

## profile

nsys / ncu。

- `ncu_sections_cheatsheet.md`：SOL 看 bound，WarpState 看 stall
- `profile_learning.md`：以 FA3 为主线的 profile 计划
- `study/`：stall、roofline、TMA 等小 demo
- `gemm/`：一次 TMA pipeline 调参记录

---

## reading

博客在 [reading/blogs.md](reading/blogs.md)。
