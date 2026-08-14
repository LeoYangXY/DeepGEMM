# SM120 FP8：对着 SASS 讲手排指令

> 机器：RTX 5050 Laptop，`sm_120a`，20 SM，boost 1905 MHz。
> Kernel：`sm120_fp8_gemm_1d1d`（裸 FP8，无 1D1D scale TMA / FFMA），tile `128×80×128`，shape `4096³`。
> Math：PTX `mma.sync ... kind::mxf8f6f4.block_scale ... e4m3.e4m3.f32.ue8m0`
> （identity `ue8m0=127`），SASS **`QMMA.SF.16832.F32.E4M3.E4M3.E8`**。
>
> 内层循环原文：`docs/sm120_sass/qmma_sf_inner_loop.sass`（`cuobjdump --dump-sass`，没改指令）。
> 指令数字：`tests/sm120_inst_microbench.cu`（必须 `-gencode arch=compute_120a,code=sm_120a`，
> `--gpu-architecture=sm_120a` 会把 `.target` 降成 `sm_120`，ptxas 拒收 `block_scale`）。

这份笔记按面试能讲的顺序写：**手排要解决什么 → 控制字怎么读 → 先测指令再排 → 对着 SASS 验证 → 源码改了 SASS 未必改**。

---

## 0. 手排在干什么（30 秒开场）

不是改 MMA 的数学，也不是手写整份 SASS。目标只有一句：

**让 tensor pipe 上的 `QMMA.SF` 按硬件吞吐连发，ALU / LSU 不要拆开这条连发，也不要复用 QMMA 刚写完的寄存器。**

一条 `QMMA.SF.16832` = `16×8×32×2` = **8192 FLOP**。这条 kernel 一个 K-block 发 80 条（`2 stage × 4 k-step × 10 个 n8 tile`）。谁夹在两条 QMMA 中间，谁就决定 tensor pipe 空不空。

源码里对应的调度在 `mma_kblock_ldsm`（`deep_gemm/include/deep_gemm/mma/sm120.cuh`）：

```
prologue:  LDSM A[k=0], LDSM B[n=0]
for k:
    LDSM A[k+1]                 // 藏在整段 N 扫的 QMMA 影子里
    for n2:
        LDSM B[n2+1]            // 藏在当前 2 条 QMMA 影子里
        QMMA n2*2
        QMMA n2*2+1
```

LDSM 走 **LSU**，QMMA 走 **Tensor**，可以双发。这是已经做对的部分。下面用 dump 证明它。

---

## 1. 怎么读这份 SASS

`cuobjdump --dump-sass` 每条指令两行：

```
/*PC  */  OPCODE dst, src ;     /* 指令编码 */
                                /* 控制字   */
```

手排真正看的是**控制字高 32 bit** 和 **QMMA 的寄存器有没有被下一条 ALU 立刻复用**。本 kernel 里反复出现的字：

| 控制字（高 32） | 本文件里的完整字 | 含义（对着 stall 对齐，sm_120 实测） |
|---|---|---|
| `000ff600` | `0x000ff60000003e28` | stall ≈ 6，**不等 scoreboard**。独立 QMMA.SF 连发。 |
| `002ff600` | `0x002ff60000003e28` | **等 barrier 1** + stall 6。第一条 QMMA 在等前面 LDSM 写回 RF。 |
| `000fe200` | `0x000e740000000200` | stall ≈ 2。`LDSM.x4` 的典型字：发出去就能跟下一条。 |
| `000fca00` | （本窗口没有） | **长 stall**。常见原因：ALU 的目的寄存器落在刚写完的 QMMA dest 组里（一组 4 个 FP32）。 |
| `000fe800` | （本窗口没有） | 以前 1D1D 版本里是 scale 的 `LDS`。裸 FP8 之后这条税没了。 |

`QMMA.SF.16832.F32.E4M3.E4M3.E8  Rd, Ra, Rb, Rc, Rsfa, Rsfb, URZ`：

- `Rd` 起始寄存器，占 **R[d .. d+3]**（4 个 FP32 累加器）。面试时这点最容易漏：复用的不一定是 `Rd` 自己，**`Rd+1/+2/+3` 也算刚写完**。
- `Ra` 占 4 个寄存器（A fragment），`Rb` 占 2 个（一个 n8 的 B）。
- `Rc`：本 kernel 累加器用 `D = A*B + C`，所以 `Rc` 经常等于 `Rd`；`RZ` 表示从 0 起。
- `Rsfa` / `Rsfb`：identity `ue8m0=127`（dump 里的 `R93`，`MOV R93, 0x7f` 在 burst 之前就算完）。

---

## 2. 先测指令，再决定排什么

`tests/sm120_inst_microbench.cu`，1 warp，延迟用依赖链，吞吐用 8 路独立累加器：

```
device: NVIDIA GeForce RTX 5050 Laptop GPU  sm_120  SMs=20  clock=1905000 kHz

op                     cyc/op
QMMA latency            34.43     unscaled, dependent D
QMMA tput               32.04     unscaled, 8 accums
QMMA.SF latency         29.00     ue8m0=127, dependent D
QMMA.SF tput            16.10     ue8m0=127, 8 accums
FFMA latency             4.06
FFMA tput                1.07
IMAD.LO latency          4.06
```

从这张表推出三条调度约束（面试就讲这三条）：

1. **Unscaled `QMMA.16832` 单 warp 吞吐 ≈ 延迟（32 vs 34）→ 几乎不流水。** `QMMA.SF` 吞吐掉到 16 cyc/op，正好 **2×**，这是上一笔 commit 换成全速 opcode 的数字证据。手排解决不了 opcode 选错。
2. **同一累加器的下一条 QMMA 要等 ~29 cycle。** 不同 n-tile 的累加器独立，控制字才会是 `000ff600`。前提是 `Ra`/`Rb` **已经在 RF 里**。
3. **`IMAD` / `LOP3` / `MOV` 走 ALU，延迟 4 cycle，但占 issue slot。** 插进 QMMA 连发里，会把 `000ff600` 拆开；如果目的寄存器还落在 QMMA dest 组上，控制字变成 `000fca00`，tensor pipe 空转的时间远大于 4 cycle。
4. **`LDSM.x4` 和 QMMA 不同管线。** SASS 上 stall 2。正确位置：跟在 QMMA 后面，用 tensor 的 16–29 cycle 窗口把下一片 fragment 搬进 RF。

所以手排的优先级是：

| 优先级 | 动作 | 本 kernel |
|---|---|---|
| 已经做了 | 选全速 opcode | commit「QMMA.SF」 |
| 已经做了 | 去掉 1D1D scale 的 TMA / FFMA | 和 cuBLASLt 同一道裸 FP8 |
| 已经做了 | LDSM 藏进 QMMA 影子（双管线） | `mma_kblock_ldsm` |
| 已经做了 | 让 ptxas 把地址留在 RF 里 | 本 dump：A/B 都是 `[R+imm]`，burst 里没有 XOR ALU |
| 做不到 | 单 warp 把 29 cycle 延迟藏完 | 要更多 warp / 2 CTA，occupancy 不够 |

---

## 3. 已经排对的：LDSM 在 QMMA 影子里

B 在 **固定 K、N 每次 +16 行** 时，`row % 8` 不变，128B swizzle 的 XOR 项不变，物理地址步长是 `16 × 128 = 2048 = 0x800`。ptxas 因此把 B 收成 **一个 per-lane 基址 + immediate**：

```
/*1160*/  LDSM.16.M88.4 R20, [R89+0x12000] ;   /* stall 2 */
/*1180*/  LDSM.16.M88.4 R16, [R89+0x12800] ;   /* +0x800 */
/*11a0*/  LDSM.16.M88.4 R60, [R89+0x13000] ;
```

数学 burst 里同样的步长夹在 QMMA 中间：

```
/*1200*/  QMMA.SF ... R44, R24, R22, R44, R93, R93, URZ ;  /* 0x000fe200 */
/*1210*/  LDSM.16.M88.4 R20, [R87+0x12000] ;               /* stall 2，下一片 B */
/*1220*/  QMMA.SF ... R48, R24, R16, R48, ... ;            /* 0x000ff600 连发 */
/*1230*/  QMMA.SF ... R52, R24, R18, R52, ... ;
```

`0x1210` 这条就在 math burst 里，控制字仍是 LDSM 的 stall 2，**没有**变成 `000fca00`。这就是源码里

```c++
if (n_pair + 1u < kNPairs)
    load_b_fragment_ldsm_x2n(b_frag[b_stage ^ 1u], ...);
mma_m16n8k32_f32_e4m3_e4m3(...);   // 当前 n
mma_m16n8k32_f32_e4m3_e4m3(...);
```

在 SASS 上的样子：下一片 B 的 LDSM 和当前 QMMA **双发**，Ra/Rb 到下两条 QMMA 时已经在 RF。

A 以前不行。K 每次 +32 字节，XOR `(col/16) ^ (row%8)` 不是常数，现场算就会冒出 `MOV Rx, 0x10` + `LOP3`。**这份 dump 里 A 也折成基址了：**

```
/*1130*/  LDSM.16.M88.4 R24, [R105+UR11+0xa000] ;
/*11d0*/  LDSM.16.M88.4 R32, [R90] ;
/*12d0*/  LDSM.16.M88.4 R24, [R85+0x12000] ;
```

`0xa000` 仍是 A tile 在 smem 里的 slab 偏移，但 XOR 已经折进 `R105` / `R90` / `R85`，burst 里不再算一遍。

---

## 4. 对着 SASS 验证：这条连发是干净的

窗口 `0x11f0`–`0x12c0`（完整原文在 `docs/sm120_sass/qmma_sf_inner_loop.sass`）：

```
/*1140*/  MOV R93, 0x7f ;                 // identity ue8m0=127，burst 前就算完

/*11f0*/  QMMA.SF ... R40, R24, R20, R40, R93, R93, URZ ;
          /* 0x002ff600  等 LDSM，第一条 math */

/*1200*/  QMMA.SF ... R44, R24, R22, R44, ... ;   /* Rd=R44 → R44–R47 */
/*1210*/  LDSM.16.M88.4 R20, [R87+0x12000] ;
/*1220*/  QMMA.SF ... R48, ... ;                  /* 0x000ff600 */
/*1230*/  QMMA.SF ... R52, ... ;
/*1240*/  LDSM ...
/*1250*/  QMMA.SF ... R12, ... ;                  /* 0x000ff600 */
...
```

逐条在说什么：

1. **`0x11f0` 的 `002ff600`** 是正常的 prologue：第一条 QMMA 必须等 A/B fragment 进 RF。后面立刻变成 `000ff600`，说明软件流水把 Ra/Rb 备好了。
2. **`Rd=R40 / R44 / R48 / …` 各占 4 个累加器。** 不同 n-tile 互不依赖，所以能连发。
3. **`R93 = 0x7f` 不在连发里重算。** 以前 1D1D 版本还要在 burst 附近 `LDS` 两路 FP32 scale；现在 identity scale 一个寄存器就够。
4. **这个窗口没有 `000fca00`，也没有 `MOV Rx, 0x10`。** 验证方法就是对着 dump 看，不是看 C++ 写没写 hoist。

swizzle 公式仍是：

```c++
// phys = row * 128 + ((col/16) ^ (row % 8)) * 16
(row << 7) + (((col_bytes >> 4) ^ (row & 7u)) << 4) + (col_bytes & 15u)
```

手排时如果看到 `MOV Rx, 0x10` 夹在 QMMA 中间，就该想到：这是 16B atom，ptxas 在现场算 XOR。本 dump 没有这条，说明地址已经在 RF。

---

## 5. 教科书改法，以及为什么以前 SASS 不买账

按第 2 节的约束，第一反应是：**4 个 A 地址在进 math burst 之前就算完，LDSM A 变成 `[Rprecomputed]`。**

源码上就是把

```c++
load_a_fragment_ldsm(a_frag[...], a_base, k_off, lane_id);
```

换成 k-block 入口预计算 `a_addr[4]`，循环里 `ldmatrix_x4(..., a_addr[k])`。

在 **还带着 1D1D scale RF** 的版本上试过两档：C++ 外提之后，SASS 同一位置仍是 `000fca00 MOV` + `LDSM [R+0xa000]`——寄存器文件已经很满（`BLOCK_N=80` 时每 lane 40 个 FP32 累加器，加上双缓冲 fragment、两路 scale、TMA 描述符），ptxas **rematerialize** XOR，认为重算比让地址活过整个 burst 更便宜。

面试要能说出这句：

**手排看的是 dump 出来的 SASS，不是源码看起来“已经外提了”。验证方法是改完再 `cuobjdump`，看 `000fca00` 还在不在 QMMA 窗口里。**

所以那一笔 **没有把 C++ 外提写进 kernel**：改了源码、SASS 不变，就是不必要的 diff。后来把 scale TMA / FFMA 拿掉（本 kernel 的产品语义改成裸 FP8），RF 松了，ptxas 自己把地址留住了。同一份 `mma_kblock_ldsm`，SASS 可以完全不一样。

---

## 6. 手排改变不了的

4096³（同一 tile，本机这一次）：DeepGEMM **83.5 TFLOPS**，cuBLASLt **90.6 TFLOPS**（约 **0.92×**）。数值 `calc_diff` 对 cuBLASLt 是 0（同一道裸 FP8）。换成 QMMA.SF 之前，unscaled QMMA 大约是半速。大头是 opcode；去掉 1D1D 税之后指令窗口也干净了。

剩下的 gap 是 **1 CTA/SM、tile、TMA**，不是“再插两条 LDSM”。

384 thread / 1536 → occupancy 18.75%。QMMA.SF 单 warp 吞吐 16 cyc/op，藏延迟要靠 **更多 warp 交错**。高寄存器 × 384 thread 塞不进 2 个 CTA。手排指令解决不了这件事。

---

## 6b. 128×128 2-stage（D 缩小之后）

Tile 换成 `128×128×128`、TMA 32 线程（288 thread）之后，4096³ 同轮大约 **0.95–0.99× cuBLAS**。内环 dump：`docs/sm120_sass/qmma_sf_128x128_inner.sass`。

一个 K-block 64 条 QMMA.SF（`4 k-step × 16 n8`），unroll 2 stage → dump 里 128 条。空隙统计（改 C++ 3-deep B 预取 / 提前 arrive 之后 **直方图不变**）：

| 空隙 | 条数 | 含义 |
|---|---:|---|
| 相邻 QMMA | 77 | 连发 |
| 一条 LDSM | 41 | 藏在 Tensor 影子里 |
| 两条 LDSM | 5 | k-step 交界要同时备下一拍 A 和 B[0] |
| MOV/SHF / SYNCS | 2 | 下一 stage 的 mbarrier 地址和 TRYWAIT |

源码现在是 3-live/1-fill（`BLOCK_N>=48`）并把 `arrive empty` 写在最后一次 B LDSM 后面；ptxas 仍吐 77/41/5，并把 arrive 沉到和下一次 wait 挨着。寄存器 148→135，2 CTA 仍不够（288 thread 要 ≤113）。把 math 循环收成 `#pragma unroll 1` 会让 swizzle `IADD` 重新进 QMMA 窗口，更差。

`000fca00` 仍只出现在 scheduler / TMA 序言，不在 math burst。真正还能改 SASS 的锤子是整段 K-block 的 fused PTX，不是再调 C++ 顺序。

---

## 7. 两分钟讲稿（可以按这个背）

1. **目标：** 全速 `QMMA.SF` 连发，控制字 `000ff600`。一条 8192 FLOP。
2. **先 bench：** unscaled 32 cyc/op，SF 16 cyc/op（2×）；FFMA/IMAD 4 cycle；LDSM 和 QMMA 不同管线。
3. **源码调度：** 下一片 B 的 `ldmatrix.x4` 写在当前两条 QMMA 前面，SASS 上变成 `LDSM [Rbase+0x800*n]` 夹在 `QMMA.SF` 中间，stall 2，这是双管线。
4. **对着 dump 验证：** `0x11f0` 第一条 `002ff600` 等 LDSM，随后 `000ff600` 连发；A/B 都是 `[R+imm]`；identity `R93=0x7f` 在 burst 外。窗口里没有 `000fca00`。
5. **验证闭环：** 以前带着 1D1D scale 时，C++ 外提地址 SASS 仍 rematerialize。手排以 dump 为准。拿掉 scale 之后 ptxas 自己把地址留住了。
6. **边界：** 和 cuBLAS 剩下的 ~1.09× 不在指令缝里，在 occupancy / 2 CTA / TMA。数值已经对齐。

---

## 8. 文件

| 路径 | 内容 |
|---|---|
| `docs/sm120_fp8_sass_hand_schedule.md` | 本页 |
| `docs/sm120_sass/qmma_sf_inner_loop.sass` | 4096³ `128×80` 内层循环原文 |
| `docs/sm120_sass/qmma_sf_128x128_inner.sass` | 4096³ `128×128` TMA32 内层循环 |
| `tests/sm120_inst_microbench.cu` | QMMA / QMMA.SF / FFMA / IMAD |
| `deep_gemm/include/deep_gemm/mma/sm120.cuh` | `mma_kblock_ldsm` 的软件流水 |
