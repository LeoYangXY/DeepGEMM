# Warp Stall Reason 实战对照手册（Blackwell sm_120 / RTX 5050 Laptop）

> 目的：看到一种 CUDA 写法，能立刻直觉判断 "这个 kernel 大概率会卡在哪个 stall"。
> 方法：每个 stall reason 都配一个**真正编译跑过 ncu** 的 demo，贴实测 stall 值。
> 编译：`nvcc -O3 -lineinfo -arch=sm_120 -o stall_demo stall_demo.cu`
> 采集：`ncu --section WarpStateStats ...`，看 `smsp__average_warps_issue_stalled_<reason>`
>
> ⚠️ **本机是 Blackwell（sm_120）**。它的 NCU 把大量 "等依赖 / 等内存 / 等单元" 归并到
> `wait`，所以**部分教科书 stall 在本机用纯 CUDA C++ 跑不出来（恒为 0）**。下面用
> ✅ / ⚠️ / ❌ 标了"本机实测是否真能跑出来"，绝不编造。

---

## 速查表（直觉养成）

| 你写了这样的代码…                          | 本机实测主要卡在        | 标记 |
|-------------------------------------------|------------------------|------|
| 串行依赖链（`a = f(a)` 每步依赖上一步）    | `wait`（Blackwell 把它归这里，而非 short_scoreboard） | ⚠️ |
| 随机 / 指针追逐的 global load              | `lg_throttle` + `drain` | ✅ |
| 高 ILP、8 条独立计算链                     | 几乎不 stall（`no_instruction` 略高，对照） | ✅ |
| `__syncthreads()` 频繁同步                 | `barrier`              | ✅ |
| 密集 FP32 FMA（想压 math pipe）            | 不生效，`math_pipe_throttle` 恒 0（FP32 太强） | ❌ |
| `__threadfence_block()` 频繁屏障           | `membar`（量小）+ `drain` | ✅ |
| `__nanosleep` / PTX `nanosleep` 主动睡     | `sleeping`             | ✅ |
| 数据相关 `if/else`（条件依赖上条结果）      | 被编译器优化成 select → 无 `branch_resolving`，归 `wait` | ❌ |
| 密集 shared memory 跨 bank 访问            | 归 `wait`，`mio_throttle` 恒 0 | ❌ |
| `cuda::barrier::arrive_and_wait()`         | `wait`（但被 mem/load 盖过） | ✅ |
| 密集 `sinf/rcp/cos` SFU                     | 不生效，`dispatch_stall` 恒 0 | ❌ |
| 单线程 block / 低占用                      | `no_instruction`       | ✅ |
| 尾部一次性发起大量 global load 后退出       | `drain`                | ✅ |
| 确定性依赖 load（load 地址来自上条 load）   | `lg_throttle` + `drain`（`long_scoreboard` 恒 0） | ❌ |

---

## Part A：本机（sm_120）实测能跑出来的 stall reason

### ✅ lg_throttle —— Load/Store 单元节流（最典型的"内存延迟"stall）
**含义**：warp 想发 load/store，但 LG（Load/Store 单元）队列满了，要等。
**直觉信号**：大量**随机 / 不可合并**的 global 内存访问。这是 pointer chasing 的招牌。
**实测**（本机 sm_120）：`kernel_lg_throttle` → `lg_throttle=149`、`drain=123`

```cuda
// 指针追逐：每次 load 的地址依赖上一次 load 的结果 -> 长延迟 global load 占满 LG
__global__ void kernel_lg_throttle(float *data, int *indices, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float sum = 0.0f;
    int ptr = idx;
    for (int i = 0; i < 80; i++) {
        ptr = indices[ptr % N];   // load -> 得到下一次地址（依赖上一条）
        sum += data[ptr % N];      // load -> 依赖上面结果
    }
    out[idx] = sum;
}
```

### ✅ drain —— warp 收尾等待 retire
**含义**：warp 即将退出，但还有已发射的长延迟指令没完成（retire），必须等。
**直觉信号**：kernel 尾部突然发起一大波 global load 然后函数就结束了。
**实测**：`kernel_drain` → `drain=16.83`；`kernel_lg_throttle` → `drain=123`；`kernel_membar` → `drain=64`

```cuda
// 主体算完，尾部一次性发 96 个分散 global load，然后立刻退出 ->
// warp 必须等这些长延迟 load retire 才能结束 -> drain
__global__ void kernel_drain(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    float acc = 0.0f;
    for (int i = 0; i < 20; i++) { v=v*1.01f+0.1f; acc += v; }
    float sum = 0.0f;
    for (int i = 0; i < 96; i++) {
        sum += data[(idx*31 + i*131) % N];
    }
    out[idx] = acc + sum;
}
```

### ✅ barrier —— 块内同步等待
**含义**：warp 在 `__syncthreads()` 处等同一 block 里其他 warp 到齐。
**直觉信号**：kernel 里有 `for { ...; __syncthreads(); }` 这种每轮都同步的写法。
**实测**：`kernel_barrier` → `barrier=18.45`、`selected=9.11`

```cuda
__global__ void kernel_barrier(float *data, float *out, int N) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx < N) smem[tid] = data[idx];
    __syncthreads();
    for (int i = 0; i < 60; i++) {
        if (idx < N) smem[tid] = smem[tid]*0.99f + smem[(tid+1)%256]*0.01f;
        __syncthreads();   // 每轮必须等所有 warp 到齐
    }
    if (idx < N) out[idx] = smem[tid];
}
```

### ✅ membar —— 内存屏障等待
**含义**：warp 在 `__threadfence_block()` 处等内存屏障完成。
**直觉信号**：循环里反复调用 `__threadfence_block()`。注意本机它伴随 `drain` 一起出现。
**实测**：`kernel_membar` → `membar=1.87`（非0）、`drain=64.04`

```cuda
__global__ void kernel_membar(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    for (int i = 0; i < 120; i++) {
        v = v*1.01f + 0.1f;
        __threadfence_block();   // 块内内存屏障
    }
    out[idx] = v;
}
```

### ✅ sleeping —— warp 主动睡眠
**含义**：warp 主动调用 nanosleep，硬件让它睡一段时间。
**直觉信号**：device 侧 `__nanosleep` 或 PTX `nanosleep`。
**重要坑**：`-O3` 下普通 `__nanosleep(1000)` 会被编译器整个删掉（实测 sleeping=0）。
必须用 **PTX `nanosleep.u32`** 才能强制保留。
**实测**：`kernel_sleeping` → `sleeping=387.53`（PTX 版）

```cuda
// 用 PTX nanosleep 指令强制触发 sleeping（纯 __nanosleep 在 -O3 会被优化掉）
__global__ void kernel_sleeping(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx];
    unsigned int acc = 0;
    for (int i = 0; i < 60; i++) {
        v = v*1.01f + 0.1f;
        unsigned int ns = 1000u + (unsigned int)(v*7.0f);
        asm volatile("nanosleep.u32 %0;" : "+r"(ns) : : "memory");  // 强制睡眠
        acc += ns;
    }
    out[idx] = v + (float)acc;
}
```

### ✅ no_instruction —— 没有指令可发
**含义**：SM 上 active warp 太少，调度器找不到可发的 warp，空闲。
**直觉信号**：极度低占用——比如每个 block 只有 1 个线程真正干活，其余 lane 立即 return。
**实测**：`kernel_no_instruction` → `no_instruction=12.53`、`selected=4.13`

```cuda
// 只有 lane 0 干活，其余 31 个 lane 立即退出 -> block 内 warp 几乎空转
__global__ void kernel_no_instruction(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x != 0) return;   // 仅 lane0 工作，其余立即退出
    if (idx >= N) return;
    float v = data[idx];
    float acc = 0.0f;
    for (int i = 0; i < 80; i++) {
        acc += v*1.01f;
        v = v + 0.1f;
    }
    out[idx] = acc;
}
```

### ✅ wait —— 等待（Blackwell 的"大杂烩"）
**含义**：warp 在等某个 scoreboard / 屏障 / 依赖，本机会把所有"等"都归到 wait。
**直觉信号**：几乎所有有依赖/同步的代码都会出现。`cuda::barrier::arrive_and_wait()` 是教科书式来源。
**实测**：`kernel_wait` → 含 `wait`（但本机被 `lg_throttle=22`、`membar=13` 盖过，故 wait 数值不高）

```cuda
__global__ void kernel_wait(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) init(&bar, blockDim.x);
    __syncthreads();
    float v = data[idx];
    float acc = 0.0f;
    for (int i = 0; i < 100; i++) {
        v = v*1.01f + 0.1f;
        bar.arrive_and_wait();   // 到达并等待屏障 -> wait
        acc += v;
    }
    out[idx] = acc;
}
```

### ⚠️ not_selected —— 就绪但没被选中（自然状态）
**含义**：warp 可以发指令，但同一周期调度器选了别的 warp。这是**健康状态**，不是 bug。
**直觉信号**：occupancy 充足（很多就绪 warp 抢发射槽）时自然出现。对照 kernel `kernel_selected` 也会带出。

```cuda
// 高 ILP 对照：8 条独立链，warp 很少 stall（no_instruction 略高，其余极低）
__global__ void kernel_selected(float *a, float *b, float *c, float *d,
                                float *e, float *f, float *g, float *h,
                                float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v0=a[idx], v1=b[idx], v2=c[idx], v3=d[idx];
    float v4=e[idx], v5=f[idx], v6=g[idx], v7=h[idx];
    for (int i = 0; i < 100; i++) {
        v0=v0*1.01f+0.1f; v1=v1*1.02f+0.1f; v2=v2*1.03f+0.1f; v3=v3*1.04f+0.1f;
        v4=v4*1.05f+0.1f; v5=v5*1.06f+0.1f; v6=v6*1.07f+0.1f; v7=v7*1.08f+0.1f;
    }
    out[idx]=v0+v1+v2+v3+v4+v5+v6+v7;
}
```

---

## Part B：本机（sm_120）实测跑不出来的 stall reason —— 诚实说明

下面这些 stall 在 **Blackwell sm_120 的 NCU 上恒为 0**（我试过朴素写法、`volatile`、
`#pragma unroll(1)`、`asm volatile`、PTX 等手段）。不是 demo 没写对，而是 **Blackwell 的
stall 归因逻辑把它们合并到了 `wait` / `lg_throttle` / `drain`**。你在其他架构（Ampere/Hopper）
上可能能看到，但在本机看不到。列出来避免你误以为"我写了 X 就该有 Y stall"。

| stall reason | 为什么本机跑不出 | 在其他架构/手段下如何触发 |
|--------------|-----------------|--------------------------|
| **short_scoreboard** | Blackwell 把"等短延迟算术依赖"归到 `wait`（kernel_short_scoreboard 实测 wait=0.94、short_scoreboard=0） | Ampere/Hopper 上串行 `a=f(a)` 依赖链会显式出现 |
| **long_scoreboard** | 本机随机 global load 归 `lg_throttle`+`drain`，不单独计 long_scoreboard | 旧架构单独计；本机用 `lg_throttle` 即当"内存延迟 stall"读 |
| **branch_resolving** | 数据相关分支被编译器优化成 predicated `select`，无真实分支解析停顿 | 用函数指针/间接跳转强制真实分支（GPU 上难构造） |
| **math_pipe_throttle** | Blackwell FP32 吞吐极高，简单 FMA 洪流打不满 math pipe（恒 0） | 极重度 FMA + Tensor Core 争用时可能偶发 |
| **mio_throttle** | shared memory 访问在本机归 `wait`，不单独计 mio_throttle | Ampere 上密集 bank conflict 会显式出现 |
| **dispatch_stall** | Blackwell 发射端口宽、SFU 充足，密集 SFU 仍归 `wait`（恒 0） | 特殊单元极度争用时偶发 |
| **tex_throttle** | 未使用 texture 单元，且本机即使访问也归其他类别 | 用 `tex1d/tex2d` 采样且纹理单元饱和时 |

对应 demo 代码（写了，但本机 ncu 确认这些 reason 恒为 0）：

```cuda
// short_scoreboard：串行算术依赖链（本机 -> wait，非 short_scoreboard）
__global__ void kernel_short_scoreboard(float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = (float)idx;
    for (int i = 0; i < 200; i++) {
        val = val * 1.0001f + 0.0007f;
        val = val * val + 0.5f;
        val = sqrtf(val + 1.0f);
        out[idx] = val;
    }
    out[idx] = val;
}

// branch_resolving：数据相关分支（本机 -> 被优化成 select，归 wait）
__global__ void kernel_branch(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx]; float acc = 0.0f;
    for (int i = 0; i < 200; i++) {
        v = v*1.01f + 0.1f;
        if (v > 1.0f) acc += v; else acc -= v*0.5f;       // 条件依赖 v
        float t = v + acc*0.001f;
        if (t < 0.0f) acc = 0.0f; else acc += t;          // 再依赖一次
        out[idx] = acc;
    }
    out[idx] = acc;
}

// mio_throttle：密集 shared memory 跨 bank（本机 -> wait）
__global__ void kernel_mio_throttle(float *data, float *out, int N) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx < N) smem[tid] = data[idx];
    __syncthreads();
    float acc = 0.0f;
    for (int i = 0; i < 200; i++) {
        acc += smem[(tid*7+1)%256];
        acc += smem[(tid*13+3)%256];
        acc += smem[(tid*5+9)%256];
        acc += smem[(tid*11+17)%256];
        acc += smem[(tid*17+5)%256];
    }
    if (idx < N) out[idx] = acc;
}

// math_pipe_throttle：密集 FMA（本机 -> 不生效，恒 0）
__global__ void kernel_math_pipe_throttle(float *a, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = a[idx];
    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    for (int i = 0; i < 300; i++) {
        s0=s0+x*1.00001f; s1=s1+x*1.00002f; s2=s2+x*1.00003f; s3=s3+x*1.00004f;
        s4=s4+x*1.00005f; s5=s5+x*1.00006f; s6=s6+x*1.00007f; s7=s7+x*1.00008f;
        s0=s0*x*2.0f; s1=s1*x*2.0f; s2=s2+x*2.0f; s3=s3+x*2.0f;
        s4=s4+x*2.0f; s5=s5+x*2.0f; s6=s6+x*2.0f; s7=s7+x*2.0f;
    }
    out[idx]=s0+s1+s2+s3+s4+s5+s6+s7;
}

// dispatch_stall：密集 SFU（本机 -> 不生效，恒 0）
__global__ void kernel_dispatch_stall(float *data, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = data[idx]; float acc = 0.0f;
    for (int i = 0; i < 200; i++) {
        float s = sinf(v + (float)i);
        float r = __frcp_rn(v + s + 1.0f);
        float c = cosf(v - r);
        float t = __fsqrt_rn(c*c + 1.0f);
        acc += s + r + c + t;
        v = v*1.001f + 0.1f;
        out[idx] = acc;
    }
    out[idx] = acc;
}

// long_scoreboard：确定性依赖 load（本机 -> lg_throttle + drain）
__global__ void kernel_long_scoreboard(float *data, int *nxt, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float sum = 0.0f; int p = idx;
    for (int i = 0; i < 80; i++) {
        float x = data[p];
        p = nxt[p % N];
        sum += x + (float)p;
    }
    out[idx] = sum;
}
```

---

## 结论 & 直觉养成要点

1. **内存类问题**（随机访问 / pointer chasing / 尾部密集 load）→ 看 `lg_throttle` + `drain`。
   这是 GPU 上最常见的性能杀手，demo 一跑就中。
2. **同步类问题**（`__syncthreads` / `__threadfence_block` / `cuda::barrier`）→
   看 `barrier` / `membar` / `wait`。
3. **调度类问题**（低占用 / 单线程）→ 看 `no_instruction`。
4. **主动睡眠**（`nanosleep`）→ 看 `sleeping`，但**必须用 PTX 版**，否则 -O3 直接删掉。
5. **Blackwell 上 `wait` 是"兜底大类"**：很多教科书里的 short_scoreboard /
   long_scoreboard / branch_resolving 在本机都被归到 `wait`。所以看到 `wait` 高别慌，
   要回到代码看它到底在等"算术依赖"还是"内存"还是"屏障"。
6. **`math_pipe_throttle` / `mio_throttle` / `dispatch_stall` 在本机很难复现**：
   因为 Blackwell FP32 / shared mem / 发射端口都太强，普通 micro-benchmark 压不垮。
   真要压，得上 Tensor Core 争用或大矩阵。
