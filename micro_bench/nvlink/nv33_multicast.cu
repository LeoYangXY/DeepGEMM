// ============================================================================
// nv33_multicast —— NVSwitch 里的组播 / 在网归约 (NVLink SHARP)
//
// Hopper + NVSwitch 提供 multicast object: 一块虚拟地址映射到 N 张卡的显存,
// 对它做 multimem.st 时 **交换机负责复制**, 发起方只发一份数据;
// 对它做 multimem.ld_reduce 时 **交换机负责求和**, 发起方只收一份结果。
//
// 这是 NVLink 上最"看不见"的一块硬件, 官方资料几乎只有 API 没有行为描述。
// 本实验要回答:
//   1) 硬件到底支不支持、粒度要求是多少
//   2) multimem.st 广播给 N 张卡时, 发起方线上到底发了几份字节
//      (用 NVML 每卡计数器交叉验证: 发起方 Tx == 1 份, 每个接收方 Rx == 1 份
//       => 复制发生在交换机内, 这就是 in-network multicast 的直接证据)
//   3) multimem.ld_reduce 的带宽 vs 自己手动读 N 份再加
//   4) 支持哪些数据类型/宽度、对齐要求
// ============================================================================
#include "nvl_common.cuh"
#include <cuda.h>

#define DK(x)                                                                  \
  do {                                                                         \
    CUresult r_ = (x);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char* s = nullptr;                                                 \
      cuGetErrorString(r_, &s);                                                \
      fprintf(stderr, "[DRV ERR] %s:%d %s -> %s\n", __FILE__, __LINE__, #x,    \
              s ? s : "?");                                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---- 每卡 NVLink Data 计数器 (聚合) ----
static void rd_cnt(int dev, unsigned long long* tx, unsigned long long* rx) {
  *tx = *rx = 0;
  char cmd[128];
  snprintf(cmd, sizeof cmd, "nvidia-smi nvlink -gt d -i %d 2>/dev/null", dev);
  FILE* f = popen(cmd, "r");
  if (!f) return;
  char line[512];
  unsigned long long v;
  int l;
  while (fgets(line, sizeof line, f)) {
    if (sscanf(line, " Link %d: Data Tx: %llu", &l, &v) == 2) *tx += v * 1024ull;
    else if (sscanf(line, " Link %d: Data Rx: %llu", &l, &v) == 2) *rx += v * 1024ull;
  }
  pclose(f);
}

// ---------------- multimem PTX ----------------
// 组播写: 一条指令, 交换机复制到所有成员
__global__ void k_mc_st(float4* mc, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) {
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc + i),
                 "f"(1.0f), "f"(2.0f), "f"(3.0f), "f"(4.0f)
                 : "memory");
  }
}
// 在网归约读: 交换机把 N 张卡的对应位置求和后返回一份
__global__ void k_mc_ldreduce(const float4* mc, float4* sink, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0, 0, 0, 0);
  for (; i < n; i += s) {
    float4 v;
    asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(mc + i));
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  if (acc.x == 1e30f) *sink = acc;
}
// 在网归约写 (all-reduce 的 reduce-scatter 侧): 无返回值
__global__ void k_mc_red(float4* mc, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) {
    asm volatile("multimem.red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc + i),
                 "f"(1.0f), "f"(1.0f), "f"(1.0f), "f"(1.0f)
                 : "memory");
  }
}
// 对照: 手动 unicast 广播到 3 个 peer
__global__ void k_uni_bcast3(float4* a, float4* b, float4* c, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (; i < n; i += s) { st128(a + i, v); st128(b + i, v); st128(c + i, v); }
}
__global__ void k_wr1(uint4* a, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (; i < n; i += s) st128(a + i, v);
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  DK(cuInit(0));
  printf("# nv33_multicast — NVSwitch 组播 / 在网归约 (NVLink SHARP)\n");
  printf("# %s x%d\n", env.name, env.ndev);

  // -------------------------------------------------- 能力探测
  hdr("A) 硬件能力与粒度");
  int nd = env.ndev;
  for (int d = 0; d < nd; ++d) {
    CUdevice cd;
    DK(cuDeviceGet(&cd, d));
    int mc = 0, vmm = 0;
    cuDeviceGetAttribute(&mc, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cd);
    cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cd);
    printf("GPU%d  MULTICAST_SUPPORTED=%d  VMM_SUPPORTED=%d\n", d, mc, vmm);
    if (!mc) { printf("硬件不支持 multicast, 退出\n"); return 0; }
  }

  CUmulticastObjectProp mprop{};
  mprop.numDevices = nd;
  mprop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  mprop.flags = 0;
  size_t gmin = 0, grec = 0;
  mprop.size = 1 << 20;
  DK(cuMulticastGetGranularity(&gmin, &mprop, CU_MULTICAST_GRANULARITY_MINIMUM));
  DK(cuMulticastGetGranularity(&grec, &mprop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  printf("multicast granularity: minimum=%zu B (%.0f KB), recommended=%zu B (%.0f MB)\n",
         gmin, gmin / 1024.0, grec, grec / 1048576.0);

  // -------------------------------------------------- 建立 MC 对象
  const size_t SZ = 256ull << 20;
  size_t mcsz = ((SZ + grec - 1) / grec) * grec;
  mprop.size = mcsz;
  CUmemGenericAllocationHandle mch;
  DK(cuMulticastCreate(&mch, &mprop));
  for (int d = 0; d < nd; ++d) {
    CUdevice cd; DK(cuDeviceGet(&cd, d));
    DK(cuMulticastAddDevice(mch, cd));
  }
  printf("multicast object created, size=%zu MB, members=%d\n", mcsz >> 20, nd);

  // 每张卡分配物理内存并绑定到 MC 对象
  CUdeviceptr uc[8];   // 各卡的 unicast 视图 (本地)
  CUdeviceptr mcptr[8];  // 各卡上的 mc 虚拟地址
  CUmemGenericAllocationHandle ph[8];
  size_t agran = 0;
  for (int d = 0; d < nd; ++d) {
    CUmemAllocationProp ap{};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = d;
    ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    DK(cuMemGetAllocationGranularity(&agran, &ap, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    DK(cuMemCreate(&ph[d], mcsz, &ap, 0));
    DK(cuMulticastBindMem(mch, 0, ph[d], 0, mcsz, 0));
    // unicast 视图
    DK(cuMemAddressReserve(&uc[d], mcsz, agran, 0, 0));
    DK(cuMemMap(uc[d], mcsz, 0, ph[d], 0));
  }
  printf("VMM allocation granularity = %zu B (%.0f KB)\n", agran, agran / 1024.0);
  // 访问权限: 所有卡都能访问所有 unicast 视图 + mc 视图
  CUmemAccessDesc ad[8];
  for (int d = 0; d < nd; ++d) {
    ad[d].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ad[d].location.id = d;
    ad[d].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
  for (int d = 0; d < nd; ++d) DK(cuMemSetAccess(uc[d], mcsz, ad, nd));
  for (int d = 0; d < nd; ++d) {
    CK(cudaSetDevice(d));
    DK(cuMemAddressReserve(&mcptr[d], mcsz, agran, 0, 0));
    DK(cuMemMap(mcptr[d], mcsz, 0, mch, 0));
    DK(cuMemSetAccess(mcptr[d], mcsz, &ad[d], 1));
  }
  printf("bound + mapped OK\n");

  CK(cudaSetDevice(0));
  int grid = env.sm * 4, blk = 256;
  size_t n4 = SZ / sizeof(float4);
  float4* sink;
  CK(cudaMalloc(&sink, 1024));

  // -------------------------------------------------- 正确性
  hdr("B) 正确性: 一次 multimem.st 是否真的写到了所有 4 张卡");
  {
    for (int d = 0; d < nd; ++d) { CK(cudaSetDevice(d)); CK(cudaMemset((void*)uc[d], 0, SZ)); }
    CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(0));
    k_mc_st<<<grid, blk>>>((float4*)mcptr[0], n4);
    CK(cudaDeviceSynchronize());
    for (int d = 0; d < nd; ++d) {
      float h[4] = {0, 0, 0, 0};
      CK(cudaMemcpy(h, (void*)uc[d], 16, cudaMemcpyDeviceToHost));
      printf("  GPU%d 本地内存首 float4 = %.0f %.0f %.0f %.0f  %s\n", d, h[0], h[1],
             h[2], h[3], (h[0] == 1.f && h[3] == 4.f) ? "OK" : "MISS");
    }
  }

  // -------------------------------------------------- 线上字节数 (核心证据)
  hdr("C) 线上字节数: 复制到底发生在哪里 (NVML 每卡 Data 计数器)");
  {
    unsigned long long tx0[8], rx0[8], tx1[8], rx1[8];
    const int REP = 8;
    double payload = (double)SZ * REP;
    CK(cudaSetDevice(0));
    k_mc_st<<<grid, blk>>>((float4*)mcptr[0], n4);
    CK(cudaDeviceSynchronize());
    for (int d = 0; d < nd; ++d) rd_cnt(d, &tx0[d], &rx0[d]);
    for (int r = 0; r < REP; ++r) k_mc_st<<<grid, blk>>>((float4*)mcptr[0], n4);
    CK(cudaDeviceSynchronize());
    for (int d = 0; d < nd; ++d) rd_cnt(d, &tx1[d], &rx1[d]);
    printf("multimem.st  由 GPU0 发起, 逻辑 payload = %.0f MB (每卡都应收到这么多)\n",
           payload / 1e6);
    for (int d = 0; d < nd; ++d)
      printf("  GPU%d  wire Tx=%8.0f MB  wire Rx=%8.0f MB\n", d,
             (tx1[d] - tx0[d]) / 1e6, (rx1[d] - rx0[d]) / 1e6);

    // 对照: 手动 unicast 广播给 3 个 peer
    for (int d = 0; d < nd; ++d) rd_cnt(d, &tx0[d], &rx0[d]);
    CK(cudaSetDevice(0));
    for (int r = 0; r < REP; ++r)
      k_uni_bcast3<<<grid, blk>>>((float4*)uc[1], (float4*)uc[2], (float4*)uc[3], n4);
    CK(cudaDeviceSynchronize());
    for (int d = 0; d < nd; ++d) rd_cnt(d, &tx1[d], &rx1[d]);
    printf("unicast 手动广播给 3 个 peer, 逻辑 payload = %.0f MB\n", payload / 1e6);
    for (int d = 0; d < nd; ++d)
      printf("  GPU%d  wire Tx=%8.0f MB  wire Rx=%8.0f MB\n", d,
             (tx1[d] - tx0[d]) / 1e6, (rx1[d] - rx0[d]) / 1e6);
  }

  // -------------------------------------------------- 带宽
  hdr("D) 带宽对比");
  CK(cudaSetDevice(0));
  {
    double t;
    printf("%-46s %10s %10s\n", "操作", "GB/s(逻辑)", "GB/s(线上)");
    t = bench_ms([&] { k_wr1<<<grid, blk>>>((uint4*)uc[1], n4); }, 5);
    printf("%-46s %10.1f %10.1f\n", "unicast 写 1 个 peer", gbps(SZ, t), gbps(SZ, t));
    t = bench_ms([&] { k_uni_bcast3<<<grid, blk>>>((float4*)uc[1], (float4*)uc[2], (float4*)uc[3], n4); }, 5);
    printf("%-46s %10.1f %10.1f\n", "unicast 广播给 3 个 peer", gbps(SZ, t), gbps(SZ * 3, t));
    t = bench_ms([&] { k_mc_st<<<grid, blk>>>((float4*)mcptr[0], n4); }, 5);
    printf("%-46s %10.1f %10.1f\n", "multimem.st 组播给 4 张卡", gbps(SZ, t), gbps(SZ, t));
    t = bench_ms([&] { k_mc_ldreduce<<<grid, blk>>>((float4*)mcptr[0], sink, n4); }, 5);
    printf("%-46s %10.1f %10.1f\n", "multimem.ld_reduce 4 卡在网求和", gbps(SZ, t), gbps(SZ, t));
    t = bench_ms([&] { k_mc_red<<<grid, blk>>>((float4*)mcptr[0], n4); }, 5);
    printf("%-46s %10.1f %10.1f\n", "multimem.red 4 卡在网累加(无返回)", gbps(SZ, t), gbps(SZ, t));
  }

  // -------------------------------------------------- ld_reduce 线上字节
  hdr("E) multimem.ld_reduce 的线上字节 (在网归约是否真省了流量)");
  {
    unsigned long long tx0[8], rx0[8], tx1[8], rx1[8];
    const int REP = 8;
    CK(cudaSetDevice(0));
    k_mc_ldreduce<<<grid, blk>>>((float4*)mcptr[0], sink, n4);
    CK(cudaDeviceSynchronize());
    for (int d = 0; d < nd; ++d) rd_cnt(d, &tx0[d], &rx0[d]);
    for (int r = 0; r < REP; ++r) k_mc_ldreduce<<<grid, blk>>>((float4*)mcptr[0], sink, n4);
    CK(cudaDeviceSynchronize());
    for (int d = 0; d < nd; ++d) rd_cnt(d, &tx1[d], &rx1[d]);
    printf("逻辑上读了 %.0f MB 的归约结果 (源数据 4 卡各 %.0f MB)\n",
           (double)SZ * REP / 1e6, (double)SZ * REP / 1e6);
    for (int d = 0; d < nd; ++d)
      printf("  GPU%d  wire Tx=%8.0f MB  wire Rx=%8.0f MB\n", d,
             (tx1[d] - tx0[d]) / 1e6, (rx1[d] - rx0[d]) / 1e6);
  }

  printf("\n[done]\n");
  return 0;
}
