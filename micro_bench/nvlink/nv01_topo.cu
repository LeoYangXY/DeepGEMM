// ============================================================================
// nv01_topo —— NVLink 拓扑 / 链路参数 / 基线带宽
//
// 目标:
//   1. 用 NVML 枚举每条 link 的对端类型(GPU 还是 NVSwitch)、速率、版本
//   2. P2P 可达矩阵 + peer atomic 能力
//   3. 三条基线: 本地 HBM / 远端 NVLink 写 / 远端 NVLink 读 / Copy Engine
//   这一支是后面所有实验的"标尺"。
// ============================================================================
#include "nvl_common.cuh"
#include <nvml.h>

static const char* nvml_dev_type(nvmlIntNvLinkDeviceType_t t) {
  switch (t) {
    case NVML_NVLINK_DEVICE_TYPE_GPU: return "GPU";
    case NVML_NVLINK_DEVICE_TYPE_IBMNPU: return "IBMNPU";
    case NVML_NVLINK_DEVICE_TYPE_SWITCH: return "NVSWITCH";
    default: return "UNKNOWN";
  }
}

// 远端写: 每线程连续 float4 store
__global__ void k_wr(uint4* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v = make_uint4(1, 2, 3, 4);
  for (; i < n; i += s) st128(dst + i, v);
}
// 远端读
__global__ void k_rd(const uint4* __restrict__ src, uint4* sink, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (; i < n; i += s) {
    uint4 v = ld128(src + i);
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if (acc.x == 0xdeadbeefu && acc.y == 0xdeadbeefu) *sink = acc;
}

int main() {
  NvlEnv env = nvl_init(2);
  printf("# nv01_topo  —  %s x%d, %d SM, SM clock %.3f GHz\n", env.name,
         env.ndev, env.sm, env.clkGHz);

  // ---------------------------------------------------------- NVML link 枚举
  hdr("1) NVML 链路枚举 (GPU0 的每条 link 对端是谁)");
  if (nvmlInit_v2() == NVML_SUCCESS) {
    unsigned n = 0;
    nvmlDeviceGetCount_v2(&n);
    printf("NVML 见到 %u 个 GPU\n", n);
    for (unsigned d = 0; d < n && d < 4; ++d) {
      nvmlDevice_t h;
      if (nvmlDeviceGetHandleByIndex_v2(d, &h) != NVML_SUCCESS) continue;
      int active = 0;
      unsigned ver = 0;
      unsigned long long cap = 0;
      printf("\nGPU%u:\n", d);
      for (unsigned l = 0; l < NVML_NVLINK_MAX_LINKS; ++l) {
        nvmlEnableState_t st;
        if (nvmlDeviceGetNvLinkState(h, l, &st) != NVML_SUCCESS) continue;
        if (st != NVML_FEATURE_ENABLED) continue;
        ++active;
        nvmlDeviceGetNvLinkVersion(h, l, &ver);
        nvmlPciInfo_t pci;
        nvmlIntNvLinkDeviceType_t dt = (nvmlIntNvLinkDeviceType_t)-1;
        nvmlDeviceGetNvLinkRemoteDeviceType(h, l, &dt);
        const char* rname = "?";
        char buf[64] = "?";
        if (nvmlDeviceGetNvLinkRemotePciInfo_v2(h, l, &pci) == NVML_SUCCESS) {
          snprintf(buf, sizeof buf, "%s", pci.busId);
          rname = buf;
        }
        nvmlDeviceGetNvLinkCapability(h, l, NVML_NVLINK_CAP_P2P_SUPPORTED, (unsigned*)&cap);
        if (d == 0)
          printf("  link%2u  ver=%u  remote=%-10s %s  p2p_cap=%llu\n", l, ver,
                 nvml_dev_type(dt), rname, cap);
      }
      printf("  active links = %d, NVLink version = %u\n", active, ver);
    }
    nvmlShutdown();
  } else {
    printf("NVML 初始化失败, 跳过\n");
  }

  // ---------------------------------------------------------- P2P 矩阵
  hdr("2) P2P 能力矩阵 (canAccessPeer / atomic / perfRank)");
  printf("     ");
  for (int j = 0; j < env.ndev; ++j) printf("   GPU%d", j);
  printf("\n");
  for (int i = 0; i < env.ndev; ++i) {
    printf("GPU%d ", i);
    for (int j = 0; j < env.ndev; ++j) {
      if (i == j) { printf("      X"); continue; }
      int can = 0, atom = 0, rank = 0;
      cudaDeviceCanAccessPeer(&can, i, j);
      cudaDeviceGetP2PAttribute(&atom, cudaDevP2PAttrNativeAtomicSupported, i, j);
      cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, i, j);
      printf("  %d/%d/%d", can, atom, rank);
    }
    printf("\n");
  }
  printf("(格式: canAccess / nativeAtomic / perfRank，perfRank 越小越快)\n");

  nvl_enable_peers(env.ndev);

  // ---------------------------------------------------------- 基线带宽
  hdr("3) 基线带宽 (256 MB, SM 驱动 vs Copy Engine)");
  const size_t BYTES = 256ull << 20;
  const size_t N = BYTES / sizeof(uint4);

  CK(cudaSetDevice(0));
  uint4* local = (uint4*)nvl_alloc(0, BYTES);
  uint4* sink = (uint4*)nvl_alloc(0, 1024);
  uint4* remote1 = (uint4*)nvl_alloc(1, BYTES);
  CK(cudaSetDevice(0));

  int grid = env.sm * 4, blk = 256;

  double t;
  printf("%-34s %10s\n", "路径", "GB/s");
  t = bench_ms([&] { k_wr<<<grid, blk, 0, 0>>>(local, N); }, 5);
  printf("%-34s %10.1f\n", "本地 HBM 写 (st.global.v4)", gbps(BYTES, t));
  t = bench_ms([&] { k_rd<<<grid, blk, 0, 0>>>(local, sink, N); }, 5);
  printf("%-34s %10.1f\n", "本地 HBM 读 (ld.global.v4)", gbps(BYTES, t));
  t = bench_ms([&] { k_wr<<<grid, blk, 0, 0>>>(remote1, N); }, 5);
  printf("%-34s %10.1f\n", "NVLink 远端写 GPU0->GPU1", gbps(BYTES, t));
  t = bench_ms([&] { k_rd<<<grid, blk, 0, 0>>>(remote1, sink, N); }, 5);
  printf("%-34s %10.1f\n", "NVLink 远端读 GPU0<-GPU1", gbps(BYTES, t));

  t = bench_ms([&] { cudaMemcpyPeerAsync(remote1, 1, local, 0, BYTES, 0); }, 5);
  printf("%-34s %10.1f\n", "Copy Engine memcpyPeer 0->1", gbps(BYTES, t));
  t = bench_ms([&] { cudaMemcpyPeerAsync(local, 0, remote1, 1, BYTES, 0); }, 5);
  printf("%-34s %10.1f\n", "Copy Engine memcpyPeer 1->0", gbps(BYTES, t));

  // ---------------------------------------------------------- 逐对
  hdr("4) 逐 GPU 对的远端写带宽 (看 NVSwitch 是否对称)");
  printf("%-6s", "src\\dst");
  for (int j = 0; j < env.ndev; ++j) printf("%9d", j);
  printf("\n");
  for (int i = 0; i < env.ndev; ++i) {
    CK(cudaSetDevice(i));
    uint4* self = (uint4*)nvl_alloc(i, BYTES);
    printf("%-6d", i);
    for (int j = 0; j < env.ndev; ++j) {
      uint4* dst;
      if (i == j) dst = self;
      else dst = (uint4*)nvl_alloc(j, BYTES);
      CK(cudaSetDevice(i));
      double ms = bench_ms([&] { k_wr<<<grid, blk, 0, 0>>>(dst, N); }, 5);
      printf("%9.1f", gbps(BYTES, ms));
      if (i != j) { CK(cudaFree(dst)); CK(cudaSetDevice(i)); }
    }
    printf("\n");
    CK(cudaFree(self));
  }
  printf("(单位 GB/s；对角线是本地 HBM)\n");

  printf("\n[done]\n");
  return 0;
}
