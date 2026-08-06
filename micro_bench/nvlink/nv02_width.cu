// ============================================================================
// nv02_width —— 访问宽度 vs 有效带宽
//
// 假设: SM 的 LSU 把 warp 的访存合并成若干「请求」发给 memory subsystem;
//       远端(NVLink)路径上每个请求都要付固定 packet 头开销。
//       如果每线程宽度 W 变小, 但保持 warp 内地址连续(完全合并),
//       那么 warp 一次访存覆盖的字节 = 32*W:
//          W=16B -> 512B (4 条 128B line)
//          W=8B  -> 256B (2 条 128B line)
//          W=4B  -> 128B (1 条 128B line)
//          W=2B  ->  64B (半条 line)
//          W=1B  ->  32B (1/4 条 line)
//       若最小传输粒度是 G 字节, 则 W<G/32 时会出现放大 -> 带宽掉。
//
// 控制变量: **总有用字节数固定 = TOTAL**, 线程总工作量按 W 缩放,
//           grid/block 不变, 每线程循环次数按 W 反比调整。
//           这样比较的是「搬同样多的有用数据要多久」。
// ============================================================================
#include "nvl_common.cuh"

// ---------------------------------------------------------------- 写 kernel
// 每线程处理 W 字节 * ITER 次, warp 内地址连续: addr = (gtid + k*stride)*W
template <int W>
__global__ void k_wr_w(char* __restrict__ base, size_t nelem, int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  uint4 v16 = make_uint4(0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);
  for (int k = 0; k < iter; ++k) {
    size_t i = g + (size_t)k * s;
    i &= (nelem - 1);  // nelem 是 2 的幂
    char* p = base + i * W;
    if (W == 16)      st128(p, v16);
    else if (W == 8)  st64(p, make_uint2(0x11111111u, 0x22222222u));
    else if (W == 4)  st32(p, 0x11111111u);
    else if (W == 2)  st16(p, (unsigned short)0x1111);
    else              st8(p, (unsigned char)0x11);
  }
}

// ---------------------------------------------------------------- 读 kernel
template <int W>
__global__ void k_rd_w(const char* __restrict__ base, unsigned* sink,
                       size_t nelem, int iter) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (int k = 0; k < iter; ++k) {
    size_t i = g + (size_t)k * s;
    i &= (nelem - 1);
    const char* p = base + i * W;
    if (W == 16) { uint4 v = ld128(p); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    else if (W == 8) { uint2 v = ld64(p); acc ^= v.x ^ v.y; }
    else if (W == 4) { acc ^= ld32(p); }
    else if (W == 2) {
      unsigned short h;
      asm volatile("ld.global.u16 %0, [%1];" : "=h"(h) : "l"(p));
      acc ^= h;
    } else {
      unsigned short h;
      asm volatile("ld.global.u8 %0, [%1];" : "=h"(h) : "l"(p));
      acc ^= h;
    }
  }
  if (acc == 0xdeadbeefu) *sink = acc;
}

struct Res { double wr_rem, rd_rem, wr_loc, rd_loc; };

template <int W>
static Res run_w(char* rem, char* loc, unsigned* sink, size_t BYTES, int grid,
                 int blk) {
  size_t nelem = BYTES / W;              // 元素个数(2 的幂)
  size_t threads = (size_t)grid * blk;
  int iter = (int)(nelem / threads);     // 每线程迭代数, 总覆盖 = BYTES 有用字节
  if (iter < 1) iter = 1;
  double payload = (double)threads * iter * W;

  Res r;
  double t;
  t = bench_ms([&] { k_wr_w<W><<<grid, blk>>>(rem, nelem, iter); }, 3);
  r.wr_rem = gbps(payload, t);
  t = bench_ms([&] { k_rd_w<W><<<grid, blk>>>(rem, sink, nelem, iter); }, 3);
  r.rd_rem = gbps(payload, t);
  t = bench_ms([&] { k_wr_w<W><<<grid, blk>>>(loc, nelem, iter); }, 3);
  r.wr_loc = gbps(payload, t);
  t = bench_ms([&] { k_rd_w<W><<<grid, blk>>>(loc, sink, nelem, iter); }, 3);
  r.rd_loc = gbps(payload, t);
  return r;
}

int main() {
  NvlEnv env = nvl_init(2);
  nvl_enable_peers(env.ndev);
  printf("# nv02_width — 访问宽度 vs 有效带宽 (%s, %d SM, %.3f GHz)\n", env.name,
         env.sm, env.clkGHz);

  const size_t BYTES = 256ull << 20;  // 256 MB 有用数据
  CK(cudaSetDevice(0));
  char* loc = (char*)nvl_alloc(0, BYTES);
  unsigned* sink = (unsigned*)nvl_alloc(0, 1024);
  char* rem = (char*)nvl_alloc(1, BYTES);
  CK(cudaSetDevice(0));

  int grid = env.sm * 4, blk = 256;

  hdr("每线程访问宽度扫描 (warp 内地址完全连续, 总有用字节固定 256MB)");
  printf("| 宽度 W | warp 覆盖 | 远端写 GB/s | 远端读 GB/s | 本地写 GB/s | 本地读 GB/s | 远端写/16B | 远端读/16B |\n");
  printf("|---|---|---|---|---|---|---|---|\n");

  Res r16 = run_w<16>(rem, loc, sink, BYTES, grid, blk);
  Res r8  = run_w<8>(rem, loc, sink, BYTES, grid, blk);
  Res r4  = run_w<4>(rem, loc, sink, BYTES, grid, blk);
  Res r2  = run_w<2>(rem, loc, sink, BYTES, grid, blk);
  Res r1  = run_w<1>(rem, loc, sink, BYTES, grid, blk);

  struct Row { const char* w; int cover; Res r; };
  Row rows[] = {{"16B", 512, r16}, {"8B", 256, r8}, {"4B", 128, r4},
                {"2B", 64, r2},    {"1B", 32, r1}};
  for (auto& x : rows)
    printf("| %s | %dB | %.1f | %.1f | %.1f | %.1f | %.3f | %.3f |\n", x.w,
           x.cover, x.r.wr_rem, x.r.rd_rem, x.r.wr_loc, x.r.rd_loc,
           x.r.wr_rem / r16.wr_rem, x.r.rd_rem / r16.rd_rem);

  printf("\n[done]\n");
  return 0;
}
