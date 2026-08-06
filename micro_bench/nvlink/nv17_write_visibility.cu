// ============================================================================
// nv17_write_visibility —— 远端写的完成语义与写回行为
//
// (a) 发射延迟 vs 完成延迟
//     fire   : st; clk()                       -> 只测把请求塞进 LSU/队列的代价
//     fence  : st; __threadfence_system(); clk()-> 等到系统级可见
//     差值 = 写确认(ack)往返 + 队列排空
//     对 本地 / 远端 分别测, 相减去掉本地 fence 固有成本。
//
// (b) 写后读
//     w->r same  : st X; ld X    (同地址)
//     w->r diff  : st X; ld Y    (不同地址, 相距远)
//     若 same 明显更快 -> 存在 write-combining / store buffer 转发
//
// (c) 计数器: 远端写时 Tx/Rx 比例 -> 推断 ack 包大小
// ============================================================================
#include "nvl_common.cuh"
#include "nvl_counters.cuh"

// ------------------------------------------------ (a) 发射 vs 完成
// mode: 0=纯 store 无 fence, 1=threadfence_block, 2=threadfence, 3=threadfence_system
template <int MODE>
__global__ void k_store_lat(unsigned* p, int iters, long long* out, size_t stride_elem) {
  // warmup
  for (int i=0;i<64;++i) st32(p + (i%iters)*stride_elem, i);
  __threadfence_system();
  long long t0 = clk();
  for (int i=0;i<iters;++i) {
    st32(p + (size_t)i*stride_elem, (unsigned)i);
    if (MODE==1) __threadfence_block();
    else if (MODE==2) __threadfence();
    else if (MODE==3) __threadfence_system();
  }
  long long t1 = clk();
  // 末尾统一 fence, 保证所有 mode 最终都完成(但不计入 MODE=0 的计时)
  __threadfence_system();
  out[0]=t1-t0;
}

// ------------------------------------------------ (b) 写后读
// kind: 0 = 写后读同地址, 1 = 写后读远地址, 2 = 纯读(基线)
template <int KIND>
__global__ void k_wr_then_rd(unsigned* p, int iters, long long* out,
                             size_t stride_elem, size_t far_off) {
  unsigned acc=0;
  for (int i=0;i<64;++i) { st32(p+(size_t)i*stride_elem, i); acc+=ld32_v(p+(size_t)i*stride_elem); }
  __threadfence_system();
  long long t0=clk();
  for (int i=0;i<iters;++i) {
    unsigned* a = p + (size_t)i*stride_elem;
    if (KIND!=2) st32(a, (unsigned)i+acc);
    unsigned* b = (KIND==1) ? (a + far_off) : a;
    acc += ld32_v(b);   // volatile: 强制真的去 memory, 不让编译器转发
  }
  long long t1=clk();
  out[0]=t1-t0; out[1]=acc;
}

// ------------------------------------------------ (c) 计数器用: 批量远端写
__global__ void k_wr_bulk(uint4* dst, size_t n) {
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  size_t s=(size_t)gridDim.x*blockDim.x;
  uint4 v=make_uint4(1,2,3,4);
  for (;i<n;i+=s) st128(dst+i,v);
}
__global__ void k_wr_bulk32(unsigned* dst, size_t n) {
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  size_t s=(size_t)gridDim.x*blockDim.x;
  for (;i<n;i+=s) st32(dst+i,(unsigned)i);
}
__global__ void k_rd_bulk(const uint4* src, size_t n, uint4* sink) {
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  size_t s=(size_t)gridDim.x*blockDim.x;
  uint4 acc=make_uint4(0,0,0,0);
  for (;i<n;i+=s){uint4 v=ld128_v(src+i);acc.x^=v.x;acc.y^=v.y;acc.z^=v.z;acc.w^=v.w;}
  if (acc.x==0xdeadbeefu&&acc.y==0xfeedfaceu)*sink=acc;
}

static double best_of(void (*launch)(), int) { return 0; }

int main() {
  NvlEnv env=nvl_init(2);
  const double GHZ=1.98;
  printf("# nv17_write_visibility — %s\n", env.name);
  nvl_enable_peers(env.ndev);
  CK(cudaSetDevice(0));
  long long* d_out; CK(cudaMalloc(&d_out,64));
  uint4* sink=(uint4*)nvl_alloc(0,1024);
  CK(cudaSetDevice(0));

  const size_t BUF = 512ull<<20;
  const int ITERS = 1024;
  const size_t STRIDE_E = 8192/4;   // 8KB 步长, 每次写不同 cache line

  unsigned* loc=(unsigned*)nvl_alloc(0,BUF);
  unsigned* rem=(unsigned*)nvl_alloc(1,BUF);
  CK(cudaSetDevice(0));

  auto run = [&](int mode, unsigned* p)->double{
    double best=1e30; long long h[2];
    for (int k=0;k<5;++k){
      switch(mode){
        case 0: k_store_lat<0><<<1,1>>>(p,ITERS,d_out,STRIDE_E); break;
        case 1: k_store_lat<1><<<1,1>>>(p,ITERS,d_out,STRIDE_E); break;
        case 2: k_store_lat<2><<<1,1>>>(p,ITERS,d_out,STRIDE_E); break;
        default:k_store_lat<3><<<1,1>>>(p,ITERS,d_out,STRIDE_E); break;
      }
      CKLAST(); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h,d_out,sizeof h,cudaMemcpyDeviceToHost));
      best=fmin(best,(double)h[0]/ITERS);
    }
    return best;
  };

  // ================================================== (a)
  hdr("(a) 单个 store 的发射延迟 vs 完成延迟 (每次写不同 8KB-stride 地址)");
  double f_l[4], f_r[4];
  const char* mn[4]={"st 裸发射 (无 fence)","st + __threadfence_block()",
                     "st + __threadfence()","st + __threadfence_system()"};
  for (int m=0;m<4;++m){ f_l[m]=run(m,loc); f_r[m]=run(m,rem); }
  printf("| 模式 | 本地 cyc | 本地 ns | 远端 cyc | 远端 ns | 远端-本地 cyc |\n|---|---|---|---|---|---|\n");
  for (int m=0;m<4;++m)
    printf("| %-30s | %8.1f | %7.1f | %8.1f | %7.1f | %8.1f |\n", mn[m],
           f_l[m], cyc2ns(f_l[m],GHZ), f_r[m], cyc2ns(f_r[m],GHZ), f_r[m]-f_l[m]);
  printf("\n推导:\n");
  printf("  远端写确认往返 (sys fence - 裸发射, 远端)   = %8.1f cyc = %7.1f ns\n",
         f_r[3]-f_r[0], cyc2ns(f_r[3]-f_r[0],GHZ));
  printf("  本地写确认往返 (sys fence - 裸发射, 本地)   = %8.1f cyc = %7.1f ns\n",
         f_l[3]-f_l[0], cyc2ns(f_l[3]-f_l[0],GHZ));
  printf("  纯 NVLink ack 附加成本 (两者相减)          = %8.1f cyc = %7.1f ns\n",
         (f_r[3]-f_r[0])-(f_l[3]-f_l[0]), cyc2ns((f_r[3]-f_r[0])-(f_l[3]-f_l[0]),GHZ));

  // ================================================== (b)
  hdr("(b) 写后读: 同地址 vs 不同地址 (看是否有 write-combining/store-forward)");
  auto run_b=[&](int kind, unsigned* p)->double{
    double best=1e30; long long h[2];
    for (int k=0;k<5;++k){
      switch(kind){
        case 0: k_wr_then_rd<0><<<1,1>>>(p,ITERS,d_out,STRIDE_E,4096); break;
        case 1: k_wr_then_rd<1><<<1,1>>>(p,ITERS,d_out,STRIDE_E,4096); break;
        default:k_wr_then_rd<2><<<1,1>>>(p,ITERS,d_out,STRIDE_E,4096); break;
      }
      CKLAST(); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h,d_out,sizeof h,cudaMemcpyDeviceToHost));
      best=fmin(best,(double)h[0]/ITERS);
    }
    return best;
  };
  const char* bn[3]={"写 X 后读 X (同地址)","写 X 后读 X+16KB (异地址)","只读 (无写, 基线)"};
  printf("| 模式 | 本地 cyc | 远端 cyc | 远端 ns |\n|---|---|---|---|\n");
  double b_l[3],b_r[3];
  for (int k=0;k<3;++k){ b_l[k]=run_b(k,loc); b_r[k]=run_b(k,rem);
    printf("| %-28s | %8.1f | %8.1f | %8.1f |\n", bn[k], b_l[k], b_r[k], cyc2ns(b_r[k],GHZ)); }
  printf("\n  远端 同地址 vs 异地址 差 = %.1f cyc (%.1f ns)  -> %s\n",
         b_r[1]-b_r[0], cyc2ns(b_r[1]-b_r[0],GHZ),
         fabs(b_r[1]-b_r[0]) < 30 ? "无显著差异, 无跨 NVLink 的 store 转发" : "存在转发/合并效应");
  printf("  远端 写+读 vs 纯读 差   = %.1f cyc  -> store 是否阻塞后续 load\n",
         b_r[0]-b_r[2]);

  // ================================================== (c)
  hdr("(c) 计数器: 远端写的 Tx/Rx 流量比 -> 反推 ack 包开销");
  int grid=env.sm*4, blk=256;
  const size_t PB = 256ull<<20;
  uint4* wbuf=(uint4*)nvl_alloc(1,PB);
  CK(cudaSetDevice(0));
  printf("| 负载 | payload MB | dataTx MB | dataRx MB | rawTx MB | rawRx MB | Rx/Tx(data) | Rx/Tx(raw) | raw/data |\n");
  printf("|---|---|---|---|---|---|---|---|---|\n");
  struct Case { const char* name; int kind; };
  // kind 0 = 128bit 写, 1 = 32bit 写, 2 = 128bit 读(对照)
  for (int kind=0;kind<3;++kind) {
    // warmup
    if (kind==0) k_wr_bulk<<<grid,blk>>>(wbuf,PB/16);
    else if (kind==1) k_wr_bulk32<<<grid,blk>>>((unsigned*)wbuf,PB/4);
    else k_rd_bulk<<<grid,blk>>>(wbuf,PB/16,sink);
    CK(cudaDeviceSynchronize());
    NvlCounters a=nvl_read_counters(0);
    const int R=4;
    for (int r=0;r<R;++r){
      if (kind==0) k_wr_bulk<<<grid,blk>>>(wbuf,PB/16);
      else if (kind==1) k_wr_bulk32<<<grid,blk>>>((unsigned*)wbuf,PB/4);
      else k_rd_bulk<<<grid,blk>>>(wbuf,PB/16,sink);
    }
    CK(cudaDeviceSynchronize());
    NvlCounters d=nvl_diff(a,nvl_read_counters(0));
    double payload=(double)PB*R;
    const char* nm = kind==0?"远端写 st.global.v4 (128b)":(kind==1?"远端写 st.global.u32 (32b)":"远端读 ld.volatile.v4 (对照)");
    printf("| %-28s | %8.0f | %9.1f | %9.1f | %8.1f | %8.1f | %10.4f | %9.4f | %8.4f |\n",
           nm, payload/1e6, d.dataTx/1e6, d.dataRx/1e6, d.rawTx/1e6, d.rawRx/1e6,
           d.dataTx? (double)d.dataRx/d.dataTx : 0.0,
           d.rawTx ? (double)d.rawRx/d.rawTx  : 0.0,
           (d.dataTx+d.dataRx)? (double)(d.rawTx+d.rawRx)/(d.dataTx+d.dataRx):0.0);
    fflush(stdout);
  }
  printf("\n对写负载: dataTx≈payload 是写数据本身; rawRx 是回来的 ack 流量。\n");
  printf("每 128B 写事务的 ack 线上字节 ≈ rawRx/(payload/128)\n");

  printf("\n[done]\n");
  return 0;
}
