// ============================================================================
// nv19_fence_cost —— 内存屏障 / scope 的代价
//
// (A) 裸代价: 空转下连续执行 fence, 无任何 in-flight 访存。
// (B) *核心* in-flight 敏感性: 先发 K 个远端 store, 再 fence, 测 fence 代价。
//     K 扫 0,1,2,4,8,16,32,64。
//     如果 fence 代价随 K 线性增长 -> fence 在逐个等 outstanding 写的 ack
//     如果 fence 代价对 K 基本不变 -> fence 只等"最后一个/整体排空", 写是流水的
// (C) PTX 原语对比: membar.gl / membar.sys / fence.acq_rel.gpu / fence.acq_rel.sys
// (D) st.release.sys  vs  st.global + membar.sys
// ============================================================================
#include "nvl_common.cuh"

// ---------------- PTX fence 原语
__device__ __forceinline__ void membar_cta(){ asm volatile("membar.cta;":::"memory"); }
__device__ __forceinline__ void membar_gl() { asm volatile("membar.gl;" :::"memory"); }
__device__ __forceinline__ void membar_sys(){ asm volatile("membar.sys;":::"memory"); }
__device__ __forceinline__ void fence_cta(){ asm volatile("fence.acq_rel.cta;":::"memory"); }
__device__ __forceinline__ void fence_gpu(){ asm volatile("fence.acq_rel.gpu;":::"memory"); }
__device__ __forceinline__ void fence_sys(){ asm volatile("fence.acq_rel.sys;":::"memory"); }
__device__ __forceinline__ void fence_sc_gpu(){ asm volatile("fence.sc.gpu;":::"memory"); }
__device__ __forceinline__ void fence_sc_sys(){ asm volatile("fence.sc.sys;":::"memory"); }

enum FK { F_NONE=0, F_MB_CTA, F_MB_GL, F_MB_SYS, F_ACQREL_CTA, F_ACQREL_GPU,
          F_ACQREL_SYS, F_SC_GPU, F_SC_SYS, F_TF_BLOCK, F_TF_DEV, F_TF_SYS, F_N };
static const char* kFN[F_N] = {
  "(无 fence, 基线)", "membar.cta", "membar.gl", "membar.sys",
  "fence.acq_rel.cta", "fence.acq_rel.gpu", "fence.acq_rel.sys",
  "fence.sc.gpu", "fence.sc.sys",
  "__threadfence_block()", "__threadfence()", "__threadfence_system()"};

template <int F> __device__ __forceinline__ void do_fence() {
  if (F==F_MB_CTA) membar_cta();
  else if (F==F_MB_GL) membar_gl();
  else if (F==F_MB_SYS) membar_sys();
  else if (F==F_ACQREL_CTA) fence_cta();
  else if (F==F_ACQREL_GPU) fence_gpu();
  else if (F==F_ACQREL_SYS) fence_sys();
  else if (F==F_SC_GPU) fence_sc_gpu();
  else if (F==F_SC_SYS) fence_sc_sys();
  else if (F==F_TF_BLOCK) __threadfence_block();
  else if (F==F_TF_DEV) __threadfence();
  else if (F==F_TF_SYS) __threadfence_system();
}

// ---------------- (A) 裸代价
template <int F>
__global__ void k_bare(int iters, long long* out) {
  for (int i=0;i<64;++i) do_fence<F>();
  long long t0=clk();
  for (int i=0;i<iters;++i) do_fence<F>();
  long long t1=clk();
  out[0]=t1-t0;
}

// ---------------- (B) K 个 in-flight 远端 store 后 fence
// 计时只包住 fence, store 的发射成本单独用 F_NONE 组去掉
template <int F>
__global__ void k_inflight(unsigned* p, int iters, int K, long long* out,
                           size_t stride_e) {
  // warmup
  for (int i=0;i<32;++i){ st32(p+(size_t)i*stride_e,i); }
  __threadfence_system();
  long long acc=0;
  for (int it=0; it<iters; ++it) {
    size_t base=(size_t)(it%256)*K;
    // 发 K 个远端 store (不计时)
    for (int k=0;k<K;++k) st32(p + (base+k)*stride_e, (unsigned)(it+k));
    long long t0=clk();
    do_fence<F>();
    long long t1=clk();
    acc += t1-t0;
  }
  __threadfence_system();
  out[0]=acc/(iters>0?iters:1);
}

// ---------------- (D) release store vs store+membar
// kind 0: st.global.u32 (裸)  1: st.release.sys  2: st.global + membar.sys
//      3: st.release.gpu      4: st.global + membar.gl
template <int KIND>
__global__ void k_release(unsigned* p, int iters, long long* out, size_t stride_e) {
  for (int i=0;i<32;++i) st32(p+(size_t)i*stride_e,i);
  __threadfence_system();
  long long t0=clk();
  for (int i=0;i<iters;++i) {
    unsigned* a = p + (size_t)i*stride_e;
    if (KIND==0) st32(a,(unsigned)i);
    else if (KIND==1) st_release_sys(a,(unsigned)i);
    else if (KIND==2) { st32(a,(unsigned)i); membar_sys(); }
    else if (KIND==3) st_release_gpu(a,(unsigned)i);
    else { st32(a,(unsigned)i); membar_gl(); }
  }
  long long t1=clk();
  __threadfence_system();
  out[0]=t1-t0;
}

static long long* g_out;
static double runk(void(*fn)()) { return 0; }

#define RUN_BARE(F) do { \
  double best=1e30; long long h[2]; \
  for (int k=0;k<5;++k){ k_bare<F><<<1,1>>>(IT,g_out); CK(cudaDeviceSynchronize()); \
    CK(cudaMemcpy(h,g_out,sizeof h,cudaMemcpyDeviceToHost)); best=fmin(best,(double)h[0]/IT);} \
  bare[F]=best; } while(0)

int main() {
  NvlEnv env=nvl_init(2);
  const double GHZ=1.98;
  printf("# nv19_fence_cost — %s\n", env.name);
  nvl_enable_peers(env.ndev);
  CK(cudaSetDevice(0));
  CK(cudaMalloc(&g_out,64));

  const int IT=2048;
  double bare[F_N];
  RUN_BARE(F_NONE); RUN_BARE(F_MB_CTA); RUN_BARE(F_MB_GL); RUN_BARE(F_MB_SYS);
  RUN_BARE(F_ACQREL_CTA); RUN_BARE(F_ACQREL_GPU); RUN_BARE(F_ACQREL_SYS);
  RUN_BARE(F_SC_GPU); RUN_BARE(F_SC_SYS);
  RUN_BARE(F_TF_BLOCK); RUN_BARE(F_TF_DEV); RUN_BARE(F_TF_SYS);

  hdr("A) fence 裸代价 (无 in-flight 访存, 空转)");
  printf("| 屏障 | cycles | ns | 扣基线后 cyc |\n|---|---|---|---|\n");
  for (int f=0;f<F_N;++f)
    printf("| %-24s | %8.2f | %7.2f | %8.2f |\n", kFN[f], bare[f],
           cyc2ns(bare[f],GHZ), bare[f]-bare[F_NONE]);

  // ---------------------------------------------------------------- (B)
  const size_t BUF=512ull<<20;
  const size_t STRIDE_E=8192/4;
  unsigned* rem=(unsigned*)nvl_alloc(1,BUF);
  unsigned* loc=(unsigned*)nvl_alloc(0,BUF);
  CK(cudaSetDevice(0));

  int Ks[]={0,1,2,4,8,16,32,64};
  const int NK=sizeof(Ks)/sizeof(Ks[0]);
  const int IT2=512;

  auto run_if=[&](int f,int K,unsigned* p)->double{
    double best=1e30; long long h[2];
    for (int k=0;k<4;++k){
      switch(f){
        case F_NONE:      k_inflight<F_NONE><<<1,1>>>(p,IT2,K,g_out,STRIDE_E); break;
        case F_TF_BLOCK:  k_inflight<F_TF_BLOCK><<<1,1>>>(p,IT2,K,g_out,STRIDE_E); break;
        case F_TF_DEV:    k_inflight<F_TF_DEV><<<1,1>>>(p,IT2,K,g_out,STRIDE_E); break;
        case F_TF_SYS:    k_inflight<F_TF_SYS><<<1,1>>>(p,IT2,K,g_out,STRIDE_E); break;
        case F_MB_GL:     k_inflight<F_MB_GL><<<1,1>>>(p,IT2,K,g_out,STRIDE_E); break;
        default:          k_inflight<F_MB_SYS><<<1,1>>>(p,IT2,K,g_out,STRIDE_E); break;
      }
      CKLAST(); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h,g_out,sizeof h,cudaMemcpyDeviceToHost));
      best=fmin(best,(double)h[0]);
    }
    return best;
  };

  hdr("B) [核心] fence 代价 vs in-flight 远端 store 数 K (只计时 fence 本身)");
  printf("目标: 若代价随 K 线性增长 -> fence 在等每个 outstanding 写的 ack\n\n");
  printf("| K (in-flight 远端 store) | 无fence | threadfence() | threadfence_system() | membar.sys |\n");
  printf("|---|---|---|---|---|\n");
  double sysv[NK];
  for (int i=0;i<NK;++i){
    double a=run_if(F_NONE,Ks[i],rem);
    double b=run_if(F_TF_DEV,Ks[i],rem);
    double c=run_if(F_TF_SYS,Ks[i],rem);
    double d=run_if(F_MB_SYS,Ks[i],rem);
    sysv[i]=c;
    printf("| %3d | %8.1f | %13.1f | %20.1f | %10.1f |\n", Ks[i],a,b,c,d);
    fflush(stdout);
  }
  printf("\n斜率分析 (__threadfence_system):\n");
  for (int i=1;i<NK;++i)
    printf("  K=%2d -> K=%2d : 增量 %+8.1f cyc, 每多一个 in-flight 写 %+7.2f cyc\n",
           Ks[i-1],Ks[i], sysv[i]-sysv[i-1],
           (Ks[i]-Ks[i-1])? (sysv[i]-sysv[i-1])/(Ks[i]-Ks[i-1]) : 0.0);
  printf("\n  K=0 -> K=64 总增量 = %.1f cyc; 若 fence 串行等每个 ack, 预期应 ~64*远端往返(~1580)=101120 cyc\n",
         sysv[NK-1]-sysv[0]);
  printf("  实测/串行预期 = %.4f  -> %s\n",
         (sysv[NK-1]-sysv[0])/(64.0*1580.0),
         (sysv[NK-1]-sysv[0]) < 64*1580*0.2 ? "远小于串行, 说明写是流水/并行发出的, fence 只等整体排空"
                                            : "接近串行");

  hdr("B2) 同上但 store 打到本地 HBM (对照, 隔离 NVLink 因素)");
  printf("| K | 无fence | threadfence() | threadfence_system() |\n|---|---|---|---|\n");
  for (int i=0;i<NK;++i){
    double a=run_if(F_NONE,Ks[i],loc);
    double b=run_if(F_TF_DEV,Ks[i],loc);
    double c=run_if(F_TF_SYS,Ks[i],loc);
    printf("| %3d | %8.1f | %13.1f | %20.1f |\n", Ks[i],a,b,c);
    fflush(stdout);
  }

  // ---------------------------------------------------------------- (D)
  hdr("D) st.release vs st.global + membar (每次 store 的摊销代价)");
  auto run_rel=[&](int kind, unsigned* p)->double{
    double best=1e30; long long h[2];
    for (int k=0;k<5;++k){
      switch(kind){
        case 0: k_release<0><<<1,1>>>(p,IT,g_out,STRIDE_E); break;
        case 1: k_release<1><<<1,1>>>(p,IT,g_out,STRIDE_E); break;
        case 2: k_release<2><<<1,1>>>(p,IT,g_out,STRIDE_E); break;
        case 3: k_release<3><<<1,1>>>(p,IT,g_out,STRIDE_E); break;
        default:k_release<4><<<1,1>>>(p,IT,g_out,STRIDE_E); break;
      }
      CKLAST(); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h,g_out,sizeof h,cudaMemcpyDeviceToHost));
      best=fmin(best,(double)h[0]/IT);
    }
    return best;
  };
  const char* rn[5]={"st.global.u32 (裸)","st.release.sys.global",
                     "st.global + membar.sys","st.release.gpu.global",
                     "st.global + membar.gl"};
  printf("| 形式 | 本地 cyc | 本地 ns | 远端 cyc | 远端 ns |\n|---|---|---|---|---|\n");
  double rl[5],rr[5];
  for (int k=0;k<5;++k){
    rl[k]=run_rel(k,loc); rr[k]=run_rel(k,rem);
    printf("| %-24s | %8.1f | %7.1f | %8.1f | %7.1f |\n", rn[k],
           rl[k],cyc2ns(rl[k],GHZ),rr[k],cyc2ns(rr[k],GHZ));
  }
  printf("\n  远端: release.sys(%.1f) vs st+membar.sys(%.1f) 差 = %+.1f cyc -> %s\n",
         rr[1],rr[2],rr[1]-rr[2],
         rr[1]<rr[2]*0.9 ? "release 更省(硬件融合)" :
         (rr[1]>rr[2]*1.1?"release 更贵":"两者等价"));
  printf("  远端: release.gpu(%.1f) vs release.sys(%.1f) 差 = %+.1f cyc -> scope 提升的代价\n",
         rr[3],rr[1],rr[1]-rr[3]);

  printf("\n[done]\n");
  return 0;
}
