// ============================================================================
// nv18_hop_jitter —— 远端读延迟的分布与抖动
//
// 方法: <<<1,1>>> 单发采样。每次采样读一个"新"地址(伪随机跳, 步长 8KB,
//       working set 512MB >> 60MB L2), 保证每发都是 cold miss。
//       用 ld.global.cv 强制绕过 GPU0 本地缓存(nv15 已证明 .cv 有效),
//       否则小概率命中本地 L2 会污染分布左尾。
//       采样值先存 shared memory, kernel 末尾一次性写回 global,
//       避免采样循环内的 global store 干扰。
//
// 输出: min / p50 / p90 / p99 / max + 直方图。
// 对象: GPU0->GPU1, 0->2, 0->3 (验证 NVSwitch 对称性) + 本地 HBM 对照。
// ============================================================================
#include "nvl_common.cuh"
#include <vector>
#include <algorithm>

#define NSAMP 4096

__device__ __forceinline__ unsigned ld_cv_(const unsigned* p) {
  unsigned v; asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

// 每次采样: 读 chain[cur] 得到下一跳(真串行), 同时记录本次耗时
__global__ void k_sample(const unsigned* chain, unsigned* out_cyc, int nsamp) {
  __shared__ unsigned s[NSAMP];
  unsigned cur = 0;
  // warmup: 让 TLB / 指令流水稳定
  for (int i=0;i<1024;++i) cur = ld_cv_(chain+cur);
  for (int i=0;i<nsamp;++i) {
    long long t0 = clk();
    cur = ld_cv_(chain + cur);
    long long t1 = clk();
    s[i] = (unsigned)(t1-t0);
  }
  __syncthreads();
  for (int i=0;i<nsamp;++i) out_cyc[i]=s[i];
  if (cur==0xffffffffu) out_cyc[0]=cur;  // 防优化
}

// 测量 clk() 本身的开销 (背靠背两次 clk)
__global__ void k_clk_ovh(unsigned* out, int nsamp) {
  __shared__ unsigned s[NSAMP];
  for (int i=0;i<64;++i){ volatile long long a=clk(); (void)a; }
  for (int i=0;i<nsamp;++i){ long long t0=clk(); long long t1=clk(); s[i]=(unsigned)(t1-t0); }
  __syncthreads();
  for (int i=0;i<nsamp;++i) out[i]=s[i];
}

static std::vector<unsigned> build_chain(size_t bytes, size_t stride, unsigned seed) {
  size_t nnode=bytes/stride;
  std::vector<unsigned> h(bytes/4,0u);
  std::vector<unsigned> idx(nnode);
  for (size_t i=0;i<nnode;++i) idx[i]=(unsigned)(i*stride/4);
  std::vector<unsigned> perm(idx.begin()+1, idx.end());
  // 线性同余打乱(可复现)
  unsigned st=seed;
  for (size_t i=perm.size();i>1;--i){ st=st*1664525u+1013904223u; std::swap(perm[i-1],perm[st%i]); }
  unsigned cur=idx[0];
  for (unsigned p:perm){h[cur]=p;cur=p;}
  h[cur]=idx[0];
  return h;
}

struct Stat { double mn,p50,p90,p99,p999,mx,mean,sd; };
static Stat calc(std::vector<unsigned> v) {
  std::sort(v.begin(), v.end());
  Stat s;
  size_t n=v.size();
  s.mn=v[0]; s.mx=v[n-1];
  s.p50=v[n*50/100]; s.p90=v[n*90/100]; s.p99=v[n*99/100]; s.p999=v[n*999/1000];
  double sum=0; for(unsigned x:v) sum+=x; s.mean=sum/n;
  double q=0; for(unsigned x:v){ double d=x-s.mean; q+=d*d; } s.sd=sqrt(q/n);
  return s;
}

static void histo(const char* tag, std::vector<unsigned> v, int nb=14) {
  std::sort(v.begin(),v.end());
  unsigned lo=v.front(), hi=v[v.size()*995/1000];  // 掐掉极端右尾便于看形状
  if (hi<=lo) hi=lo+1;
  unsigned w=(hi-lo)/nb+1;
  std::vector<int> cnt(nb+1,0);
  for (unsigned x:v){ int b=(x<lo)?0:(int)((x-lo)/w); if(b>nb)b=nb; cnt[b]++; }
  printf("\n直方图 %s (bucket=%u cyc, 最后一桶=溢出/尾部):\n", tag, w);
  int mxc=0; for(int c:cnt) mxc=std::max(mxc,c);
  for (int b=0;b<=nb;++b){
    if (!cnt[b]) continue;
    char rng[48];
    if (b==nb) snprintf(rng,48,">=%u", lo+w*nb);
    else snprintf(rng,48,"%u-%u", lo+w*b, lo+w*(b+1)-1);
    int bar=mxc? cnt[b]*50/mxc : 0;
    printf("  %-14s %6d %5.1f%% |", rng, cnt[b], 100.0*cnt[b]/v.size());
    for (int i=0;i<bar;++i) putchar('#');
    printf("\n");
  }
}

int main() {
  NvlEnv env=nvl_init(2);
  const double GHZ=1.98;
  printf("# nv18_hop_jitter — %s x%d, %d 次单发采样/目标\n", env.name, env.ndev, NSAMP);
  nvl_enable_peers(env.ndev);

  const size_t BUF = 512ull<<20;   // >> 60MB L2
  const size_t STRIDE = 8192;

  CK(cudaSetDevice(0));
  unsigned* d_cyc; CK(cudaMalloc(&d_cyc, NSAMP*sizeof(unsigned)));
  std::vector<unsigned> host(NSAMP);

  // ---- clk 开销基线
  k_clk_ovh<<<1,1>>>(d_cyc,NSAMP); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(host.data(),d_cyc,NSAMP*4,cudaMemcpyDeviceToHost));
  Stat co=calc(host);
  printf("\nclk() 背靠背开销: min=%.0f p50=%.0f max=%.0f cyc "
         "(下表数字未扣除, 影响 <%.0f cyc)\n", co.mn, co.p50, co.mx, co.p50);

  std::vector<unsigned> hc = build_chain(BUF, STRIDE, 20260807u);

  struct Target { const char* name; int dev; };
  std::vector<Target> tg;
  tg.push_back({"local HBM (GPU0)", 0});
  for (int d=1; d<env.ndev; ++d) {
    static char nm[8][32];
    snprintf(nm[d],32,"GPU0 -> GPU%d", d);
    tg.push_back({nm[d], d});
  }

  hdr("A) 单发远端读延迟分布 (cycles), 512MB WS, ld.global.cv 强制 miss");
  printf("| 目标 | min | p50 | p90 | p99 | p99.9 | max | mean | stddev | p99/p50 |\n");
  printf("|---|---|---|---|---|---|---|---|---|---|\n");
  std::vector<std::vector<unsigned>> all;
  std::vector<Stat> stats;
  for (auto& t : tg) {
    unsigned* d=(unsigned*)nvl_alloc(t.dev, BUF);
    CK(cudaMemcpy(d,hc.data(),BUF,cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(0));
    k_sample<<<1,1>>>(d,d_cyc,NSAMP); CKLAST(); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(host.data(),d_cyc,NSAMP*4,cudaMemcpyDeviceToHost));
    Stat s=calc(host);
    printf("| %-16s | %5.0f | %5.0f | %5.0f | %5.0f | %5.0f | %6.0f | %6.1f | %6.1f | %5.2f |\n",
           t.name,s.mn,s.p50,s.p90,s.p99,s.p999,s.mx,s.mean,s.sd,s.p50?s.p99/s.p50:0);
    fflush(stdout);
    all.push_back(host); stats.push_back(s);
    CK(cudaFree(d)); CK(cudaSetDevice(0));
  }

  hdr("B) 同上, 换算 ns @1.98GHz");
  printf("| 目标 | min | p50 | p90 | p99 | max |\n|---|---|---|---|---|---|\n");
  for (size_t i=0;i<tg.size();++i){
    Stat&s=stats[i];
    printf("| %-16s | %6.1f | %6.1f | %6.1f | %6.1f | %7.1f |\n", tg[i].name,
           cyc2ns(s.mn,GHZ),cyc2ns(s.p50,GHZ),cyc2ns(s.p90,GHZ),
           cyc2ns(s.p99,GHZ),cyc2ns(s.mx,GHZ));
  }

  hdr("C) NVSwitch 对称性: 各 peer 相对 GPU1 的 p50 偏差");
  if (stats.size()>1) {
    double ref=stats[1].p50;
    for (size_t i=1;i<tg.size();++i)
      printf("  %-16s p50=%6.0f cyc  偏差 %+6.1f cyc (%+5.2f%%)\n", tg[i].name,
             stats[i].p50, stats[i].p50-ref, ref?100.0*(stats[i].p50-ref)/ref:0);
    double mn=1e30,mx=-1e30;
    for (size_t i=1;i<tg.size();++i){ mn=fmin(mn,stats[i].p50); mx=fmax(mx,stats[i].p50); }
    printf("\n  peer 间 p50 极差 = %.0f cyc (%.1f ns) -> %s\n", mx-mn, cyc2ns(mx-mn,GHZ),
           (mx-mn) < 20 ? "完全对称, 与 1-hop NVSwitch 全互联一致" : "存在不对称");
  }

  hdr("D) 直方图");
  for (size_t i=0;i<tg.size();++i) histo(tg[i].name, all[i]);

  printf("\n[done]\n");
  return 0;
}
