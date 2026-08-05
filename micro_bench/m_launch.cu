#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
using namespace std::chrono;
struct Big{ char b[4096]; };
__global__ void empty(){}
__global__ void withp(Big b,int* o){ if(threadIdx.x==9999) o[0]=b.b[0]; }
__global__ void tiny(int* o){ if(threadIdx.x==9999) o[0]=1; }
static double us(steady_clock::time_point a,steady_clock::time_point b){
  return duration_cast<nanoseconds>(b-a).count()/1000.0;
}
int main(){
  int* o; cudaMalloc(&o,4); Big bg{};
  cudaStream_t s; cudaStreamCreate(&s);
  for(int i=0;i<100;++i) empty<<<1,1,0,s>>>(); cudaStreamSynchronize(s);
  const int N=2000; auto t0=steady_clock::now();
  for(int i=0;i<N;++i) empty<<<1,1,0,s>>>();
  auto t1=steady_clock::now(); cudaStreamSynchronize(s);
  auto t2=steady_clock::now();
  printf("CPU launch API      : %7.3f us/kernel\n",us(t0,t1)/N);
  printf("back-to-back GPU    : %7.3f us/kernel\n",us(t0,t2)/N);
  double acc=0;
  for(int i=0;i<200;++i){ auto a=steady_clock::now(); empty<<<1,1,0,s>>>(); cudaStreamSynchronize(s); acc+=us(a,steady_clock::now()); }
  printf("launch+sync latency : %7.3f us\n",acc/200);
  auto t3=steady_clock::now();
  for(int i=0;i<N;++i) withp<<<1,1,0,s>>>(bg,o);
  cudaStreamSynchronize(s);
  printf("4KB-param launch    : %7.3f us/kernel\n",us(t3,steady_clock::now())/N);
  cudaGraph_t g; cudaGraphExec_t ge;
  cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);
  for(int i=0;i<100;++i) tiny<<<1,32,0,s>>>(o);
  cudaStreamEndCapture(s,&g); cudaGraphInstantiate(&ge,g,0,0,0);
  cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
  auto t4=steady_clock::now();
  for(int r=0;r<50;++r) cudaGraphLaunch(ge,s);
  cudaStreamSynchronize(s);
  double gt=us(t4,steady_clock::now())/(50*100);
  auto t5=steady_clock::now();
  for(int r=0;r<50;++r) for(int i=0;i<100;++i) tiny<<<1,32,0,s>>>(o);
  cudaStreamSynchronize(s);
  double stt=us(t5,steady_clock::now())/(50*100);
  printf("graph  100-kernel   : %7.3f us/kernel\n",gt);
  printf("stream 100-kernel   : %7.3f us/kernel (graph %.2fx)\n",stt,stt/gt);
  return 0;
}
