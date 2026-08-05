#define CYC(NM,T,B) __global__ void NM(int* o,long long* t,int it){\
 T x0=threadIdx.x+1,x1=x0+1,x2=x0+2,x3=x0+3;\
 asm volatile("":::"memory"); long long a=clock64();\
 for(int i=0;i<it;++i){B}\
 long long b=clock64(); asm volatile("":::"memory");\
 T s=x0+x1+x2+x3; o[threadIdx.x]=(int)s;\
 if(threadIdx.x==1)*(T*)(o+2048)=s;\
 if(threadIdx.x==0)*t=b-a;}
CYC(kc1,int,x0+=x1;x1+=x2;x2+=x3;x3+=x0;)
CYC(kc2,int,x0=x0*x1+x2;x1=x1*x2+x3;x2=x2*x3+x0;x3=x3*x0+x1;)
CYC(kc3,int,x0*=x1;x1*=x2;x2*=x3;x3*=x0;)
CYC(kc4,long long,x0+=x1;x1+=x2;x2+=x3;x3+=x0;)
CYC(kc5,long long,x0*=x1;x1*=x2;x2*=x3;x3*=x0;)
CYC(kc6,double,x0=x0*x1+x2;x1=x1*x2+x3;x2=x2*x3+x0;x3=x3*x0+x1;)
CYC(kc7,float,x0=x0*x1+x2;x1=x1*x2+x3;x2=x2*x3+x0;x3=x3*x0+x1;)
#include <cuda_fp16.h>
__global__ void kc8(int* o,long long* t,int it){
  __half2 x0=__floats2half2_rn(1.f,2.f),x1=__floats2half2_rn(1.001f,1.002f);
  __half2 x2=x0,x3=x1;
  asm volatile("":::"memory"); long long a=clock64();
  for(int i=0;i<it;++i){x0=__hfma2(x0,x1,x2);x1=__hfma2(x1,x2,x3);x2=__hfma2(x2,x3,x0);x3=__hfma2(x3,x0,x1);}
  long long b=clock64(); asm volatile("":::"memory");
  __half2 s=__hadd2(__hadd2(x0,x1),__hadd2(x2,x3));
  o[threadIdx.x]=(int)__low2float(s); if(threadIdx.x==0)*t=b-a;
}
