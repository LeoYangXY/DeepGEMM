// tmp demo: MoE combine_recv stage - volatile vs non-volatile flag polling
//
// Scenario:
//   A token has 8 topk sources. Each source has a "ready" flag (in L3 / cross-rank).
//   The combine_recv kernel polls all 8 flags until they are all set (arrived),
//   then does the combine: sum over k of (value_k * weight_k).
//
//   We compare two flag declarations:
//     - non-volatile int  (compiler may hoist the load out of the poll loop)
//     - volatile int      (every poll re-reads memory, sees cross-rank updates)
//
//   A separate "producer" kernel sets the flags after a delay to simulate
//   async arrival (cross-card / L3 latency).
//
// compile: nvcc -O3 -arch=sm_120 -o moe_combine_volatile_demo moe_combine_volatile_demo.cu
// run:     ./moe_combine_volatile_demo

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

static const int NTOPK = 8;     // topk sources per token
static const int NTOK  = 256;   // number of tokens
static const int MAX_POLL = 1000000;  // safety cap so a hoisted-load loop can't truly hang

// ---------- producer kernel: set flags for a token, with a delay ----------
// delay_cycles: spin on GPU clock before setting flags, to simulate arrival latency
__global__ void producer_set_flags(int* flags, int token, int delay_cycles) {
    if (blockIdx.x != 0 && threadIdx.x != 0) return;  // single thread does it
    uint64_t start = clock64();
    while (clock64() - start < (uint64_t)delay_cycles) { /* busy wait */ }
    // set all 8 flags for this token
    for (int k = 0; k < NTOPK; ++k) flags[token * NTOPK + k] = 1;
}

// ---------- combine_recv: NON-VOLATILE flags ----------
__global__ void combine_recv_nonvolatile(const int* flags, const float* values,
                                          const float* weights, float* out,
                                          int token, uint64_t* ts) {
    // single thread handles the whole token (matches "query 8 flags then combine")
    if (threadIdx.x != 0) return;

    uint64_t t0 = clock64();
    asm volatile("" ::: "memory");

    // Phase A: poll until all 8 flags are set (non-volatile)
    int all_ready = 0;
    int iter = 0;
    while (!all_ready && iter < MAX_POLL) {
        all_ready = 1;
        for (int k = 0; k < NTOPK; ++k) {
            if (flags[token * NTOPK + k] == 0) { all_ready = 0; break; }
        }
        ++iter;
    }
    uint64_t t1 = clock64();
    asm volatile("" ::: "memory");

    // Phase B: combine = sum_k value_k * weight_k
    float acc = 0.f;
    for (int k = 0; k < NTOPK; ++k) {
        acc += values[token * NTOPK + k] * weights[token * NTOPK + k];
    }
    out[token] = acc;
    uint64_t t2 = clock64();

    ts[0] = t0; ts[1] = t1; ts[2] = t2;
    ts[3] = (uint64_t)iter;  // record iterations (debug signal)
}

// ---------- combine_recv: VOLATILE flags ----------
__global__ void combine_recv_volatile(const volatile int* flags, const float* values,
                                       const float* weights, float* out,
                                       int token, uint64_t* ts) {
    if (threadIdx.x != 0) return;

    uint64_t t0 = clock64();
    asm volatile("" ::: "memory");

    // Phase A: poll (volatile - every read hits memory)
    int all_ready = 0;
    int iter = 0;
    while (!all_ready && iter < MAX_POLL) {
        all_ready = 1;
        for (int k = 0; k < NTOPK; ++k) {
            if (flags[token * NTOPK + k] == 0) { all_ready = 0; break; }
        }
        ++iter;
    }
    uint64_t t1 = clock64();
    asm volatile("" ::: "memory");

    // Phase B: combine
    float acc = 0.f;
    for (int k = 0; k < NTOPK; ++k) {
        acc += values[token * NTOPK + k] * weights[token * NTOPK + k];
    }
    out[token] = acc;
    uint64_t t2 = clock64();

    ts[0] = t0; ts[1] = t1; ts[2] = t2;
    ts[3] = (uint64_t)iter;
}

int main() {
    const int nFlag = NTOK * NTOPK;
    const int nVal  = NTOK * NTOPK;

    int* dFlags; float *dVals, *dW, *dOut; uint64_t* dTs;
    cudaMalloc(&dFlags, nFlag * sizeof(int));
    cudaMalloc(&dVals,  nVal  * sizeof(float));
    cudaMalloc(&dW,     nVal  * sizeof(float));
    cudaMalloc(&dOut,   NTOK  * sizeof(float));
    cudaMalloc(&dTs,    4 * sizeof(uint64_t));

    float *hV = new float[nVal];
    float *hW = new float[nVal];
    for (int i = 0; i < nVal; ++i) { hV[i] = 1.0f; hW[i] = 0.125f; }
    cudaMemcpy(dVals, hV, nVal * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW,    hW, nVal * sizeof(float), cudaMemcpyHostToDevice);

    // clock rate
    int clkKHz = 0;
    cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);
    double ghz = clkKHz / 1e6;

    int token = 0;
    // producer delay: simulate cross-rank arrival latency (in GPU cycles)
    // We want consumer to START polling BEFORE flags arrive, so they must run
    // concurrently. cycles = seconds * Hz.
    int delay_us = 20;
    int delay_cyc = (int)(delay_us * 1e-6 * ghz * 1e9); // = delay_us * ghz * 1000

    auto run_once = [&](bool use_volatile) -> uint64_t* {
        // reset flags to 0
        cudaMemset(dFlags, 0, nFlag * sizeof(int));
        cudaMemset(dOut, 0, NTOK * sizeof(float));
        cudaDeviceSynchronize();

        cudaStream_t sProd, sCons;
        cudaStreamCreate(&sProd);
        cudaStreamCreate(&sCons);

        // consumer starts FIRST (begins polling immediately while flags are still 0)
        if (use_volatile)
            combine_recv_volatile<<<1, 1, 0, sCons>>>((const volatile int*)dFlags, dVals, dW, dOut, token, dTs);
        else
            combine_recv_nonvolatile<<<1, 1, 0, sCons>>>(dFlags, dVals, dW, dOut, token, dTs);

        // producer sets flags after a delay, on a different stream (concurrent)
        producer_set_flags<<<1, 1, 0, sProd>>>(dFlags, token, delay_cyc);

        cudaStreamSynchronize(sCons);
        cudaStreamDestroy(sProd);
        cudaStreamDestroy(sCons);

        uint64_t* h = new uint64_t[4];
        cudaMemcpy(h, dTs, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        return h;
    };

    printf("=== MoE combine_recv: volatile vs non-volatile flag polling ===\n");
    printf("GPU clock: %.3f GHz  token=%d  topk=%d  producer_delay=~%d us\n\n",
           ghz, token, NTOPK, delay_us);

    uint64_t* nv = run_once(false);
    uint64_t* vol = run_once(true);

    auto cyc2us = [ghz](unsigned long long c){ return c / ghz / 1000.0; };

    printf("%-22s %14s %14s %14s %12s\n", "version", "poll(cyc)", "poll(us)", "combine(us)", "poll_iters");
    printf("%-22s %14llu %14.3f %14.3f %12llu\n", "non-volatile",
           (unsigned long long)(nv[1]-nv[0]), cyc2us(nv[1]-nv[0]), cyc2us(nv[2]-nv[1]),
           (unsigned long long)nv[3]);
    printf("%-22s %14llu %14.3f %14.3f %12llu\n", "volatile",
           (unsigned long long)(vol[1]-vol[0]), cyc2us(vol[1]-vol[0]), cyc2us(vol[2]-vol[1]),
           (unsigned long long)vol[3]);

    // correctness check on output
    float check; cudaMemcpy(&check, dOut + token, sizeof(float), cudaMemcpyDeviceToHost);
    float expected = 0.f; for (int k=0;k<NTOPK;++k) expected += 1.0f*0.125f;
    printf("\nout[token]=%.4f (expected %.4f, sum of 8 * 0.125)\n", check, expected);

    // analyze
    if (nv[3] >= (uint64_t)MAX_POLL) {
        printf("\n[KEY FINDING] non-volatile poll HIT THE ITER CAP (%d): the compiler hoisted\n"
               "  the flag load out of the loop, so the consumer NEVER saw the producer's update\n"
               "  and spun forever. This is a CORRECTNESS BUG, not just a perf diff.\n"
               "  -> For cross-rank / cross-thread flags you MUST use volatile (or atomic/acquire).\n",
               MAX_POLL);
    } else if (nv[1]-nv[0] <= 2) {
        printf("\n[WARN] non-volatile poll took ~0 cycles -> compiler likely hoisted the flag\n"
               "  load; correctness risk for cross-rank flags. Use volatile.\n");
    } else {
        printf("\n[OK] both actually polled. volatile poll=%.3fus vs non-volatile=%.3fus\n"
               "  (volatile re-reads memory every time => more L3/mem traffic, slightly slower).\n",
               cyc2us(vol[1]-vol[0]), cyc2us(nv[1]-nv[0]));
    }

    delete[] hV; delete[] hW; delete[] nv; delete[] vol;
    cudaFree(dFlags); cudaFree(dVals); cudaFree(dW); cudaFree(dOut); cudaFree(dTs);
    return 0;
}