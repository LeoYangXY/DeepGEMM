#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t e = (call);                                        \
        if (e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(e));                   \
            exit(1);                                                   \
        }                                                              \
    } while (0)

// 一个故意慢一点的 kernel，方便在 nsys 里看清楚
__global__ void slow_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        // 多做几轮运算，让 kernel 时间长一些，好观察重叠
        for (int i = 0; i < 200; i++) {
            val = sinf(val) * cosf(val) + 1.0f;
        }
        data[idx] = val;
    }
}

//=============================================================
// 实验1：单 stream，全部串行
//=============================================================
void test_single_stream(float *h_data, float *d_data, int N, size_t bytes) {
    // H2D
    CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    // Compute
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    slow_kernel<<<blocks, threads>>>(d_data, N);
    // D2H
    CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
}

//=============================================================
// 实验2：多 stream + pinned memory，可以真正重叠
//=============================================================
void test_multi_stream_pinned(float *h_pinned, float *d_data, int N,
                               size_t bytes, int num_streams) {
    cudaStream_t *streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++)
        CHECK(cudaStreamCreate(&streams[i]));

    int chunk = N / num_streams;
    size_t chunk_bytes = chunk * sizeof(float);
    int threads = 256;
    int blocks = (chunk + threads - 1) / threads;

    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk;
        CHECK(cudaMemcpyAsync(d_data + offset, h_pinned + offset,
                              chunk_bytes, cudaMemcpyHostToDevice, streams[i]));
        slow_kernel<<<blocks, threads, 0, streams[i]>>>(d_data + offset, chunk);
        CHECK(cudaMemcpyAsync(h_pinned + offset, d_data + offset,
                              chunk_bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < num_streams; i++)
        CHECK(cudaStreamDestroy(streams[i]));
    delete[] streams;
}

//=============================================================
// 实验3：多 stream + pageable memory（async 其实是假的）
//=============================================================
void test_multi_stream_pageable(float *h_data, float *d_data, int N,
                                 size_t bytes, int num_streams) {
    cudaStream_t *streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++)
        CHECK(cudaStreamCreate(&streams[i]));

    int chunk = N / num_streams;
    size_t chunk_bytes = chunk * sizeof(float);
    int threads = 256;
    int blocks = (chunk + threads - 1) / threads;

    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk;
        // 用 pageable memory 调 Async —— 驱动会偷偷同步
        CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                              chunk_bytes, cudaMemcpyHostToDevice, streams[i]));
        slow_kernel<<<blocks, threads, 0, streams[i]>>>(d_data + offset, chunk);
        CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                              chunk_bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < num_streams; i++)
        CHECK(cudaStreamDestroy(streams[i]));
    delete[] streams;
}

int main() {
    const int N = 1 << 22;  // 4M elements, ~16MB
    const size_t bytes = N * sizeof(float);
    const int NUM_STREAMS = 4;

    printf("N = %d (%.1f MB), streams = %d\n", N, bytes / 1e6, NUM_STREAMS);

    //--- 分配 pageable memory ---
    float *h_pageable = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_pageable[i] = 1.0f;

    //--- 分配 pinned memory ---
    float *h_pinned;
    CHECK(cudaMallocHost(&h_pinned, bytes));
    for (int i = 0; i < N; i++) h_pinned[i] = 1.0f;

    //--- 分配 device memory ---
    float *d_data;
    CHECK(cudaMalloc(&d_data, bytes));

    // warmup
    CHECK(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    //--- NVTX range markers 用 CUDA event 计时 ---
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float ms;

    // 实验1：单 stream 串行
    printf("\n=== Test 1: Single Stream (serial) ===\n");
    CHECK(cudaEventRecord(start));
    test_single_stream(h_pageable, d_data, N, bytes);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.2f ms\n", ms);

    // 实验2：多 stream + pinned
    printf("\n=== Test 2: %d Streams + Pinned Memory ===\n", NUM_STREAMS);
    CHECK(cudaEventRecord(start));
    test_multi_stream_pinned(h_pinned, d_data, N, bytes, NUM_STREAMS);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.2f ms\n", ms);

    // 实验3：多 stream + pageable
    printf("\n=== Test 3: %d Streams + Pageable Memory (fake async) ===\n", NUM_STREAMS);
    CHECK(cudaEventRecord(start));
    test_multi_stream_pageable(h_pageable, d_data, N, bytes, NUM_STREAMS);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.2f ms\n", ms);

    // 清理
    CHECK(cudaFree(d_data));
    CHECK(cudaFreeHost(h_pinned));
    free(h_pageable);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    printf("\n--- Done! ---\n");
    printf("在 nsys 报告里你会看到:\n");
    printf("  Test1: H2D / Kernel / D2H 完全串行，没有重叠\n");
    printf("  Test2: 不同 stream 的传输和计算有重叠（流水线效果）\n");
    printf("  Test3: 虽然用了多 stream，但 pageable 内存让 async 退化为同步\n");
    return 0;
}
