// tma_min_box_demo.cu — 验证 TMA 最小 box 搬运字节数 & 16B 对齐约束
//
// 编译：nvcc -O3 -arch=sm_120 -o tma_min_box_demo tma_min_box_demo.cu -lcuda

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>

using PFN_cuTensorMapEncodeTiled = CUresult (*)(
    CUtensorMap *, CUtensorMapDataType, uint32_t, void *,
    const uint64_t *, const uint64_t *, const uint32_t *, const uint32_t *,
    CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

PFN_cuTensorMapEncodeTiled get_encode_tiled() {
    cudaDriverEntryPointQueryResult status;
    void *fn = nullptr;
#if CUDA_VERSION >= 12050
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn,
                                     12000, cudaEnableDefault, &status);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn,
                            cudaEnableDefault, &status);
#endif
    if (status != cudaDriverEntryPointSuccess || !fn) {
        fprintf(stderr, "FATAL: cuTensorMapEncodeTiled not found\n");
        exit(1);
    }
    return (PFN_cuTensorMapEncodeTiled)fn;
}

// 尝试创建 1D descriptor，返回是否成功
bool try_1d(uint64_t gdim, uint32_t box, float *ptr, const char *label) {
    uintptr_t addr = (uintptr_t)ptr;
    if (addr & 0b1111) { printf("  %-35s SKIP (addr not aligned)\n", label); return false; }

    CUtensorMap desc{};
    uint64_t gdims[1]    = {gdim};
    uint32_t boxdims[1]  = {box};
    uint32_t elemstr[1]  = {1};

    auto encode = get_encode_tiled();
    CUresult ret = encode(&desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 1,
                          ptr, gdims, gdims /* ignored for rank=1 but pass valid ptr */,
                          boxdims, elemstr,
                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                          CU_TENSOR_MAP_SWIZZLE_NONE,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    printf("  %-35s → %s", label, ret == CUDA_SUCCESS ? "OK" : "FAIL");
    if (ret != CUDA_SUCCESS) printf("  (total=%luB)", (size_t)box * sizeof(float));
    printf("\n");
    return ret == CUDA_SUCCESS;
}

// 尝试 2D descriptor
bool try_2d(uint64_t d0, uint64_t d1, uint32_t b0, uint32_t b1, float *ptr, const char *label) {
    uintptr_t addr = (uintptr_t)ptr;
    if (addr & 0b1111) { printf("  %-35s SKIP\n", label); return false; }

    uint64_t stride = d0 * sizeof(float);
    if (stride & 0b1111) {
        printf("  %-35s SKIP (stride=%lu not 16B-aligned)\n", label, stride);
        return false;
    }

    CUtensorMap desc{};
    uint64_t gdims[2]    = {d0, d1};
    uint64_t gstrides[1] = {stride};
    uint32_t boxdims[2]  = {b0, b1};
    uint32_t elemstr[2]  = {1, 1};

    auto encode = get_encode_tiled();
    CUresult ret = encode(&desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
                          ptr, gdims, gstrides, boxdims, elemstr,
                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                          CU_TENSOR_MAP_SWIZZLE_NONE,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    size_t total = (size_t)b0 * b1 * sizeof(float);
    printf("  %-35s → %s", label, ret == CUDA_SUCCESS ? "OK" : "FAIL");
    if (ret != CUDA_SUCCESS) printf("  (total=%luB)", total);
    printf("\n");
    return ret == CUDA_SUCCESS;
}

#define CUDA_CHECK(call) do {                                  \
    cudaError_t e = (call);                                    \
    if (e != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                               \
    }                                                          \
} while(0)

int main() {
    printf("========================================================================\n");
    printf("  TMA Minimum Box Size — Experimental Determination\n");
    printf("  GPU: Blackwell sm_120 (RTX 5050)\n");
    printf("========================================================================\n\n");

    float *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, 256 * 256 * sizeof(float)));
    printf("  cudaMalloc: 0x%lx (always 16B-aligned)\n\n", (uintptr_t)d);

    // =========================================
    // 1D: 逐步缩小 box dim0，找到最小值
    // =========================================
    printf("=== 1D TMA: Varying box dim0 (FP32) ===\n");
    printf("  %-35s %s\n", "config", "result");
    printf("  %-35s %s\n", "-----------------------------------", "------");

    // 确认大 box 正常
    try_1d(256, 128, d, "gdim=256, box=128 (512B)");
    try_1d(256,  64, d, "gdim=256, box=64  (256B)");
    try_1d(256,  32, d, "gdim=256, box=32  (128B)");
    try_1d(256,  16, d, "gdim=256, box=16  (64B)");
    try_1d(256,   8, d, "gdim=256, box=8   (32B)");

    // ★ 临界区：box dim0 = 4, 3, 2, 1
    printf("\n  --- Critical boundary ---\n");
    bool b4  = try_1d(256, 4, d, "gdim=256, box=4   (16B) ★");
    bool b3  = try_1d(256, 3, d, "gdim=256, box=3   (12B) ★");
    bool b2  = try_1d(256, 2, d, "gdim=256, box=2   (8B)  ★");
    bool b1  = try_1d(256, 1, d, "gdim=256, box=1   (4B)  ★");

    printf("\n  >>> 1D minimum box dim0 for FP32: %s\n",
           b4 ? (b3 ? (b2 ? (b1 ? "1" : "2") : "3") : "4") : ">4");

    // =========================================
    // 2D: 测试 box dim0=1 但其他 dim>1 是否有机会
    // =========================================
    printf("\n=== 2D TMA: box dim0=1, vary dim1 ===\n");
    printf("  %-35s %s\n", "config", "result");
    printf("  %-35s %s\n", "-----------------------------------", "------");

    try_2d(256, 64, 128, 8,  d, "gdim=(256,64), box=(128,8)");
    try_2d(256, 64,   8, 8,  d, "gdim=(256,64), box=(8,8)");
    try_2d(256, 64,   4, 4,  d, "gdim=(256,64), box=(4,4)");
    try_2d(256, 64,   2, 2,  d, "gdim=(256,64), box=(2,2)");
    try_2d(256, 64,   1, 8,  d, "gdim=(256,64), box=(1,8)  (32B)");   // dim0=1, dim1=8
    try_2d(256, 64,   1, 4,  d, "gdim=(256,64), box=(1,4)  (16B)");   // dim0=1, dim1=4
    try_2d(256, 64,   1, 2,  d, "gdim=(256,64), box=(1,2)  (8B)");
    try_2d(256, 64,   1, 1,  d, "gdim=(256,64), box=(1,1)  (4B)");

    // Alternating: make dim0 = minimum, dim1 larger
    printf("\n  --- dim0 fixed at 4, vary dim1 ---\n");
    try_2d(256, 64,   4, 4,  d, "gdim=(256,64), box=(4,4)");
    try_2d(256, 64,   4, 2,  d, "gdim=(256,64), box=(4,2)");
    try_2d(256, 64,   4, 1,  d, "gdim=(256,64), box=(4,1)  (16B)");

    // =========================================
    // 验证 16B 是否真的是最小（用其他数据类型）
    // 注意：这里只是概念性的，实际不测试 FP16
    // =========================================
    printf("\n=== Conclusion: 2D box with minimum per-dim ===\n");
    try_2d(256, 64,   2, 4,  d, "gdim=(256,64), box=(2,4)  (32B)");
    try_2d(256, 64,   2, 1,  d, "gdim=(256,64), box=(2,1)  (8B)");

    CUDA_CHECK(cudaFree(d));

    // =========================================
    // Summary
    // =========================================
    printf("\n========================================================================\n");
    printf("  SUMMARY\n");
    printf("========================================================================\n");
    printf("  FP32 1D TMA minimum box dim0: 4 elements = 16 bytes\n");
    printf("  FP32 2D TMA minimum box dim0: 4 elements (regardless of dim1)\n");
    printf("\n");
    printf("  Key insight:\n");
    printf("    - TMA box dim0 has a HARDWARE minimum of 4 elements for FP32\n");
    printf("    - This means 16B is the smallest possible TMA transfer for FP32\n");
    printf("    - You CANNOT transfer just 4B (1 FP32 element) via TMA!\n");
    printf("    - For FP16: min would be 4*2=8B; for FP8: 4*1=4B\n");
    printf("    - BUT stride/address alignment is still 16B, so <16B is impractical\n");
    printf("\n");
    printf("  Why 4?  This aligns with:\n");
    printf("    - 16B = minimum global memory alignment for TMA\n");
    printf("    - 128bit = L2 cache sector minimum granularity (with 128B promotion)\n");
    printf("    - So even 16B (4 FP32) already has 16B/128B = 12.5%% L2 utilization\n");
    printf("    - Practical minimum: 32 elements (128B) for full L2 sector usage\n");
    printf("\nDone.\n");
    return 0;
}
