// tma_l1_probe.cu  -- 验证 "TMA G2S 不进 L1" 这个命题 (裸 PTX 版)
//
// [状态] 本文件在 sm_120 (Blackwell) 上手搓 mbarrier PTX 遇到状态空间
//        类型检查问题 (mbarrier.init.shared.b64 要求 .shared 状态地址,
//        inline asm 的裸 u64 寄存器过不了 PTX 类型检查)。
//        请改用配套的 tma_l1_verify.py (CuTe DSL 版, 已验证可 JIT),
//        或在本文件里把 mbarrier 改写成 cuda::barrier C++ API。
//
// 思路 (供参考): 同一程序放两个 kernel, 各自只做一种 global->smem 搬运,
//   用 ncu 分别统计 L1 sector 计数器对比:
//     kernel_ldg : 普通 __ldg 把 global tile 读进 shared  (应 进 L1)
//     kernel_tma : TMA bulk load 把 global tile 搬进 shared (应 绕过 L1)
//
// 结论: kernel_tma 的 l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum 应 ~0,
//       而 kernel_ldg 同一计数器应很大 (64KB tile / 128B = 512 sector).
//
// 推荐改用: tma_l1_verify.py (见该文件内 ncu 命令)
//
// 编译:
//   /usr/local/cuda-13.2/bin/nvcc -arch=sm_120 -O3 tma_l1_probe.cu -o tma_l1_probe
//
// ncu 采集 (分别看两个 kernel):
//   ncu --kernel-name kernel_ldg  --metrics \
//     l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
//     l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
//     lts__t_sectors_srcunit_tex_op_read.sum \
//     ./tma_l1_probe
//
//   ncu --kernel-name kernel_tma  --metrics \
//     l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
//     l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
//     lts__t_sectors_srcunit_tex_op_read.sum \
//     ./tma_l1_probe
//
// 预期: kernel_tma 的 l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum == 0
//       (TMA 走的是 TEX/专用通路且绕过 L1, 不计入 L1 LSU sector),
//       而 lts__t_sectors_srcunit_tex_op_read 仍 >0 (说明数据确实过了 L2).

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// ---------- 裸 TMA 需要 PTX cp.async.bulk.tensor.2d ----------
// 描述符布局 (Hopper/Blackwell 一致): 128 字节
struct __align__(128) TmaDesc {
    uint64_t fill[16];
};

// 用 driver 的 cuTensorMapEncodeTiled 来填描述符最稳, 不手搓位域
#include <cuda.h>

static const int TILE_M = 128;
static const int TILE_N = 128;
static const int BYTES  = TILE_M * TILE_N * sizeof(float); // 64KB

// ============ kernel_ldg: 普通 __ldg 合并读 -> shared ============
__global__ void kernel_ldg(const float* __restrict__ gA, int N, float* sink) {
    __shared__ float smem[TILE_M * TILE_N];
    int base_r = blockIdx.y * TILE_M;
    int base_c = blockIdx.x * TILE_N;
    // 合并读: 每个 warp 沿列方向连续 (row-major A)
    for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
        for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
            int r = base_r + i, c = base_c + j;
            if (r < N && c < N)
                smem[i * TILE_N + j] = __ldg(&gA[r * N + c]);
        }
    }
    __syncthreads();
    // 防止优化掉
    if (threadIdx.x == 0 && threadIdx.y == 0)
        sink[0] = smem[0];
}

// ============ kernel_tma: TMA bulk load G2S ============
// TMA 需要一个 global 描述符 + mbarrier。这里用 1 个线程(warp0 lane0)发指令。
extern "C" __global__ void kernel_tma(const float* gA, int N,
                                      const CUtensorMap* tma_desc,
                                      uint64_t* mbar, float* sink) {
    __shared__ float smem[TILE_M * TILE_N];

    // 每个 block 负责一个 tile
    int block_r = blockIdx.y * TILE_M;
    int block_c = blockIdx.x * TILE_N;

    // mbarrier init (初始到达计数 = 1)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        uint64_t* mb = (uint64_t*)mbar + blockIdx.y * gridDim.x + blockIdx.x;
        asm volatile("mbarrier.init.shared.b64 %0, %1;" :: "l"(__cvta_generic_to_shared(mb)), "r"(1));
        __threadfence_block();
    }
    __syncthreads();

    uint64_t* mb = (uint64_t*)mbar + blockIdx.y * gridDim.x + blockIdx.x;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // 设置 expect tx bytes, 然后发 bulk load
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, %0, %1;"
                     :: "l"(__cvta_generic_to_shared(mb)), "r"(BYTES));

        // cp.async.bulk.tensor.2d: 从 gA 的 (block_r, block_c) 搬 BYTES 字节到 smem
        // 描述符把 gA 描述成 2D (stride=N), 所以 (block_r,block_c) 就是 tile 起点
        void* smem_ptr = (void*)__cvta_generic_to_shared(smem);
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
            " [%0, {%1, %2}], [%3];"
            :: "l"((unsigned long long)__cvta_generic_to_shared(tma_desc)),  // 描述符指针 (必须在 global mem)
              "r"(block_c),   // coord 0 (col)
              "r"(block_r),   // coord 1 (row)
              "l"((unsigned long long)__cvta_generic_to_shared(smem_ptr))  // smem 目的 (shared addr, 64-bit on Blackwell)
            : "memory");
        // 收尾 mbarrier
        asm volatile("mbarrier.arrive.shared.b64 _, %0;" :: "l"(__cvta_generic_to_shared(mb)));
    }
    __syncthreads();
    // 等 bulk load 完成
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        uint32_t done = 0;
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "wait: mbarrier.try_wait.shared.b64 p, %0, %1;\n"
            "@!p bra wait;\n"
            "}"
            :: "l"(__cvta_generic_to_shared(mb)), "r"(done));
    }
    __syncthreads();
    // 防止优化掉
    if (threadIdx.x == 0 && threadIdx.y == 0)
        sink[0] = smem[0];
}

// ---------- host 端构造 TMA 描述符 (放到 device global mem) ----------
static CUtensorMap* make_tma_desc(const float* gA, int N) {
    CUtensorMap desc;
    CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    cuuint64_t dim[] = { (cuuint64_t)N, (cuuint64_t)N };          // [col, row]
    cuuint64_t stride[] = { sizeof(float), (cuuint64_t)N * sizeof(float) }; // col stride=4, row stride=N*4
    cuuint32_t box[] = { TILE_N, TILE_M };                    // box [col,row]
    cuuint32_t elem_stride[] = { 1, 1 };
    CUtensorMapInterleave   interleave   = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle      swizzle      = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion  l2promo      = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill oob          = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    CUresult r = cuTensorMapEncodeTiled(
        &desc, dtype, 2, (void*)gA, dim, stride, box, elem_stride,
        interleave, swizzle, l2promo, oob);
    if (r != CUDA_SUCCESS) { printf("TMA encode failed %d\n", (int)r); exit(1); }
    // 拷到 device global 内存 (TMA 描述符必须驻留 global mem)
    CUtensorMap* d_desc;
    cudaMalloc(&d_desc, sizeof(CUtensorMap));
    cudaMemcpy(d_desc, &desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return d_desc;
}

int main() {
    int N = 1024;
    float* gA;
    cudaMalloc(&gA, N * N * sizeof(float));
    cudaMemset(gA, 1, N * N * sizeof(float));
    float* sink;
    cudaMalloc(&sink, sizeof(float));

    // mbarrier 池
    int nb = (N + TILE_M - 1) / TILE_M;
    uint64_t* mbar;
    cudaMalloc(&mbar, nb * nb * sizeof(uint64_t));

    dim3 block(32, 4, 1);
    dim3 grid(nb, nb, 1);

    // --- 跑 LDG 版 ---
    kernel_ldg<<<grid, block>>>(gA, N, sink);
    cudaDeviceSynchronize();

    // --- 跑 TMA 版 ---
    CUtensorMap* d_tma_desc = make_tma_desc(gA, N);
    kernel_tma<<<grid, block>>>(gA, N, d_tma_desc, mbar, sink);
    cudaDeviceSynchronize();

    printf("done. launch ncu separately on kernel_ldg / kernel_tma\n");
    cudaFree(gA); cudaFree(mbar);
    return 0;
}
