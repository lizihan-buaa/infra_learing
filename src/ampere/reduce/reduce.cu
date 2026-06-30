#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// V0: 朴素的树形规约（步长从小到大）
__global__ void reduce_v0(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存数据加载到共享内存
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 树形规约：步长从 1 开始逐步翻倍
    for (int step = 1; step < blockDim.x; step *= 2) {
        if (tid % (2 * step) == 0) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    // 每个 Block 的结果写回全局内存
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
// Warp Divergence（Warp 分化）：
// GPU 以 Warp（32 个线程）为调度单位，Warp 内所有线程必须执行相同的指令。
// 当 Warp 内部分线程满足 if 条件、部分不满足时，GPU 会分两次执行（先执行满足条件的线程，再执行不满足的），实际吞吐减半。

// V1: strided index 方式，减少 Warp Divergence
__global__ void reduce_v1(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 步长从 1 开始逐步翻倍，但用 strided index 映射活跃线程
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = threadIdx.x * 2 * s;
        if (index < blockDim.x) {
            smem[index] += smem[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
// 前几轮完全消除分化，仅最后几轮（工作线程很少时）存在 Warp 内分化。

// 第 1 轮（s=1）：tid 0 访问 smem[0]、smem[1]；tid 16 访问 smem[32]、smem[33]。smem[0] 和 smem[32] 都落在 Bank 0 → 2 路 Bank Conflict
// 第 2 轮（s=2）：tid 0 访问 smem[0,2]；tid 8 访问 smem[32,34]；tid 16 访问 smem[64,66]；tid 24 访问 smem[96,98]。smem[0]、smem[32]、smem[64]、smem[96] 都在 Bank 0 → 4 路 Bank Conflict
// 第 3 轮（s=4）：8 路 Bank Conflict

// V2: 步长从大到小，同时消除 Warp Divergence 与 Bank Conflict
__global__ void reduce_v2(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 步长从 blockDim.x/2 开始，每轮减半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
// 第 1 轮（step=128）：tid 0 访问 smem[0, 128]，tid 1 访问 smem[1, 129]，…，tid 31 访问 smem[31, 159]。这 32 个线程分别访问 Bank 0~31 的不同地址 → 无 Bank Conflict
// 第 2 轮（step=64）、第 3 轮（step=32） 同理，32 个线程刚好覆盖 32 个 Bank

// V3: 每线程处理 2 个元素，减少空闲线程
__global__ void reduce_v3(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每个线程加载 2 个相距 blockDim.x 的元素并求和
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 步长从大到小的规约（同 V2）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}