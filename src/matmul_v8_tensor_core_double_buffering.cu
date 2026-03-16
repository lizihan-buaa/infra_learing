#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/pipeline>

// double buffering & async_copy(without using register,g_mem2s_mem directly)
// 异步拷贝，不占用寄存器资源，实现搬运和计算并行完成

using namespace nvcuda;

// 如果每个block有多个warp，这会导致全局内存高重复访问，可以通过smem进行优化
// Block内所有thread协同搬数到smem，然后由warp 加载fragment

// Tensor Core 对 FP16 的标准形状要求通常是 16x16x16
// 假设一个 Block 处理64*64的区域，由4*4个 Warp 组成（每个 Warp 处理16*16）
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define S_M 64
#define S_N 64
#define S_K 64

#define M_GLOBAL 1024
#define N_GLOBAL 512
#define K_GLOBAL 1024

// 使用 managed 内存简化数据传输
__managed__ half a[M_GLOBAL * N_GLOBAL];
__managed__ half b[N_GLOBAL * K_GLOBAL];
__managed__ float c_gpu[M_GLOBAL * K_GLOBAL];
__managed__ float c_cpu[M_GLOBAL * K_GLOBAL];

__global__ void gpu_matmul_fp16(const half *a, const half *b, float *c, int m, int n, int k)
{
    // 1. 声明双缓冲 Shared Memory (2层 stage)
    // 增加一层维度 [2] 用于切换 buffer
    __shared__ alignas(16) half sub_a[2][S_M][S_N]; // 内存对16对齐（首地址是16的整数倍），使得可以进行向量化访问（可能牺牲空间，换取时间性能提升）
    __shared__ alignas(16) half sub_b[2][S_N][S_K]; // 16字节 = 128位 GPU 的单条访存指令最大就是 128 位（16 字节）

    int warpId = threadIdx.x / 32;
    int warpM = (warpId / 4) * WMMA_M; 
    int warpK = (warpId % 4) * WMMA_K; 

    int blockM = blockIdx.y * S_M;
    int blockK = blockIdx.x * S_K;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // 管理异步内存拷贝 (memcpy_async) 与 计算任务 (Tensor Core) 之间的先后顺序，从而实现“边搬运、边计算”
    auto pipe = cuda::make_pipeline();

    // --- 预加载第 0 块数据 ---
    pipe.producer_acquire(); // 生产者阶段1：申请当前stage的空间
    for (int t = threadIdx.x; t < S_M * S_N; t += blockDim.x)
    {
        int r = t / S_N;
        int c_idx = t % S_N;
        cuda::memcpy_async(&sub_a[0][r][c_idx], &a[(blockM + r) * n + c_idx], sizeof(half), pipe); // 生产者阶段2：派发异步搬运任务，绑定到该 pipe
    }
    for (int t = threadIdx.x; t < S_N * S_K; t += blockDim.x)
    {
        int r = t / S_K;
        int c_idx = t % S_K;
        cuda::memcpy_async(&sub_b[0][r][c_idx], &b[r * k + (blockK + c_idx)], sizeof(half), pipe);
    }
    pipe.producer_commit(); // 生产者阶段3：提交任务：告诉系统这批搬运已经进队列了

    // --- 主循环 ---
    int stage = 0;
    for (int i = 0; i < n; i += S_N)
    {
        // 计算下一个 stage 的索引
        int next_stage = 1 - stage;
        int next_i = i + S_N;

        // 1. 提交下一块数据的异步拷贝请求 (Prologue for next tile)
        if (next_i < n)
        {
            pipe.producer_acquire();
            for (int t = threadIdx.x; t < S_M * S_N; t += blockDim.x)
            {
                int r = t / S_N;
                int c_idx = t % S_N;
                cuda::memcpy_async(&sub_a[next_stage][r][c_idx], &a[(blockM + r) * n + (next_i + c_idx)], sizeof(half), pipe);
            }
            for (int t = threadIdx.x; t < S_N * S_K; t += blockDim.x)
            {
                int r = t / S_K;
                int c_idx = t % S_K;
                cuda::memcpy_async(&sub_b[next_stage][r][c_idx], &b[(next_i + r) * k + (blockK + c_idx)], sizeof(half), pipe);
            }
            pipe.producer_commit();
        }

        // 2. 等待当前 stage 的数据搬运完成
        pipe.consumer_wait(); // 消费者阶段1：挡住计算逻辑：确认当前 Stage 的数据搬完了吗？
        __syncthreads();

        // 3. 计算当前 stage 的数据 (Compute)
        for (int j = 0; j < S_N; j += WMMA_N)
        {
            wmma::load_matrix_sync(a_frag, (half*)&sub_a[stage][warpM][j], S_N);
            wmma::load_matrix_sync(b_frag, (half*)&sub_b[stage][j][warpK], S_K);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        // 4. 释放当前 stage 的空间
        __syncthreads();
        pipe.consumer_release(); // 消费者阶段2：释放空间，告诉系统这块内存可以给下一轮搬运用了

        // 切换 buffer 索引
        stage = next_stage;
    }

    // --- 写回结果 ---
    int g_m = blockM + warpM;
    int g_k = blockK + warpK;
    if (g_m < m && g_k < k)
    {
        wmma::store_matrix_sync(c + g_m * k + g_k, acc_frag, k, wmma::mem_row_major);
    }
}

void cpu_matmul(const half *a, const half *b, float *c, int m, int n, int k)
{
    for(int y = 0; y < m; y++)
    {
        for(int x = 0; x < k; x++)
        {
            float tmp = 0.0f;
            for(int step = 0; step < n; step++)
            {
                // 将 half 转为 float 计算
                tmp += __half2float(a[y * n + step]) * __half2float(b[step * k + x]);
            }
            c[y * k + x] = tmp;
        }
    }
}

int main()
{
    // 初始化数据
    for(int i = 0; i < M_GLOBAL * N_GLOBAL; i++)
    {
        a[i] = __float2half((float)(rand() % 10) / 10.0f);
    }

    for(int i = 0; i < N_GLOBAL * K_GLOBAL; i++)
    {
        b[i] = __float2half((float)(rand() % 10) / 10.0f);
    }
        
    // 每个 block 配置 4 * 4 个 warp (512 threads)
    dim3 dimBlock(512, 1);
    // 每个 block 处理 4 个 16x16 tile (纵向排列)
    dim3 dimGrid((K_GLOBAL / WMMA_K) / 4, (M_GLOBAL / WMMA_M) / 4);

    gpu_matmul_fp16<<<dimGrid, dimBlock>>>(a, b, c_gpu, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    cudaDeviceSynchronize();

    // CPU 验证
    cpu_matmul(a, b, c_cpu, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    // 浮点数验证（使用较小的阈值，因为 half 存在精度损失）
    bool errors = false;
    for(int i = 0; i < M_GLOBAL * K_GLOBAL; i++)
    {
        if(abs(c_cpu[i] - c_gpu[i]) > 0.1f) 
        {
            errors = true;
            break;
        }
    }

    printf("Result: %s\n", errors ? "Failed" : "Passed");

    return 0;
}