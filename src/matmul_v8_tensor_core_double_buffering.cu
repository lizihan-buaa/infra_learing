#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda/pipeline>

// double buffering & async_copy(without using register,g_mem2s_mem directly)
// 异步拷贝，不占用寄存器资源，实现搬运和计算并行完成
using namespace nvcuda;

// 如果每个block有多个warp，这会导致全局内存高重复访问，可以通过smem进行优化
// Block内所有thread协同搬数到smem，然后由warp 加载fragment

// Tensor Core 对 FP16 的标准形状要求通常是 16x16x16
// 假设一个 Block 处理64*64的区域，由4*4个 Warp 组成（每个 Warp 处理16*16）

#define MMA_M 16
#define MMA_N 16
#define MMA_K 16
// 每个block有4个warp，处理64*64的区域
// blockDim(512, 1)，16个warp一维排开
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64

#define M 1024
#define N 512
#define K 1024

__managed__ half a[M * N];
__managed__ half b[N * K];
__managed__ float c_gpu[M * K];
__managed__ float c_cpu[M * K];

__global__ void gpu_matmul_fp16(const half *a, const half *b, float *c, int m, int n, int k)
{
    __shared__ alignas(32) half sub_a[2][BLOCK_M][BLOCK_N + 8];
    __shared__ alignas(32) half sub_b[2][BLOCK_N][BLOCK_K + 8];

    // 计算当前thread所在的warp所处理的16*16块在block中的位置
    int warpId = threadIdx.x / 32;
    int warpM = (warpId / 4) * MMA_M;
    int warpK = (warpId % 4) * MMA_K;

    // 计算当前thread所在的block所处理的64*64块在c矩阵中的位置
    int blockM = blockIdx.y * BLOCK_M;
    int blockK = blockIdx.x * BLOCK_K;

    // 申请tensor_core空间
    wmma::fragment<wmma::matrix_a, MMA_M, MMA_K, MMA_N, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, MMA_M, MMA_K, MMA_N, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, MMA_M, MMA_K, MMA_N, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // 双缓冲管理
    auto pipe = cuda::make_pipeline();
    
    // 经过并行思路思考，需要每个thread做循环的是寄存器分块部分
    // ping-0，a,b地址的计算较简单
    pipe.producer_acquire();
    for(int t=threadIdx.x; t<(BLOCK_M*BLOCK_N); t+=blockDim.x)
    {
        int row = t / BLOCK_N;
        int col = t % BLOCK_N;
        cuda::memcpy_async(&sub_a[0][row][col], &a[(blockM+row)*n+col], sizeof(half), pipe); 
    }
    for(int t=threadIdx.x; t<(BLOCK_N*BLOCK_K); t+=blockDim.x)
    {
        int row = t / BLOCK_K;
        int col = t % BLOCK_K;
        cuda::memcpy_async(&sub_b[0][row][col], &b[row*k+(blockK+col)], sizeof(half), pipe);
    }
    pipe.producer_commit(); // 计算核心不停，搬运由专用硬件完成

    // 从pong0开始的主循环（包括ping0的计算部分）
    int stage = 0;
    for(int i=0; i<n; i+=BLOCK_N)
    {
        int next_stage = 1 - stage;
        int next_i = i + BLOCK_N;
        // 提交pong0的load请求
        if(next_i < n)
        {
            pipe.producer_acquire();
            for(int t=threadIdx.x; t<(BLOCK_M*BLOCK_N); t+=blockDim.x)
            {
                int row = t / BLOCK_N;
                int col = t % BLOCK_N;
                cuda::memcpy_async(&sub_a[next_stage][row][col], &a[(blockM+row)*n+(col+next_i)], sizeof(half), pipe);
            }
            for(int t=threadIdx.x; t<(BLOCK_N*BLOCK_K); t+=blockDim.x)
            {
                int row = t / BLOCK_K;
                int col = t % BLOCK_K;
                cuda::memcpy_async(&sub_b[next_stage][row][col], &b[(next_i+row)*k+(blockK+col)], sizeof(half), pipe);
            }
            pipe.producer_commit();
        }
        pipe.consumer_wait();
        __syncthreads();

        for(int j=0; j<BLOCK_N; j+=MMA_N)
        {
            wmma::load_matrix_sync(a_frag, (half*)&sub_a[stage][warpM][j], BLOCK_N + 8);
            wmma::load_matrix_sync(b_frag, (half*)&sub_b[stage][j][warpK], BLOCK_K + 8);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
        pipe.consumer_release();

        stage = next_stage;
    }
    int g_m = blockM + warpM;
    int g_k = blockK + warpK;
    if(g_m < m && g_k < k)
    {
        wmma::store_matrix_sync(c+g_m*k+g_k, acc_frag, k, wmma::mem_row_major);
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
    for(int i = 0; i < M * N; i++)
    {
        a[i] = __float2half((float)(rand() % 10) / 10.0f);
    }

    for(int i = 0; i < N * K; i++)
    {
        b[i] = __float2half((float)(rand() % 10) / 10.0f);
    }
        
    // 每个 block 配置 4 * 4 个 warp (512 threads)
    dim3 dimBlock(512, 1);
    // 每个 block 处理 4 个 16x16 tile (纵向排列)
    dim3 dimGrid((K / MMA_K) / 4, (K / MMA_M) / 4);

    gpu_matmul_fp16<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);

    cudaDeviceSynchronize();

    // CPU 验证
    cpu_matmul(a, b, c_cpu, M, N, K);

    // 浮点数验证（使用较小的阈值，因为 half 存在精度损失）
    bool errors = false;
    for(int i = 0; i < M * K; i++)
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