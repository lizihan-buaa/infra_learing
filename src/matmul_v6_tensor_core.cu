#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Tensor Core 对 FP16 的标准形状要求通常是 16x16x16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

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
    // 每个 warp 负责一个 16x16 的 tile
    int warpId = threadIdx.x / 32; // 计算当前线程属于当前 Block 中的第几个 Warp

    // 计算当前 Warp 负责计算输出矩阵C中的哪个16*16的小块（Tile）
    // 一个block中有4个warp，负责C中64*16的小块
    int warpM = (blockIdx.y * (blockDim.x / 32) + warpId) * WMMA_M;
    int warpK = blockIdx.x * WMMA_K;

    // 声明 fragments: a, b 使用 half, accumulator 使用 float
    // （角色（左矩阵、右矩阵、累加矩阵），A的行，B的列，共享维度，数据类型，排布方式（行优先、列优先））
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for(int i = 0; i < n; i += WMMA_N)
    {
        if(warpM < m && warpK < k)
        {
            const half *a_tile = a + warpM * n + i;
            const half *b_tile = b + i * k + warpK;

            // 加载矩阵
            wmma::load_matrix_sync(a_frag, a_tile, n); // （目标容器，矩阵块地址，跳步长度）
            wmma::load_matrix_sync(b_frag, b_tile, k);

            // 矩阵乘加
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if(warpM < m && warpK < k)
    {
        float *c_tile = c + warpM * k + warpK;
        // 将结果存回内存
        wmma::store_matrix_sync(c_tile, acc_frag, k, wmma::mem_row_major);
    }
}

// CPU 端用于验证的 FP16 矩阵乘法
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
        
    // 每个 block 配置 4 个 warp (128 threads)
    dim3 dimBlock(128, 1);
    // 每个 block 处理 4 个 16x16 tile (纵向排列)
    dim3 dimGrid(K_GLOBAL / WMMA_K, (M_GLOBAL / WMMA_M) / 4);

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