#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// 利用shared memory优化矩阵乘 
// data global mem -> shared memory
// thread shared mem -> register
// shared mem is in SM(stream multi-processor) same block shared mem 

//                    b00 b01 b02 b03
//                    b10 b11 b12 b13
//                    b20 b21 b22 b23
//                    b30 b31 b32 b33
//
// a00 a01 a02 a03    c00 c01 c02 c03
// a10 a11 a12 a12    c10 c11 c12 c13    block(1,0) -> shared mem(将该block所需直接load到smem上)
// a20 a21 a22 a23    c20 c21 c22 c23     c20 c21
// a30 a31 a32 a33    c30 c31 c32 c33     c30 c31

// 若smem放不下，需要分块计算
//                          b00 b01 -> sub_b_step0
//                          b10 b11
//
//                          b20 b21 -> sub_b_step1
//                          b30 b31
// sub_a_step0   sub_a_step1
// a20 a21       a22 a23    c20 c21
// a30 a31       a32 a33    c30 c31 -> sub_c
//
// sub_c = sub_a_step0 @ sub_b_step0 + sub_a_step1 @ sub_b_step1
//
// for(int step=0; step< N/block_size; step++) 其中，N是A的列，B的行
// {
//     load sub_a_step to smem;
//     load sub_b_step to smem;
//     tmp += sub_a_step_on_sram @ sub_b_step_on_sram
// }
// sub_c = temp;
//
#define M 1000
#define N 500
#define K 1000

__managed__ int a[M * N]; // 统一内存
__managed__ int b[N * K];
__managed__ int c_gpu[M * K];
__managed__ int c_cpu[M * K];

#define BLOCK_SIZE 16

__global__ void gpu_matmul1(int *a, int *b, int *c, int m, int n, int k)
{
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE]; // 每个block内的所有线程都指向这同一块空间
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tmp = 0;
    int idx;
    for(int step=0; step<((n + BLOCK_SIZE - 1) / BLOCK_SIZE); step++) // 每个结果block对应多个A,B块，需要多次load（shared_mem大小有限）
    {
        // load 子矩阵a
        // 计算子矩阵对应位置的线程号
        int idx_x = step * BLOCK_SIZE + threadIdx.x; // block内某个线程对应的位置的读数
        int idx_y = y;
        idx = idx_y * n + idx_x; // 转化为一维
        // 判断越界 补0 （脏数据）
        if(idx_x >= n || idx_y >= m)
        {
            sub_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            sub_a[threadIdx.y][threadIdx.x] = a[idx];
        }
        //load 子矩阵b
        idx_x = x;
        idx_y = step * BLOCK_SIZE + threadIdx.y;
        idx = idx_y *  k + idx_x;
        if(idx_x >= k || idx_y >= n)
        {
            sub_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            sub_b[threadIdx.y][threadIdx.x] = b[idx];
        }

        __syncthreads(); // 同步

        for(int i=0; i<BLOCK_SIZE; i++) // 循环的是a的一行，b的一列
        {
            tmp += sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x]; // 一个线程负责c的一个元素
        }
        __syncthreads();
    }
    // 判断越界
    if(x < k && y < m)
    {
        c[y * k + x] = tmp;
    }
}

void cpu_matmul1(int *a, int *b, int *c, int m, int n, int k)
{
    for(int y=0; y<m; ++y)
    {
        for(int x=0; x<k; ++x)
        {
            int tmp = 0;
            for(int step=0; step<n; step++)
            {
                tmp += a[y * n + step] * b[step * k + x];
            }
            c[y * k + x] = tmp;
        }
    }
}

int main()
{
    // y表示行，x表示列
    // 数据初始化
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            a[y * N + x] = rand() % 1024;
        }
    }

    for(int y=0; y<N; ++y)
    {
        for(int x=0; x<K; ++x)
        {
            b[y * K + x] = rand() % 1024;
        }
    }

    unsigned int grid_x = (K + BLOCK_SIZE - 1) / BLOCK_SIZE; // 列数
    unsigned int grid_y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE; // 行数

    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matmul1<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);
    cpu_matmul1(a, b, c_cpu, M, N, K);

    bool errors = false;
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<K; ++x)
        {
            if(fabs(c_cpu[y * K + x] - c_gpu[y * K + x]) > (1.0e-10))
            {
                errors = true;
            }
        }
    }

    printf("Result: %s\n", errors?"Failed":"Passed");

    return 0;
}