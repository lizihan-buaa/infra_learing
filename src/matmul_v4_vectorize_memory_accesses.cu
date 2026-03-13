#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// 解决以下问题：增大带宽利用率，按照int4类型进行数据读取
// 这种优化的核心在于减少指令发射数量和提升内存带宽利用率
// MIO访存指令的执行与发射管线占满导致warp等待（v3并未完全解决）

#define M 1000
#define N 500
#define K 1000
// 每个block有64个线程
// 每个线程负责C中64个对象
#define BM 64
#define BK 64
#define BN 8
#define TM 8
#define TK 8

__managed__ int a[M * N]; // 统一内存
__managed__ int b[N * K];
__managed__ int c_gpu[M * K];
__managed__ int c_cpu[M * K];

__global__ void gpu_matmul4(int *a, int *b, int *c, int m, int n, int k)
{
    __shared__ int sub_a[BM][BN + 1]; 
    __shared__ int sub_b[BN][BK];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    const int numThreads = blockDim.x * blockDim.y;

    const int threadRow = threadIdx.y * TM; 
    const int threadCol = threadIdx.x * TK;

    int threadResults[TM * TK] = {0};

    for (int step = 0; step < (n + BN - 1) / BN; step++) 
    {
        // ---- 向量化加载 A (sub_a: 64x8) ----
        // 512个元素，每个线程搬运 512/64=8个。由于BN=8，每行搬2个int4
        for (int i = tid; i < (BM * BN) / 4; i += numThreads)
        {
            int row = i / (BN / 4);
            int col_v = i % (BN / 4);
            int g_r = cRow * BM + row;
            int g_c = step * BN + col_v * 4;

            if (g_r < m && (g_c + 3) < n) 
            {
                // 只有完全在边界内且对齐时才使用 int4
                // 注意：如果矩阵起始地址不对齐，此处仍需小心。__managed__ 默认对齐。
                // *((int4*)&sub_a[row][col_v * 4]) = *((int4*)&a[g_r * n + g_c]);
                int4 val = *((int4*)&a[g_r * n + g_c]);
                sub_a[row][col_v * 4 + 0] = val.x;
                sub_a[row][col_v * 4 + 1] = val.y;
                sub_a[row][col_v * 4 + 2] = val.z;
                sub_a[row][col_v * 4 + 3] = val.w;
            }
            else
            {
                // 逐个处理边缘部分
                for (int v = 0; v < 4; v++)
                {
                    sub_a[row][col_v * 4 + v] = (g_r < m && (g_c + v) < n) ? a[g_r * n + g_c + v] : 0;
                }
            }
        }

        // ---- 向量化加载 B (sub_b: 8x64) ----
        // 512个元素，BN=8, BK=64。每行16个int4
        for (int i = tid; i < (BN * BK) / 4; i += numThreads)
        {
            int row = i / (BK / 4);
            int col_v = i % (BK / 4);
            int g_r = step * BN + row;
            int g_c = cCol * BK + col_v * 4;

            if (g_r < n && (g_c + 3) < k)
            {
                *((int4*)&sub_b[row][col_v * 4]) = *((int4*)&b[g_r * k + g_c]);
            }
            else
            {
                for (int v = 0; v < 4; v++)
                {
                    sub_b[row][col_v * 4 + v] = (g_r < n && (g_c + v) < k) ? b[g_r * k + g_c + v] : 0;
                }
            }
        }

        __syncthreads();

        // ---- 计算 (寄存器分块) ----
        #pragma unroll
        // 告诉编译器,把这个循环拆开，直接把里面的代码重复写出来，不做循环判断
        for (int dotIdx = 0; dotIdx < BN; dotIdx++)
        {
            int regA[TM];
            int regB[TK];
            #pragma unroll
            for (int i = 0; i < TM; i++)
            {
                regA[i] = sub_a[threadRow + i][dotIdx];
            }
            #pragma unroll
            for (int i = 0; i < TK; i++)
            {
                regB[i] = sub_b[dotIdx][threadCol + i];
            }
            for (int i = 0; i < TM; i++)
            {
                for (int j = 0; j < TK; j++)
                {
                    threadResults[i * TK + j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    // ---- 带有边界检查的写回 ----
    for (int i = 0; i < TM; i++)
    {
        for (int j = 0; j < TK; j++)
        {
            int g_r = cRow * BM + threadRow + i;
            int g_c = cCol * BK + threadCol + j;
            if (g_r < m && g_c < k)
            {
                c[g_r * k + g_c] = threadResults[i * TK + j];
            }
        }
    }
}


void cpu_matmul(int *a, int *b, int *c, int m, int n, int k)
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

    dim3 dimBlock((BK / TK), (BM / TM)); // 线程数变少了，但每个线程变强了
    dim3 dimGrid((K + BK - 1) / BK, (M + BM - 1) / BM);

    gpu_matmul4<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);
    cpu_matmul(a, b, c_cpu, M, N, K);

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