#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// 单线程负责C中多结果（TM*TK），结合共享内存分块和寄存器分块
// BM, BK, BK: Block 级别的分块尺寸。一个 Block 负责输出C矩阵中BM*BK的区域
// TM, TK: Thread 级别的分块尺寸。一个 Thread 负责输出TM*TK的区域
// numThreadsBlocktile: 每个 Block 里的线程数。由于每个线程算TM*TK个点，所以线程数 =(BM*BK) / (TM*TK)

// 解决以下问题：
// MIO访存指令的执行与发射管线占满导致warp等待（并未完全解决）
// 在 GPU SM 中有一个专门处理 memory / shared memory / special math 指令的 pipeline：MIO pipeline（并不是它本身搬运数据）

// 这个代码之所以快，是因为它建立了三层数据流：
// Global Memory: 原始大矩阵
// Shared Memory (sub_a, sub_b): 存放当前正在处理的 Block Tile
// Registers (regM, regK, threadResults): 存放当前线程正在处理的 Thread Tile

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

__global__ void gpu_matmul3(int *a, int *b, int *c, int m, int n, int k)
{
    __shared__ int sub_a[BM][BN];
    __shared__ int sub_b[BN][BK];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // 将线程号转化成一维，是为了后面的循环
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // thread一维索引（分配warp按这个序号）

    const int threadRow = threadIdx.y * TM; // thread负责的矩阵对象在C中的位置（block内）
    const int threadCol = threadIdx.x * TK;

    int threadResults[TM * TK] = {0};
    int regA[TM];
    int regB[TK];

    int numThreads = blockDim.x * blockDim.y; // 每个block内线程个数

    for (int step = 0; step < (n + BN - 1) / BN; step++) // 共享内存分块
    {

        // ---- load A tile ----
        for (int i = tid; i < BM * BN; i += numThreads)
        // 由于一个 Block 有 8*8 = 64 个线程，而 sub_a 有 64*8 = 512 个元素
        // 这个循环就是让每个线程通过循环多次，共同把 Global 里的数据“搬”到 Shared Memory 
        // 这实现了内存合并访问 (Coalesced Access)。
        {
            int row = i / BN;
            int col = i % BN; // 得到线程在block中所处的位置

            int g_r = cRow * BM + row;
            int g_c = step * BN + col; // 得到线程要加载的数据在原矩阵中的位置

            if (g_r < m && g_c < n)
                sub_a[row][col] = a[g_r * n + g_c];
            else
                sub_a[row][col] = 0;
        }

        // ---- load B tile ----
        for (int i = tid; i < BN * BK; i += numThreads)
        {
            int row = i / BK;
            int col = i % BK;

            int g_r = step * BN + row;
            int g_c = cCol * BK + col;

            if (g_r < n && g_c < k)
                sub_b[row][col] = b[g_r * k + g_c];
            else
                sub_b[row][col] = 0;
        }

        __syncthreads();

        // ---- compute ----
        // 对block内的每个thread编程，首先找到它在block内（smem内）的坐标，然后进行寄存器分块，和smem分块方式完全相同
        for (int dotIdx = 0; dotIdx < BN; dotIdx++)
        {

            for (int i = 0; i < TM; i++)
                regA[i] = sub_a[threadRow + i][dotIdx];

            for (int i = 0; i < TK; i++)
                regB[i] = sub_b[dotIdx][threadCol + i];

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

    // ---- write back ----
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

    gpu_matmul3<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);
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