#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// 添加双缓冲机制，数据乒乓

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

__global__ void gpu_matmul5(int *a, int *b, int *c, int m, int n, int k)
{
    // Shared Memory 布局：[2][行][列]
    // BN 为 8，使用 int4 加载时，每行刚好是 2 个 int4
    __shared__ int sub_a[2][BM][BN]; 
    __shared__ int sub_b[2][BN][BK];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    const int numThreads = blockDim.x * blockDim.y;

    const int threadRow = threadIdx.y * TM; 
    const int threadCol = threadIdx.x * TK;

    int threadResults[TM * TK] = {0};

    int write_idx = 0; // 当前写入 Shared Memory 的索引

    // --- 1. 预加载第 0 块数据 ---
    {
        // 加载 A (每线程加载一部分)
        for (int i = tid; i < (BM * BN) / 4; i += numThreads) {
            int row = i / (BN / 4); int col_v = i % (BN / 4);
            int g_r = cRow * BM + row; int g_c = 0 * BN + col_v * 4;
            if (g_r < m && (g_c + 3) < n) 
                *((int4*)&sub_a[write_idx][row][col_v * 4]) = *((int4*)&a[g_r * n + g_c]);
            else 
                for (int v = 0; v < 4; v++) 
                    sub_a[write_idx][row][col_v * 4 + v] = (g_r < m && (g_c + v) < n) ? a[g_r * n + g_c + v] : 0;
        }
        // 加载 B
        for (int i = tid; i < (BN * BK) / 4; i += numThreads) {
            int row = i / (BK / 4); int col_v = i % (BK / 4);
            int g_r = 0 * BN + row; int g_c = cCol * BK + col_v * 4;
            if (g_r < n && (g_c + 3) < k) 
                *((int4*)&sub_b[write_idx][row][col_v * 4]) = *((int4*)&b[g_r * k + g_c]);
            else 
                for (int v = 0; v < 4; v++) 
                    sub_b[write_idx][row][col_v * 4 + v] = (g_r < n && (g_c + v) < k) ? b[g_r * k + g_c + v] : 0;
        }
    }
    __syncthreads(); // 确保第 0 块加载完成

    // --- 2. 循环计算 ---
    int num_steps = (n + BN - 1) / BN;
    for (int step = 0; step < num_steps; step++) 
    {
        int read_idx = write_idx;      // 当前计算用的缓冲区
        write_idx = 1 - read_idx;      // 下一轮加载用的缓冲区

        // A. 异步发射下一轮数据的加载 (如果存在下一轮)
        if (step + 1 < num_steps) {
            int next_step = step + 1;
            for (int i = tid; i < (BM * BN) / 4; i += numThreads) {
                int row = i / (BN / 4); int col_v = i % (BN / 4);
                int g_r = cRow * BM + row; int g_c = next_step * BN + col_v * 4;
                if (g_r < m && (g_c + 3) < n) 
                    *((int4*)&sub_a[write_idx][row][col_v * 4]) = *((int4*)&a[g_r * n + g_c]);
                else 
                    for (int v = 0; v < 4; v++) 
                        sub_a[write_idx][row][col_v * 4 + v] = (g_r < m && (g_c + v) < n) ? a[g_r * n + g_c + v] : 0;
            }
            for (int i = tid; i < (BN * BK) / 4; i += numThreads) {
                int row = i / (BK / 4); int col_v = i % (BK / 4);
                int g_r = next_step * BN + row; int g_c = cCol * BK + col_v * 4;
                if (g_r < n && (g_c + 3) < k) 
                    *((int4*)&sub_b[write_idx][row][col_v * 4]) = *((int4*)&b[g_r * k + g_c]);
                else 
                    for (int v = 0; v < 4; v++) 
                        sub_b[write_idx][row][col_v * 4 + v] = (g_r < n && (g_c + v) < k) ? b[g_r * k + g_c + v] : 0;
            }
        }

        // B. 计算当前 read_idx 中的数据
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BN; dotIdx++) {
            int regA[TM], regB[TK];
            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = sub_a[read_idx][threadRow + i][dotIdx];
            #pragma unroll
            for (int i = 0; i < TK; i++) regB[i] = sub_b[read_idx][dotIdx][threadCol + i];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TK; j++) {
                    threadResults[i * TK + j] += regA[i] * regB[j];
                }
            }
        }
        
        // C. 关键：同步，确保下一轮加载完成且本轮计算结束
        __syncthreads(); 
    }

    // --- 3. 写回结果 ---
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TK; j++) {
            int g_r = cRow * BM + threadRow + i;
            int g_c = cCol * BK + threadCol + j;
            if (g_r < m && g_c < k) c[g_r * k + g_c] = threadResults[i * TK + j];
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

    gpu_matmul5<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);
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