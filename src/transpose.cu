#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// 全局内存合并访存问题：连续的thread访问连续的空间效率高、
// transpose中读可以连续，但写地址不连续
// 通过shared memory来解决（将其当作缓存），写的时候去取地址连续的顺序写入

#define BLOCK_SIZE 32

#define M 3000
#define N 1000

__managed__ int matrix[N * M];
__managed__ int gpu_result[M * N];
__managed__ int cpu_result[M * N];

__global__ void gpu_transpose(int *in, int *out, int m, int n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ int sub[BLOCK_SIZE * (BLOCK_SIZE + 1)];

    if(x < M && y < N)
    {
        sub[threadIdx.y * (BLOCK_SIZE + 1) + threadIdx.x] = in[y * M + x];
    }
    __syncthreads();

    // 线程在block内坐标不变，block位置发生变化
    int x1 = threadIdx.x + blockDim.y * blockIdx.y;
    int y1 = threadIdx.y + blockDim.x * blockIdx.x;

    if(x1 < N && y1 < M)
    {
        out[y1 * N + x1] = sub[threadIdx.x * (BLOCK_SIZE + 1) + threadIdx.y];
    }
}
void cpu_transpose(int *in, int *out, int m, int n)
{
    for(int y=0; y<n; y++)
    {
        for(int x=0; x<m; ++x)
        {
            out[x * n + y] = in[y * m + x];
        }
    }
}

int main()
{
    for(int y=0; y<N; ++y)
    {
        for(int x=0; x<M; ++x)
        {
            matrix[y * M + x] = rand() % 1024;
        }
    }
    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    dim3 dimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    for(int i=0; i<20; i++) // 调用20次核函数，用于计时
    {
        gpu_transpose<<<dimGrid, dimBlock>>>(matrix, gpu_result, M, N);
    }
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cpu_transpose(matrix, cpu_result, M, N);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    float time_cpu, time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);
    bool errors = false;
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            if(fabs(cpu_result[y * N + x] - gpu_result[y * N + x]) > 1.0e-10)
            {
                errors = true;
            }
        }
    }
    printf("Results:%s\n", errors?"Failed":"Passed");
    printf("CPU Time:%.2f\nGPU Time:%.2f\n", time_cpu, time_gpu);
    return 0;
}