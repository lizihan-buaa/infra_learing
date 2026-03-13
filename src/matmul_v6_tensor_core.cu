#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define M_GLOBAL 1024
#define N_GLOBAL 512
#define K_GLOBAL 1024

__managed__ signed char a[M_GLOBAL * N_GLOBAL];
__managed__ signed char b[N_GLOBAL * K_GLOBAL];
__managed__ int c_gpu[M_GLOBAL * K_GLOBAL];
__managed__ int c_cpu[M_GLOBAL * K_GLOBAL];

__global__ void gpu_matmul6(const signed char *a, const signed char *b, int *c,
                            int m, int n, int k)
{
    // 每个warp负责一个16x16 tile
    int warpId = threadIdx.x / 32;

    int warpM = (blockIdx.y * (blockDim.x / 32) + warpId) * WMMA_M;
    int warpN = blockIdx.x * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;

    wmma::fill_fragment(acc_frag, 0);

    for(int i = 0; i < n; i += WMMA_K)
    {
        if(warpM < m && warpN < k)
        {
            const signed char *a_tile = a + warpM * n + i;
            const signed char *b_tile = b + i * k + warpN;

            wmma::load_matrix_sync(a_frag, a_tile, n);
            wmma::load_matrix_sync(b_frag, b_tile, k);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if(warpM < m && warpN < k)
    {
        int *c_tile = c + warpM * k + warpN;
        wmma::store_matrix_sync(c_tile, acc_frag, k, wmma::mem_row_major);
    }
}

void cpu_matmul(const signed char *a, const signed char *b, int *c,
                int m, int n, int k)
{
    for(int y = 0; y < m; y++)
    {
        for(int x = 0; x < k; x++)
        {
            int tmp = 0;

            for(int step = 0; step < n; step++)
            {
                tmp += a[y * n + step] * b[step * k + x];
            }

            c[y * k + x] = tmp;
        }
    }
}

int main()
{
    // 初始化数据（避免 int8 溢出）
    for(int y = 0; y < M_GLOBAL; y++)
    {
        for(int x = 0; x < N_GLOBAL; x++)
        {
            a[y * N_GLOBAL + x] = rand() % 16 - 8;
        }
    }

    for(int y = 0; y < N_GLOBAL; y++)
    {
        for(int x = 0; x < K_GLOBAL; x++)
        {
            b[y * K_GLOBAL + x] = rand() % 16 - 8;
        }
    }

    // 每个block 4 warp
    dim3 dimBlock(128,1);

    // 每个warp算一个16x16
    dim3 dimGrid(K_GLOBAL / WMMA_N, (M_GLOBAL / WMMA_M) / 4);

    gpu_matmul6<<<dimGrid, dimBlock>>>(a, b, c_gpu, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    cudaDeviceSynchronize();

    cpu_matmul(a, b, c_cpu, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    bool errors = false;

    for(int y = 0; y < M_GLOBAL; y++)
    {
        for(int x = 0; x < K_GLOBAL; x++)
        {
            int idx = y * K_GLOBAL + x;

            if(c_cpu[idx] != c_gpu[idx])
            {
                errors = true;
                break;
            }
        }
    }

    printf("Result: %s\n", errors ? "Failed" : "Passed");

    return 0;
}