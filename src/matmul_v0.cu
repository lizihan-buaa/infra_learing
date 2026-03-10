#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
// a[][] @ b[][] = c[][]

//                    b00 b01 b02 b03
//                    b10 b11 b12 b13
//                    b20 b21 b22 b23
//                    b30 b31 b32 b33
//
// a00 a01 a02 a03    c00 c01 c02 c03
// a10 a11 a12 a12    c10 c11 c12 c13
// a20 a21 a22 a23    c20 c21 c22 c23
// a30 a31 a32 a33    c30 c31 c32 c33
//
// 矩阵按照行优先一维存储 
// index = y(行) * size + x(列)
// step 0 -> 3
// a_index = y * size + step (一行)
// b_index = step * size + x (一列)

void cpu_matmul(int *a, int *b, int *c, const int size)
{
    for(int y=0; y<size; ++y)
    {
        for(int x=0; x<size; ++x) //外面两层循环定位了结果的位置
        {
            int tmp = 0;
            for(int step=0; step<size; ++step) //step用于找A的哪一行，B的哪一列
            {
                tmp += a[y*size + step] * b[step * size + x];
            }
            c[y * size + x] = tmp;
        }
    }
}

__global__ void gpu_matmul(int *a, int *b, int *c, const int size)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    // blockDim：Block 的维度（每个维度上的线程数)
    // blockIdx：当前 Block 在 Grid 中的索引（从 0 开始）
    // threadIdx：当前 Thread 在 Block 中的索引（从 0 开始）
    int tmp = 0;
    if(x < size && y < size) //否则出越界报错
    {
        for(int step=0; step<size; ++step) // 该方法存在重复访存的问题
        {
            tmp += a[y*size + step] * b[step * size + x];
        }
        c[y * size + x] = tmp;
    }
}


int main()
{
    int matrix_size = 1000;
    int memsize = sizeof(int) * matrix_size * matrix_size;

    int *h_a, *h_b, *h_c, *h_cc; // on host(cpu)
    cudaMallocHost((void**)&h_a, memsize);
    cudaMallocHost((void**)&h_b, memsize);
    cudaMallocHost((void**)&h_c, memsize);
    cudaMallocHost((void**)&h_cc, memsize);
    // 必须传 d_a 的地址（&d_a），让函数能直接操作 d_a 这个变量本身。
    // 如果直接传 d_a（int* 类型），只是传了指针的 “值拷贝”，函数内部修改的是拷贝后的临时值，不会影响外部的 d_a。

    // h_a/b/c 初始化
    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            h_a[y * matrix_size + x] = rand() % 1024;
            h_b[y * matrix_size + x] = rand() % 1024;
        }
    }

    int *d_a, *d_b, *d_c; // on device(gpu) 
    cudaMalloc((void**) &d_a, memsize); // global memory
    cudaMalloc((void**) &d_b, memsize);
    cudaMalloc((void**) &d_c, memsize);
    
    // d_a/b/c 初始化
    cudaMemcpy(d_a, h_a, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, memsize, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // 一般一个block不多于1024个线程,最好是32整数倍（warp = 32，32个线程共享指令）

    gpu_matmul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, matrix_size);
    cudaGetLastError();  // 检查核函数启动是否出错
    cudaDeviceSynchronize();  // 同步设备，等待核函数执行完毕

    cudaMemcpy(h_c, d_c, memsize, cudaMemcpyDeviceToHost);

    cpu_matmul(h_a, h_b, h_cc, matrix_size);

    bool errors = false;
    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            if(fabs(h_cc[y*matrix_size + x] - h_c[y*matrix_size + x]) > (1.0e-10))
            {
                errors = true;
            }
        }
    }
    printf("Result: %s\n", errors?"Failed":"Passed");

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}