#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hello_from_gpu(void)
{
    printf("GPU-2: Hello world!\n");
}

__global__ void hello_world(void)
{
    printf("GPU: Hello world!\n");
    hello_from_gpu<<<1, 10>>>();
    // cudaDeviceReset();
}

int main(int argc, char **argv)
{
    printf("CPU: Hello world!\n");
    hello_world<<<1, 10>>>();
    cudaDeviceReset(); // if no this line ,it can not output hello world from gpu
    return 0;
}