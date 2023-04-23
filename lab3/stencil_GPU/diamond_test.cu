#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 128
#define R 2
#define EPSILON 0.001

__global__ void stencil(float *in, float *out, int n, int r)
{
    // Allocate shared memory for diamond tile
    extern __shared__ float tile[];

    // Calculate global indices for this thread block
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // Calculate local indices within the diamond tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Calculate global indices within the input and output arrays
    int x = bx * (2 * r + 1) + tx - r;
    int y = by * (2 * r + 1) + ty - r;
    int z = bz * (2 * r + 1) + tz - r;

    // Copy input data to shared memory
    if (x >= 0 && x < n && y >= 0 && y < n && z >= 0 && z < n)
    {
        tile[tz * (2 * r + 1) * (2 * r + 1) + ty * (2 * r + 1) + tx] = in[z * n * n + y * n + x];
    }
    else
    {
        tile[tz * (2 * r + 1) * (2 * r + 1) + ty * (2 * r + 1) + tx] = 0;
    }

    // Synchronize threads to ensure data is loaded into shared memory
    __syncthreads();

    // Perform stencil computation on diamond tile
    float result = 0;
    for (int dz = -r; dz <= r; dz++)
    {
        for (int dy = -r; dy <= r; dy++)
        {
            for (int dx = -r; dx <= r; dx++)
            {
                int tx = dx + r;
                int ty = dy + r;
                int tz = dz + r;
                result += tile[tz * (2 * r + 1) * (2 * r + 1) + ty * (2 * r + 1) + tx];
            }
        }
    }
    if (x < n && y < n && z < n)
    {
        out[z * n * n + y * n + x] = result;
    }
}

int main()
{
    float *in, *out, *d_in, *d_out;
    int i, j, k;
    int size = N * N * N * sizeof(float);

    // Allocate memory on host and device
    in = (float *)malloc(size);
    out = (float *)malloc(size);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Initialize input data
    for (k = 0; k < N; k++)
    {
        for (j = 0; j < N; j++)
        {
            for (i = 0; i < N; i++)
            {
                in[k * N * N + j * N + i] = (float)(i + j + k);
            }
        }
    }

    // Copy input data to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    // Launch kernel with appropriate block and grid sizes
    dim3 threadsPerBlock(2 * R + 1, 2 * R + 1, 2 * R + 1);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);
    stencil<<<blocksPerGrid, threadsPerBlock, (2 * R + 1) * (2 * R + 1) * (2 * R + 1) * sizeof(float)>>>(d_in, d_out, N, R);

    // Copy output data back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Verify correctness of output data
    for (k = 0; k < N; k++)
    {
        for (j = 0; j < N; j++)
        {
            for (i = 0; i < N; i++)
            {
                float expected = (float)(i + j + k) * ((2 * R + 1) * (2 * R + 1) * (2 * R + 1));
                float actual = out[k * N * N + j * N + i];
                if (fabs(expected - actual) > EPSILON)
                {
                    printf("Error at (%d,%d,%d): expected %f, actual %f\n", i, j, k, expected, actual);
                }
            }
        }
    }

    // Free memory on host and device
    free(in);
    free(out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}