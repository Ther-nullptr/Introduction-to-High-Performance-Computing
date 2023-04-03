
#include <stdio.h> 
#include <stdlib.h>

__global__  void strideCopy(float* A, float* B, int stride) 
{ 
    // threadIdx.x is a built-in variable provided by CUDA at runtime 
    long int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride; 
    A[i] = B[i];
}


int  main(int argc, char **argv)
{
    unsigned long int N = 16777216;  // default vector size
    int stride = 1;     // default offset
    float *devPtrA;
    float *devPtrB; 

    if (argc > 1) N = atoi(argv[1]) / 4;  // get size of the vectors
    if (argc > 2) stride = atoi(argv[2]);  // get size of the vectors

    printf("Running GPU misaligned copy for %u bytes and stride %d\n", N * sizeof(float), stride);

    cudaSetDevice(0);

    cudaMalloc((void**)&devPtrA, (N*stride) * sizeof(float)); 
    cudaMalloc((void**)&devPtrB, (N*stride) * sizeof(float)); 

    cudaEvent_t start1, stop1, start2, stop2;
    float time, k_time;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    // figure out time to start a kernel
    cudaEventRecord(start1, 0);
    strideCopy<<<1, 1>>>(devPtrA, devPtrB, stride);
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&k_time, start1, stop1);

    // do the bandwidth test
    cudaEventRecord(start2, 0);

    strideCopy<<<N/512, 512>>>(devPtrA, devPtrB, stride);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) 
    {
        fprintf(stderr, "CUDA error: %s.\n", cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }

    cudaEventElapsedTime(&time, start2, stop2);

    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cudaFree(devPtrA); 
    cudaFree(devPtrB);    

    printf("effective bandwidth %f GBytes/sec\n", 2.0f * N * sizeof(float) / ((time-k_time) / 1000) / 1e9);
}

