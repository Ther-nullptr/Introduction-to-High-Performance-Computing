
#include <stdio.h> 
#include <stdlib.h>

__global__  void vecAdd(float* A, float* B, float* C) 
{ 
    // threadIdx.x is a built-in variable provided by CUDA at runtime 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    C[i] = A[i] + B[i]; 
}


int  main(int argc, char **argv)
{
    int i, N = 16384;  // default vector size
    float *A, *devPtrA;
    float *B, *devPtrB; 
    float *C, *devPtrC; 

    if (argc > 1) N = atoi(argv[1]);  // get size of the vectors
    printf("Running GPU vecAdd for %i elements\n", N);

    A = (float*)malloc(N * sizeof(float));  // allocate memory 
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));

    for (i = 0; i < N; i++)  // generate random data
    {
        A[i] = (float)random();
        B[i] = (float)RAND_MAX - A[i];
    }
    
    // ------------
    cudaMalloc((void**)&devPtrA, N * sizeof(float)); 
    cudaMalloc((void**)&devPtrB, N * sizeof(float)); 
    cudaMalloc((void**)&devPtrC, N * sizeof(float)); 
    
    cudaMemcpy(devPtrA, A, N * sizeof(float),  cudaMemcpyHostToDevice); 
    cudaMemcpy(devPtrB, B, N * sizeof(float),  cudaMemcpyHostToDevice); 
    
    vecAdd<<<N/512, 512>>>(devPtrA, devPtrB, devPtrC); // call compute kernel
    
    cudaMemcpy(C, devPtrC, N * sizeof(float),  cudaMemcpyDeviceToHost); 
    
    cudaFree(devPtrA); 
    cudaFree(devPtrB); 
    cudaFree(devPtrC); 
    // ------------
        
    for (i = 0; i < 10; i++)  // print out first 10 results
        printf("C[%i]=%.2f\n", i, C[i]);
  
    free(A);  // free allocated memory
    free(B);
    free(C);
}

