
#include <stdio.h> 
#include <stdlib.h>
#include <sys/time.h>

/* wrong !*/
__global__ void sum0(double *v)
{
    unsigned int t = threadIdx.x;
    unsigned int stride;

    for (stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (t < stride)
            v[t] += v[t+stride];
    }
}

__global__ void sum(double *v)
{
    extern double __shared__ sd[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int s;

    sd[tid] = v[i] + v[i+blockDim.x];
 
    __syncthreads();

    // do reduction in shared mem
    for (s = blockDim.x/2; s > 0; s >>= 1) 
    {
        if (tid < s)
            sd[tid] += sd[tid + s];
        __syncthreads();
    }
    
    // write result for this block to global mem 
    if (tid == 0) v[blockIdx.x] = sd[0];
}



int  main(int argc, char **argv)
{
    int i, N = 2097152;  // default vector size
    double s = 0.0, *A, *devPtrA;

    struct timeval t1, t2;
    long msec1, msec2;
    float gflop;

    if (argc > 1) N = atoi(argv[1]);  // get size of the vectors
    A = (double*)malloc(N * sizeof(double));  // allocate memory 

    srand(1);
    for (i = 0; i < N; i++)  // generate random data
    {
        A[i] = (double)rand()/RAND_MAX;
    }

    printf("Running GPU sum for %i elements\n", N);
    
    // ------------ GPU-related
    cudaMalloc((void**)&devPtrA, N * sizeof(double)); 
    cudaMemcpy(devPtrA, A, N * sizeof(double),  cudaMemcpyHostToDevice); 
    
    int threads = 64;
    int old_blocks, blocks = N / threads / 2;
    blocks = (blocks == 0) ? 1 : blocks;
    old_blocks = blocks;

    gettimeofday(&t1, NULL);
    msec1 = t1.tv_sec * 1000000 + t1.tv_usec;

    while (blocks > 0) 
    {
        printf("Grid/thread dims are (%d), (%d)\n", blocks, threads);
        sum<<<blocks, threads, threads*sizeof(double)>>>(devPtrA); // call compute kernel
        old_blocks = blocks;
        blocks = blocks / threads / 2;
    };

    if (blocks == 0 && old_blocks != 1)
    {
        printf("*Grid/thread dims are (%d), (%d)\n", 1, old_blocks/2);
        sum<<<1, old_blocks/2, old_blocks/2*sizeof(double)>>>(devPtrA); // call compute kernel
    }

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) 
    {
        fprintf(stderr, "CUDA error: %s.\n", cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
         
    // wait until GPU kernel is done
    cudaThreadSynchronize();

    gettimeofday(&t2, NULL);
    msec2 = t2.tv_sec * 1000000 + t2.tv_usec;

    cudaMemcpy(&s, devPtrA, sizeof(double),  cudaMemcpyDeviceToHost); 

    cudaFree(devPtrA); 
    // .............
    
    printf("sum=%.2f\n", s);
    gflop = ((N-1) / (msec2 - msec1)) / 1000.0f;
    printf("sec = %f   GFLOPS = %.3f\n", (msec2-msec1)/1000000.0f, gflop);

    free(A);  // free allocated memory
}

