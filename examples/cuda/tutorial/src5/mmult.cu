
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// a = b * c
 __global__ void mmult1(float *a, float *b, float *c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    float sum = 0.0;
    
    for (int k = 0; k < N; k++)
        sum += b[i+N*k] * c[k+N*j];

    a[i+N*j] = sum;
}

 __global__ void mmult2(float *a, float *b, float *c, int N)
{
    extern __shared__ float cb[];

    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y;
    float sum = 0.0;

    for (int ks = 0; ks < N; ks += blockDim.x ) 
    {
      cb[tx] = c[ks+tx+N*j];
      for (int k = ks; k < ks+blockDim.x; ++k)
        sum += b[i+N*k] * cb[k-ks];
    }

    a[i+N*j] = sum;
}

 __global__ void mmult3(float *a, float *b, float *c, int N)
{
    extern __shared__ float cb[];

    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x*2 + tx;
    int j = blockIdx.y;
    float sum0 = 0.0;
    float sum1 = 0.0;

    for (int ks = 0; ks < N; ks += blockDim.x) 
    {
      cb[tx] = c[ks+tx+N*j];
      __syncthreads();

      for (int k = ks; k < ks+blockDim.x; ++k)
      {
        sum0 += b[i+N*k] * cb[k-ks];
        sum1 += b[i+blockDim.x+N*k] * cb[k-ks];
      }
      __syncthreads();
    }

    a[i+N*j] = sum0;
    a[i+blockDim.x+N*j] = sum1;
}


 __global__ void mmult4(float *a, float *b, float *c, int N)
{
    extern __shared__ float cb[];
    float *cb0 = &cb[0];
    float *cb1 = &cb[blockDim.x];

    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * 2;
    float sum0 = 0.0;
    float sum1 = 0.0;

    for (int ks = 0; ks < N; ks += blockDim.x) 
    {
      cb0[tx] = c[ks+tx+N*j];
      cb1[tx] = c[ks+tx+N*(j+1)];
      __syncthreads();

      for (int k = ks; k < ks+blockDim.x; ++k)
      {
        float rb = b[i+N*k];
        sum0 += rb * cb0[k-ks];
        sum1 += rb * cb1[k-ks];
      }
      __syncthreads();
    }

    a[i+N*j] = sum0;
    a[i+N*(j+1)] = sum1;
}

 __global__ void mmult4x4(float *a, float *b, float *c, int N)
{
    extern __shared__ float cb[];
    float *cb0 = &cb[0];
    float *cb1 = &cb[blockDim.x];
    float *cb2 = &cb[2*blockDim.x];
    float *cb3 = &cb[3*blockDim.x];

    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * 2;
    float sum0 = 0.0;
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;

    for (int ks = 0; ks < N; ks += blockDim.x) 
    {
      cb0[tx] = c[ks+tx+N*j];
      cb1[tx] = c[ks+tx+N*(j+1)];
      cb2[tx] = c[ks+tx+N*(j+2)];
      cb3[tx] = c[ks+tx+N*(j+3)];
      __syncthreads();

      for (int k = ks; k < ks+blockDim.x; ++k)
      {
        float rb = b[i+N*k];
        sum0 += rb * cb0[k-ks];
        sum1 += rb * cb1[k-ks];
        sum2 += rb * cb2[k-ks];
        sum3 += rb * cb3[k-ks];
      }
      __syncthreads();
    }

    a[i+N*j] = sum0;
    a[i+N*(j+1)] = sum1;
    a[i+N*(j+2)] = sum2;
    a[i+N*(j+3)] = sum3;
}





// init 
void minit(float *a, float *b, float *c, int N)
{
    int i, j;
    
    for (j = 0; j < N; j++)
    {
	for (i = 0; i < N; i++)
	{
	    a[i+N*j] = 0.0f;
	    b[i+N*j] = 1.0f;
	    c[i+N*j] = 1.0f;
	}
    }
}

// print
void mprint(float *a, int N, int M)
{
    int i, j;
    
    for (j = 0; j < M; j++)
    {
        for (i = 0; i < M; i++)
        {
            printf("%.2f ", a[i+N*j]);
        }
        printf("...\n");
    }
    printf("...\n");
}


int main(int argc, char* argv[])
{
    long int N = 4096;
    int T = 32;
    
    if (argc == 2)
        T = atoi(argv[1]);

    cudaEvent_t start, stop;
    float time, flop, mflops, gflops;
    
    float *a = (float *)malloc(N*N*sizeof(float));
    float *b = (float *)malloc(N*N*sizeof(float));
    float *c = (float *)malloc(N*N*sizeof(float));

    minit(a, b, c, N);
    
    // allocate device memory
    float *devPtrA, *devPtrB, *devPtrC;
    cudaMalloc((void**)&devPtrA, N*N*sizeof(float)); 
    cudaMalloc((void**)&devPtrB, N*N*sizeof(float)); 
    cudaMalloc((void**)&devPtrC, N*N*sizeof(float)); 

    // copu input arrays to the device meory    
    cudaMemcpy(devPtrB, b, N*N*sizeof(float),  cudaMemcpyHostToDevice); 
    cudaMemcpy(devPtrC, c, N*N*sizeof(float),  cudaMemcpyHostToDevice); 

    // define grid and thread block sizes
    dim3 threads(T);
    dim3 grid(N/T, N);

    // version 3 only
    //grid.x /= 2;

    // version 4 only
    //grid.y /= 2;

    // version 4x4 only
    //grid.y /= 4;


    printf("matrix %ix%i\n", N, N);
    printf("grid %ix%i\n", grid.x, grid.y);
    printf("block %ix%ix%i\n", threads.x, threads.y, threads.z);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // launch GPU kernel
    mmult1<<<grid, threads>>>(devPtrA, devPtrB, devPtrC, N);
    //mmult2<<<grid, threads, T*sizeof(float)>>>(devPtrA, devPtrB, devPtrC, N);
    //mmult3<<<grid, threads, T*sizeof(float)>>>(devPtrA, devPtrB, devPtrC, N);
    //mmult4<<<grid, threads, 2*T*sizeof(float)>>>(devPtrA, devPtrB, devPtrC, N);
    //mmult4x4<<<grid, threads, 4*T*sizeof(float)>>>(devPtrA, devPtrB, devPtrC, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // copy results to host
    cudaMemcpy(a, devPtrA, N*N*sizeof(float),  cudaMemcpyDeviceToHost); 

    //mprint(a, N, 5);
    
    // free device memory
    cudaFree(devPtrA); 
    cudaFree(devPtrB); 
    cudaFree(devPtrC); 
    
    free(a);
    free(b);
    free(c);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    time /= 1000.0f;  // convert from milliseconds to seconds
    flop = N*N*N*2.0f;
    mflops = flop / time / 1000000.0f;
    gflops = mflops / 1000.0f;
    printf("sec = %.2f   GFLOPS = %.3f\n", time, gflops);
}
