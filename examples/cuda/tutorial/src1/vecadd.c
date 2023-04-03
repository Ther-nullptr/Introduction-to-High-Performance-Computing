
#include  <stdio.h> 
#include <stdlib.h>

void vecAdd(int N, float* A, float* B, float* C)
{ 
    int i;
    
    for (i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i]; 
    }
}


int  main(int argc, char **argv)
{
    int i, N = 16384;  // default vector size
    float *A;
    float *B; 
    float *C; 

    if (argc > 1) N = atoi(argv[1]);  // get size of the vectors
    printf("Running CPU vecAdd for %i elements\n", N);

    A = (float*)malloc(N * sizeof(float));  // allocate memory 
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));

    for (i = 0; i < N; i++)  // generate random data
    {
        A[i] = (float)random();
        B[i] = (float)RAND_MAX - A[i];
    }
    
    vecAdd(N, A, B, C);  // call compute kernel
    
    for (i = 0; i < 10; i++)  // print out first 10 results
        printf("C[%i]=%.2f\n", i, C[i]);
  
    free(A);  // free allocated memory
    free(B);
    free(C);
}

