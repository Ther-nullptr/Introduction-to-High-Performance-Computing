
#include <stdio.h> 
#include <stdlib.h>

double sum(double* A, int N)
{ 
    int i;
    double s = 0.0f;
    
    for (i = 0; i < N; i++)
    {
        s += A[i]; 
    }

    return s;
}


int  main(int argc, char **argv)
{
    long long int i, N = 2097152;  // vector size
    double *A;
    double s = 0.0f;

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

    printf("Running CPU sum for %d elements\n", N);

    gettimeofday(&t1, NULL);
    msec1 = t1.tv_sec * 1000000 + t1.tv_usec;
    
    s = sum(A, N);  // call compute kernel

    gettimeofday(&t2, NULL);
    msec2 = t2.tv_sec * 1000000 + t2.tv_usec;
    
    printf("sum=%.2f\n", s);

    gflop = ((N-1) / (msec2 - msec1)) / 1000.0f;
    printf("sec = %f   GFLOPS = %.3f\n", (msec2-msec1)/1000000.0f, gflop);
  
    free(A);  // free allocated memory
}

