
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// a = b * c
void mmult(float *a, float *b, float *c, int N)
{
    int i, j, k;
    
    for (j = 0; j < N; j++)
	for (k = 0; k < N; k++)
	    for (i = 0; i < N; i++)
		a[i+j*N] += b[i+k*N]*c[k+j*N];
}

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
    int N = 1024;
    
    struct timeval t1, t2, ta, tb;
    long msec1, msec2;
    float flop, mflop, gflop;
    
    float *a = (float *)malloc(N*N*sizeof(float));
    float *b = (float *)malloc(N*N*sizeof(float));
    float *c = (float *)malloc(N*N*sizeof(float));

    minit(a, b, c, N);

    gettimeofday(&t1, NULL);
    msec1 = t1.tv_sec * 1000000 + t1.tv_usec;

    mmult(a, b, c, N);

    gettimeofday(&t2, NULL);
    msec2 = t2.tv_sec * 1000000 + t2.tv_usec;

    mprint(a, N, 5);
    
    free(a);
    free(b);
    free(c);

    msec2 -= msec1;
    flop = N*N*N*2.0f;
    mflop = flop / msec2;
    gflop = mflop / 1000.0f;
    printf("msec = %10ld   GFLOPS = %.3f\n", msec2, gflop);
}
