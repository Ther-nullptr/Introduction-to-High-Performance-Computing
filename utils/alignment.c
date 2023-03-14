#include <stdio.h>

int main()
{
    // float A[10] __attribute__((aligned(64))) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // float B[10] __attribute__((aligned(64))) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // printf("%p ", A);
    // printf("%p ", B);
    float A[10];
    printf("%d", A[0]);
}