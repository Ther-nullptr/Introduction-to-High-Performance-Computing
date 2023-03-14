#include <stdio.h>

int main()
{
    float A[5] = {0, 1, 2, 3, 4};
    float *B = A + 3;
    printf("%f", B[1]);
}