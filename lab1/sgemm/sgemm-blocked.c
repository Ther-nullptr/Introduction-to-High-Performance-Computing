#include <assert.h>

const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define A(i,j) A[ (j)*lda + (i) ]
#define B(i,j) B[ (j)*lda + (i) ]
#define C(i,j) C[ (j)*lda + (i) ]

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. 
*/
static inline void do_block_divide_4x4(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each row i of A */
    for (int i = 0; i < M;)
    {
        if (M - i >= 4)
        {
            // 4 x 4 parallel
            for (int j = 0; j < N;)
            {
                if (N - j >= 4)
                {
                    /* Compute C(i,j) */
                    register float c_00 = 0., c_01 = 0., c_02 = 0., c_03 = 0.,
                        c_10 = 0., c_11 = 0., c_12 = 0., c_13 = 0.,
                        c_20 = 0., c_21 = 0., c_22 = 0., c_23 = 0.,
                        c_30 = 0., c_31 = 0., c_32 = 0., c_33 = 0.;
                    register float a_0k, a_1k, a_2k, a_3k;
                    register float b_k0, b_k1, b_k2, b_k3;

                    c_00 = C(i, j);
                    c_01 = C(i, j + 1);
                    c_02 = C(i, j + 2);
                    c_03 = C(i, j + 3);

                    c_10 = C(i + 1, j);
                    c_11 = C(i + 1, j + 1);
                    c_12 = C(i + 1, j + 2);
                    c_13 = C(i + 1, j + 3);

                    c_20 = C(i + 2, j);
                    c_21 = C(i + 2, j + 1);
                    c_22 = C(i + 2, j + 2);
                    c_23 = C(i + 2, j + 3);

                    c_30 = C(i + 3, j);
                    c_31 = C(i + 3, j + 1);
                    c_32 = C(i + 3, j + 2);
                    c_33 = C(i + 3, j + 3);

                    float *p_b_k0, *p_b_k1, *p_b_k2, *p_b_k3;

                    p_b_k0 = &B(0, 0 + j);
                    p_b_k1 = &B(0, 1 + j);
                    p_b_k2 = &B(0, 2 + j);
                    p_b_k3 = &B(0, 3 + j);

                    for (int k = 0; k < K; ++k)
                    {
                        a_0k = A(0 + i, k);
                        a_1k = A(1 + i, k);
                        a_2k = A(2 + i, k);
                        a_3k = A(3 + i, k);

                        b_k0 = *p_b_k0;
                        p_b_k0++;
                        b_k1 = *p_b_k1;
                        p_b_k1++;
                        b_k2 = *p_b_k2;
                        p_b_k2++;
                        b_k3 = *p_b_k3;
                        p_b_k3++;

                        c_00 += a_0k * b_k0;
                        c_01 += a_0k * b_k1;
                        c_02 += a_0k * b_k2;
                        c_03 += a_0k * b_k3;

                        c_10 += a_1k * b_k0;
                        c_11 += a_1k * b_k1;
                        c_12 += a_1k * b_k2;
                        c_13 += a_1k * b_k3;

                        c_20 += a_2k * b_k0;
                        c_21 += a_2k * b_k1;
                        c_22 += a_2k * b_k2;
                        c_23 += a_2k * b_k3;

                        c_30 += a_3k * b_k0;
                        c_31 += a_3k * b_k1;
                        c_32 += a_3k * b_k2;
                        c_33 += a_3k * b_k3;
                    }

                    C(i, j) = c_00;
                    C(i, j + 1) = c_01;
                    C(i, j + 2) = c_02;
                    C(i, j + 3) = c_03;

                    C(i + 1, j) = c_10;
                    C(i + 1, j + 1) = c_11;
                    C(i + 1, j + 2) = c_12;
                    C(i + 1, j + 3) = c_13;

                    C(i + 2, j) = c_20;
                    C(i + 2, j + 1) = c_21;
                    C(i + 2, j + 2) = c_22;
                    C(i + 2, j + 3) = c_23;

                    C(i + 3, j) = c_30;
                    C(i + 3, j + 1) = c_31;
                    C(i + 3, j + 2) = c_32;
                    C(i + 3, j + 3) = c_33;

                    j += 4;
                }
                else // only unrolling A
                {
                    /* Compute C(i,j) */
                    register float cij_0 = C[i + j * lda];
                    register float cij_1 = C[i + 1 + j * lda];
                    register float cij_2 = C[i + 2 + j * lda];
                    register float cij_3 = C[i + 3 + j * lda];

                    for (int k = 0; k < K; ++k)
                    {
                        register float bkj = B[k + j * lda];
                        cij_0 += bkj * A[i + k * lda];
                        cij_1 += bkj * A[i + 1 + k * lda];
                        cij_2 += bkj * A[i + 2 + k * lda];
                        cij_3 += bkj * A[i + 3 + k * lda];
                    }

                    C[i + j * lda] = cij_0;
                    C[i + 1 + j * lda] = cij_1;
                    C[i + 2 + j * lda] = cij_2;
                    C[i + 3 + j * lda] = cij_3;
                    
                    j += 1;
                }
            }
            i += 4;
        }
        else
        {
            for (int j = 0; j < N;)
            {
                if (N - j >= 4) // only unrolling B
                {
                    /* Compute C(i,j) */
                    register float cij_0 = C[i + j * lda];
                    register float cij_1 = C[i + (j + 1) * lda];
                    register float cij_2 = C[i + (j + 2) * lda];
                    register float cij_3 = C[i + (j + 3) * lda];

                    for (int k = 0; k < K; ++k)
                    {
                        register float aik = A[i + k * lda];
                        cij_0 += aik * B[k + j * lda];
                        cij_1 += aik * B[k + (j + 1) * lda];
                        cij_2 += aik * B[k + (j + 2) * lda];
                        cij_3 += aik * B[k + (j + 3) * lda];
                    }

                    C[i + j * lda] = cij_0;
                    C[i + (j + 1) * lda] = cij_1;
                    C[i + (j + 2) * lda] = cij_2;
                    C[i + (j + 3) * lda] = cij_3;

                    j += 4;
                }
                else
                {
                    register float cij = C[i + j * lda];
                    for (int k = 0; k < K; ++k)
                    {
                        register float aik = A[i + k * lda];
                        cij += aik * B[k + j * lda];
                    }

                    C[i + j * lda] = cij;

                    j += 1;
                }
            }
            i += 1;
        }
    }
}

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. 
*/
void square_sgemm(int lda, float *A, float *B, float *C)
{
    /* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE)
    {
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
        {
            /* Accumulate block sgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);

                /* Perform individual block sgemm */
                do_block_divide_4x4(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

