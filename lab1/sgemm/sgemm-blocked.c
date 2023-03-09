#include <assert.h>

const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. 
*/
static inline void do_block_divide_unrolling_a(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
        /* For each row i of A */
        for (int i = 0; i < M;)
        {
            if (M - i >= 4)
            {
                /* Compute C(i,j) */
                float cij_0 = C[i + j * lda];
                float cij_1 = C[i + 1 + j * lda];
                float cij_2 = C[i + 2 + j * lda];
                float cij_3 = C[i + 3 + j * lda];

                for (int k = 0; k < K; ++k)
                {
                    float bkj = B[k + j * lda];
                    cij_0 += bkj * A[i + k * lda];
                    cij_1 += bkj * A[i + 1 + k * lda];
                    cij_2 += bkj * A[i + 2 + k * lda];
                    cij_3 += bkj * A[i + 3 + k * lda];
                }

                C[i + j * lda] = cij_0;
                C[i + 1 + j * lda] = cij_1;
                C[i + 2 + j * lda] = cij_2;
                C[i + 3 + j * lda] = cij_3;
                
                i += 4;
            }
            else
            {
                float cij = C[i + j * lda];

                for (int k = 0; k < K; ++k)
                {
                    float bkj = B[k + j * lda];
                    cij += bkj * A[i + k * lda];
                }

                C[i + j * lda] = cij;
                i += 1;
            }
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
                do_block_divide_unrolling_a(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

