const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#if !defined(UNROLLING_NUM)
#define UNROLLING_NUM 4
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. 
*/
static void do_block(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
    {
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
            /* Compute C(i,j) */
            float cij = C[i + j * lda];
            for (int k = 0; k < K; ++k)
                cij += A[i + k * lda] * B[k + j * lda]; // C[j][i] = B[j][k] * A[k][i] 
            C[i + j * lda] = cij;
        }
    }
}

static void do_block_divide_a(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
    {
        /* For each column j of B */
        for (int j = 0; j < N; j += UNROLLING_NUM)
        {
            /* Compute C(i,j) */
            float cij = C[i + j * lda];
            for (int k = 0; k < K; ++k)
            {
                float aik = A[i + k * lda];
                for (int l = 0; l < UNROLLING_NUM; ++l)
                {
                    C[i + (j + l) * lda] += aik * B[k + (j + l) * lda];
                }
            }
        }
    }
}

static void do_block_divide_b(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each row i of A */
    for (int i = 0; i < M; i += UNROLLING_NUM)
    {
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
            /* Compute C(i,j) */
            float cij = C[i + j * lda];
            for (int k = 0; k < K; ++k)
            {
                float bkj = B[k + j * lda];
                for (int l = 0; l < UNROLLING_NUM; ++l)
                {
                    C[i + l + j * lda] += A[i + l + k * lda] * bkj;
                }
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
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

void square_sgemm_with_unrolling_a(int lda, float *A, float *B, float *C)
{
    assert(BLOCK_SIZE % UNROLLING_NUM == 0);
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
                do_block_divide_a(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

void square_sgemm_with_unrolling_b(int lda, float *A, float *B, float *C)
{
    assert(BLOCK_SIZE % UNROLLING_NUM == 0);
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
                do_block_divide_b(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}