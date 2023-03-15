#include <assert.h>
#include <string.h>
#include <immintrin.h>

const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 256
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define A(i, j) A[(j)*lda + (i)]
#define B(i, j) B[(j)*lda + (i)]
#define C(i, j) C[(j)*lda + (i)]
#define temp_array(i, j) temp_array[(j)*lda + (i)]

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static inline void do_block_divide_4x4(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each row i of A */
    for (int i = 0; i < M; i+=4)
    {

        // 4 x 4 parallel
        for (int j = 0; j < N; j+=4)
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
                a_0k = A(4 * k, i);
                a_1k = A(4 * k + 1, i);
                a_2k = A(4 * k + 2, i);
                a_3k = A(4 * k + 3, i);

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
        }
    }
}

static inline void copy_memory_continuously(int lda, int M, int N, int K, float *A, float *B, float *C, float *ABC)
{
    // copy A
    float *p_ABC = ABC;
    float *p_A = A;
    for (int k = 0; k < K; k++)
    {
        memcpy(p_ABC, p_A, M * sizeof(float));
        p_ABC += BLOCK_SIZE;
        p_A += lda;
    }

    // copy B
    p_ABC = ABC + BLOCK_SIZE * BLOCK_SIZE;
    float *p_B = B;
    for (int j = 0; j < N; j++)
    {
        memcpy(p_ABC, p_B, K * sizeof(float));
        p_ABC += BLOCK_SIZE;
        p_B += lda;
    }

    // copy C
    p_ABC = ABC + 2 * BLOCK_SIZE * BLOCK_SIZE;
    float *p_C = C;
    for (int j = 0; j < N; j++)
    {
        memcpy(p_ABC, p_C, M * sizeof(float));
        p_ABC += BLOCK_SIZE;
        p_C += lda;
    }
}

static inline void transpose_A(int lda, int M, int N, int K, float *ABC, float *temp_array)
{
    // transpose A in ABC
    for (int i = 0; i < M; i += 4)
    {
        for (int k = 0; k < K; k += 4)
        {
            __m128 temp = _mm_load_ps(ABC + k * BLOCK_SIZE + i);
            _mm_store_ps(&temp_array(4 * k, i), temp);
            temp = _mm_load_ps(ABC + (k + 1) * BLOCK_SIZE + i);
            _mm_store_ps(&temp_array(4 * k + 4, i), temp);
            temp = _mm_load_ps(ABC + (k + 2) * BLOCK_SIZE + i);
            _mm_store_ps(&temp_array(4 * k + 8, i), temp);
            temp = _mm_load_ps(ABC + (k + 3) * BLOCK_SIZE + i);
            _mm_store_ps(&temp_array(4 * k + 12, i), temp);
        }
    }
    memcpy(ABC, temp_array, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
}

static inline void write_back(int lda, int M, int N, int K, float *C, float *ABC)
{
    for (int j = 0; j < N; j++)
    {
        float *p_ABC = ABC + j * BLOCK_SIZE;
        float *p_C = C + j * lda;
        memcpy(p_C, p_ABC, M * sizeof(float));
    }
}

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_sgemm(int lda, float *A, float *B, float *C)
{
    float ABC[3 * BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(64))); // A,B,C blocks in same matrix
    float temp_array[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(64)));
    /* For each block-row of A */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        /* For each block-column of B */
        int N = min(BLOCK_SIZE, lda - j);
        int N_padding = (N % 4 == 0) ? N : ((N / 4 + 1) * 4);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            int K = min(BLOCK_SIZE, lda - k);
            int K_padding = (K % 4 == 0) ? K : ((K / 4 + 1) * 4);
            /* Accumulate block sgemms into block of C */
            for (int i = 0; i < lda; i += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min(BLOCK_SIZE, lda - i);
                int M_padding = (M % 4 == 0) ? M : ((M / 4 + 1) * 4);

                if (lda > BLOCK_SIZE)
                {
                    memset(ABC, 0, sizeof(ABC));
                }

                /* copy the A\B\C matrix into continue memory */
                copy_memory_continuously(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda, ABC);

                transpose_A(BLOCK_SIZE, M_padding, N_padding, K_padding, ABC, temp_array);
                /* perform individual block sgemm */
                do_block_divide_4x4(BLOCK_SIZE, M_padding, N_padding, K_padding, ABC, ABC + BLOCK_SIZE * BLOCK_SIZE, ABC + 2 * BLOCK_SIZE * BLOCK_SIZE);

                /* writeback the data */
                write_back(lda, M, N, K, C + i + j * lda, ABC + 2 * BLOCK_SIZE * BLOCK_SIZE);
            }
        }
    }
}
