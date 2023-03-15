#include <assert.h>
#include <immintrin.h>
#include <string.h>

const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define A(i, j) A[(j)*lda + (i)]
#define B(i, j) B[(j)*lda + (i)]
#define C(i, j) C[(j)*lda + (i)]

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static inline void do_block_divide_simd(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each row i of A */
    for (int i = 0; i < M; i += 4)
    {
        // 4 x 4 parallel
        for (int j = 0; j < N; j += 4)
        {
            float *p_b_k0, *p_b_k1, *p_b_k2, *p_b_k3;

            p_b_k0 = &B(0, 0 + j);
            p_b_k1 = &B(0, 1 + j);
            p_b_k2 = &B(0, 2 + j);
            p_b_k3 = &B(0, 3 + j);

            __m128 c_0 = _mm_load_ps(&C(i, 0 + j)); // c_00, c_10, c_20, c_30
            __m128 c_1 = _mm_load_ps(&C(i, 1 + j)); // c_01, c_11, c_21, c_31
            __m128 c_2 = _mm_load_ps(&C(i, 2 + j)); // c_02, c_12, c_22, c_32
            __m128 c_3 = _mm_load_ps(&C(i, 3 + j)); // c_03, c_13, c_23, c_33

            for (int k = 0; k < K; k += 4)
            {
                __m128 a_0 = _mm_load_ps(&A(i, k)); // a_0k, a_1k, a_2k, a_3k
                __m128 a_1 = _mm_load_ps(&A(i, k + 1));
                __m128 a_2 = _mm_load_ps(&A(i, k + 2));
                __m128 a_3 = _mm_load_ps(&A(i, k + 3));

                // the position of c_00, c_10, c_20, c_30 is continuous
                __m128 b_00 = _mm_set1_ps(*p_b_k0); // b_1j, b_1j, b_1j, b_1j
                __m128 b_10 = _mm_set1_ps(*p_b_k1); // b_2j, b_2j, b_2j, b_2j
                __m128 b_20 = _mm_set1_ps(*p_b_k2); // b_3j, b_3j, b_3j, b_3j
                __m128 b_30 = _mm_set1_ps(*p_b_k3); // b_4j, b_4j, b_4j, b_4j

                __m128 b_01 = _mm_set1_ps(*(p_b_k0 + 1)); // b_1j, b_1j, b_1j, b_1j
                __m128 b_11 = _mm_set1_ps(*(p_b_k1 + 1)); // b_2j, b_2j, b_2j, b_2j
                __m128 b_21 = _mm_set1_ps(*(p_b_k2 + 1)); // b_3j, b_3j, b_3j, b_3j
                __m128 b_31 = _mm_set1_ps(*(p_b_k3 + 1)); // b_4j, b_4j, b_4j, b_4j

                __m128 b_02 = _mm_set1_ps(*(p_b_k0 + 2)); // b_1j, b_1j, b_1j, b_1j
                __m128 b_12 = _mm_set1_ps(*(p_b_k1 + 2)); // b_2j, b_2j, b_2j, b_2j
                __m128 b_22 = _mm_set1_ps(*(p_b_k2 + 2)); // b_3j, b_3j, b_3j, b_3j
                __m128 b_32 = _mm_set1_ps(*(p_b_k3 + 2)); // b_4j, b_4j, b_4j, b_4j

                __m128 b_03 = _mm_set1_ps(*(p_b_k0 + 3)); // b_1j, b_1j, b_1j, b_1j
                __m128 b_13 = _mm_set1_ps(*(p_b_k1 + 3)); // b_2j, b_2j, b_2j, b_2j
                __m128 b_23 = _mm_set1_ps(*(p_b_k2 + 3)); // b_3j, b_3j, b_3j, b_3j
                __m128 b_33 = _mm_set1_ps(*(p_b_k3 + 3)); // b_4j, b_4j, b_4j, b_4j

                p_b_k0 += 4;
                p_b_k1 += 4;
                p_b_k2 += 4;
                p_b_k3 += 4;

                c_0 = _mm_add_ps(c_0,
                                 _mm_add_ps(
                                     _mm_add_ps(_mm_mul_ps(a_0, b_00), _mm_mul_ps(a_1, b_01)),
                                     _mm_add_ps(_mm_mul_ps(a_2, b_02), _mm_mul_ps(a_3, b_03))));

                c_1 = _mm_add_ps(c_1,
                                 _mm_add_ps(
                                     _mm_add_ps(_mm_mul_ps(a_0, b_10), _mm_mul_ps(a_1, b_11)),
                                     _mm_add_ps(_mm_mul_ps(a_2, b_12), _mm_mul_ps(a_3, b_13))));

                c_2 = _mm_add_ps(c_2,
                                 _mm_add_ps(
                                     _mm_add_ps(_mm_mul_ps(a_0, b_20), _mm_mul_ps(a_1, b_21)),
                                     _mm_add_ps(_mm_mul_ps(a_2, b_22), _mm_mul_ps(a_3, b_23))));

                c_3 = _mm_add_ps(c_3,
                                 _mm_add_ps(
                                     _mm_add_ps(_mm_mul_ps(a_0, b_30), _mm_mul_ps(a_1, b_31)),
                                     _mm_add_ps(_mm_mul_ps(a_2, b_32), _mm_mul_ps(a_3, b_33))));

            }
            _mm_store_ps(&C(i, 0 + j), c_0);
            _mm_store_ps(&C(i, 1 + j), c_1);
            _mm_store_ps(&C(i, 2 + j), c_2);
            _mm_store_ps(&C(i, 3 + j), c_3);
        }
    }
}

void copy_memory_continuously(int lda, int M, int N, int K, float *A, float *B, float *C, float *ABC)
{
    // copy A
    for (int k = 0; k < K; k++)
    {
        float *p_ABC = ABC + k * BLOCK_SIZE;
        float *p_A = A + k * lda;
        memcpy(p_ABC, p_A, M * sizeof(float));
    }

    // copy B
    for (int j = 0; j < N; j++)
    {
        float *p_ABC = ABC + j * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE;
        float *p_B = B + j * lda;
        memcpy(p_ABC, p_B, K * sizeof(float));
    }

    // copy C
    for (int j = 0; j < N; j++)
    {
        float *p_ABC = ABC + j * BLOCK_SIZE + 2 * BLOCK_SIZE * BLOCK_SIZE;
        float *p_C = C + j * lda;
        memcpy(p_ABC, p_C, M * sizeof(float));
    }
}

void write_back(int lda, int M, int N, int K, float *C, float *ABC)
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
    /* For each block-row of A */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        /* For each block-column of B */
        for (int i = 0; i < lda; i += BLOCK_SIZE)
        {
            /* Accumulate block sgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                memset(ABC, 0, sizeof(ABC));

                /* copy the A\B\C matrix into continue memory */
                copy_memory_continuously(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda, ABC);

                /* perform individual block sgemm */
                int M_padding = (M % 4 == 0) ? M : ((M / 4 + 1) * 4);
                int N_padding = (N % 4 == 0) ? N : ((N / 4 + 1) * 4);
                int K_padding = (K % 4 == 0) ? K : ((K / 4 + 1) * 4);
                do_block_divide_simd(BLOCK_SIZE, M_padding, N_padding, K_padding, ABC, ABC + BLOCK_SIZE * BLOCK_SIZE, ABC + 2 * BLOCK_SIZE * BLOCK_SIZE);

                /* writeback the data */
                write_back(lda, M, N, K, C + i + j * lda, ABC + 2 * BLOCK_SIZE * BLOCK_SIZE);
            }
        }
    }
}