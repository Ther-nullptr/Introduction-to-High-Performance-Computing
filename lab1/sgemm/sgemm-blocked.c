#include <assert.h>
#include <immintrin.h>

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
    for (int i = 0; i < M;)
    {
        if (M - i >= 4)
        {
            // 4 x 4 parallel
            for (int j = 0; j < N;)
            {
                if (N - j >= 4)
                {
                    float *p_b_k0, *p_b_k1, *p_b_k2, *p_b_k3;

                    p_b_k0 = &B(0, 0 + j);
                    p_b_k1 = &B(0, 1 + j);
                    p_b_k2 = &B(0, 2 + j);
                    p_b_k3 = &B(0, 3 + j);

                    for (int k = 0; k < K;)
                    {
                        if (K - k >= 4)
                        {
                            __m128 a_0 = _mm_loadu_ps(&A(i, k)); // a_0k, a_1k, a_2k, a_3k
                            __m128 a_1 = _mm_loadu_ps(&A(i, k + 1));
                            __m128 a_2 = _mm_loadu_ps(&A(i, k + 2));
                            __m128 a_3 = _mm_loadu_ps(&A(i, k + 3));

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

                            __m128 c_0 = _mm_loadu_ps(&C(i, 0 + j)); // c_00, c_10, c_20, c_30
                            __m128 c_1 = _mm_loadu_ps(&C(i, 1 + j)); // c_01, c_11, c_21, c_31
                            __m128 c_2 = _mm_loadu_ps(&C(i, 2 + j)); // c_02, c_12, c_22, c_32
                            __m128 c_3 = _mm_loadu_ps(&C(i, 3 + j)); // c_03, c_13, c_23, c_33

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

                            _mm_storeu_ps(&C(i, 0 + j), c_0);
                            _mm_storeu_ps(&C(i, 1 + j), c_1);
                            _mm_storeu_ps(&C(i, 2 + j), c_2);
                            _mm_storeu_ps(&C(i, 3 + j), c_3);

                            k += 4;
                        }
                        else
                        {
                            __m128 a = _mm_loadu_ps(&A(i, k)); // a_0k, a_1k, a_2k, a_3k

                            // the position of c_00, c_10, c_20, c_30 is continuous
                            __m128 b_0 = _mm_set1_ps(*p_b_k0); // b_1j, b_1j, b_1j, b_1j
                            __m128 b_1 = _mm_set1_ps(*p_b_k1); // b_2j, b_2j, b_2j, b_2j
                            __m128 b_2 = _mm_set1_ps(*p_b_k2); // b_3j, b_3j, b_3j, b_3j
                            __m128 b_3 = _mm_set1_ps(*p_b_k3); // b_4j, b_4j, b_4j, b_4j

                            p_b_k0++;
                            p_b_k1++;
                            p_b_k2++;
                            p_b_k3++;

                            __m128 c_0 = _mm_loadu_ps(&C(i, 0 + j)); // c_00, c_10, c_20, c_30
                            __m128 c_1 = _mm_loadu_ps(&C(i, 1 + j)); // c_01, c_11, c_21, c_31
                            __m128 c_2 = _mm_loadu_ps(&C(i, 2 + j)); // c_02, c_12, c_22, c_32
                            __m128 c_3 = _mm_loadu_ps(&C(i, 3 + j)); // c_03, c_13, c_23, c_33

                            c_0 = _mm_add_ps(c_0, _mm_mul_ps(a, b_0));
                            c_1 = _mm_add_ps(c_1, _mm_mul_ps(a, b_1));
                            c_2 = _mm_add_ps(c_2, _mm_mul_ps(a, b_2));
                            c_3 = _mm_add_ps(c_3, _mm_mul_ps(a, b_3));

                            _mm_storeu_ps(&C(i, 0 + j), c_0);
                            _mm_storeu_ps(&C(i, 1 + j), c_1);
                            _mm_storeu_ps(&C(i, 2 + j), c_2);
                            _mm_storeu_ps(&C(i, 3 + j), c_3);

                            k++;
                        }
                    }
                    j += 4;
                }
                else // only unrolling A
                {
                    /* Compute C(i,j) */
                    __m128 c_0 = _mm_loadu_ps(&C(i, 0 + j)); // c_00, c_10, c_20, c_30
                    float *p_b;
                    p_b = &B(0, 0 + j);

                    for (int k = 0; k < K; ++k)
                    {
                        __m128 a = _mm_loadu_ps(&A(i, k)); // a_0k, a_1k, a_2k, a_3k
                        __m128 b = _mm_set1_ps(*p_b);
                        p_b++;

                        __m128 c = _mm_loadu_ps(&C(i, 0 + j)); // c_00, c_10, c_20, c_30
                        c = _mm_add_ps(c, _mm_mul_ps(a, b));

                        _mm_storeu_ps(&C(i, 0 + j), c);
                    }
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
                do_block_divide_simd(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}
