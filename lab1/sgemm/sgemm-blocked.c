#include <assert.h>
#include <emmintrin.h>
#include <string.h>

const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 8
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

            __m128 a_0, a_1, a_2, a_3;
            __m128 b_0, b_1, b_2, b_3;
            __m128 tmp1, tmp2, tmp3, tmp4;

            register float b00, b01, b02, b03;
            register float b10, b11, b12, b13;
            register float b20, b21, b22, b23;
            register float b30, b31, b32, b33;

            for (int k = 0; k < K; k += 4)
            {
                b00 = *p_b_k0;
                b01 = *(p_b_k0 + 1);
                b02 = *(p_b_k0 + 2);
                b03 = *(p_b_k0 + 3);

                b10 = *p_b_k1;
                b11 = *(p_b_k1 + 1);
                b12 = *(p_b_k1 + 2);
                b13 = *(p_b_k1 + 3);

                b20 = *p_b_k2;
                b21 = *(p_b_k2 + 1);
                b22 = *(p_b_k2 + 2);
                b23 = *(p_b_k2 + 3);

                b30 = *p_b_k3;
                b31 = *(p_b_k3 + 1);
                b32 = *(p_b_k3 + 2);
                b33 = *(p_b_k3 + 3);

                a_0 = _mm_load_ps(&A(4 * k, i)); // a_0k, a_1k, a_2k, a_3k
                a_1 = _mm_load_ps(&A(4 * k + 4, i));
                a_2 = _mm_load_ps(&A(4 * k + 8, i));
                a_3 = _mm_load_ps(&A(4 * k + 12, i));

                // the position of c_00, c_10, c_20, c_30 is continuous
                b_0 = _mm_set1_ps(b00); // b_1j, b_1j, b_1j, b_1j
                b_1 = _mm_set1_ps(b01); // b_1j, b_1j, b_1j, b_1j
                b_2 = _mm_set1_ps(b02); // b_1j, b_1j, b_1j, b_1j
                b_3 = _mm_set1_ps(b03); // b_1j, b_1j, b_1j, b_1j

                tmp1 = _mm_mul_ps(a_0, b_0);
                tmp2 = _mm_mul_ps(a_1, b_1);
                tmp3 = _mm_mul_ps(a_2, b_2);
                tmp4 = _mm_mul_ps(a_3, b_3);

                c_0 = _mm_add_ps(c_0,
                                 _mm_add_ps(
                                     _mm_add_ps(tmp1, tmp2),
                                     _mm_add_ps(tmp3, tmp4)));

                b_0 = _mm_set1_ps(b10); // b_2j, b_2j, b_2j, b_2j
                b_1 = _mm_set1_ps(b11); // b_2j, b_2j, b_2j, b_2j
                b_2 = _mm_set1_ps(b12); // b_2j, b_2j, b_2j, b_2j
                b_3 = _mm_set1_ps(b13); // b_2j, b_2j, b_2j, b_2j

                tmp1 = _mm_mul_ps(a_0, b_0);
                tmp2 = _mm_mul_ps(a_1, b_1);
                tmp3 = _mm_mul_ps(a_2, b_2);
                tmp4 = _mm_mul_ps(a_3, b_3);

                c_1 = _mm_add_ps(c_1,
                                 _mm_add_ps(
                                     _mm_add_ps(tmp1, tmp2),
                                     _mm_add_ps(tmp3, tmp4)));

                b_0 = _mm_set1_ps(b20); // b_3j, b_3j, b_3j, b_3j
                b_1 = _mm_set1_ps(b21); // b_3j, b_3j, b_3j, b_3j
                b_2 = _mm_set1_ps(b22); // b_3j, b_3j, b_3j, b_3j
                b_3 = _mm_set1_ps(b23); // b_3j, b_3j, b_3j, b_3j

                tmp1 = _mm_mul_ps(a_0, b_0);
                tmp2 = _mm_mul_ps(a_1, b_1);
                tmp3 = _mm_mul_ps(a_2, b_2);
                tmp4 = _mm_mul_ps(a_3, b_3);

                c_2 = _mm_add_ps(c_2,
                                 _mm_add_ps(
                                     _mm_add_ps(tmp1, tmp2),
                                     _mm_add_ps(tmp3, tmp4)));

                b_0 = _mm_set1_ps(b30); // b_4j, b_4j, b_4j, b_4j
                b_1 = _mm_set1_ps(b31); // b_4j, b_4j, b_4j, b_4j
                b_2 = _mm_set1_ps(b32); // b_4j, b_4j, b_4j, b_4j
                b_3 = _mm_set1_ps(b33); // b_4j, b_4j, b_4j, b_4j

                tmp1 = _mm_mul_ps(a_0, b_0);
                tmp2 = _mm_mul_ps(a_1, b_1);
                tmp3 = _mm_mul_ps(a_2, b_2);
                tmp4 = _mm_mul_ps(a_3, b_3);

                c_3 = _mm_add_ps(c_3,
                                 _mm_add_ps(
                                     _mm_add_ps(tmp1, tmp2),
                                     _mm_add_ps(tmp3, tmp4)));

                p_b_k0 += 4;
                p_b_k1 += 4;
                p_b_k2 += 4;
                p_b_k3 += 4; 
            }

            _mm_store_ps(&C(i, 0 + j), c_0);
            _mm_store_ps(&C(i, 1 + j), c_1);
            _mm_store_ps(&C(i, 2 + j), c_2);
            _mm_store_ps(&C(i, 3 + j), c_3);
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

static inline void transpose_A(int lda, int M, int N, int K, float* ABC, float* temp_array)
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
                do_block_divide_simd(BLOCK_SIZE, M_padding, N_padding, K_padding, ABC, ABC + BLOCK_SIZE * BLOCK_SIZE, ABC + 2 * BLOCK_SIZE * BLOCK_SIZE);

                /* writeback the data */
                write_back(lda, M, N, K, C + i + j * lda, ABC + 2 * BLOCK_SIZE * BLOCK_SIZE);
            }
        }
        
    }
}