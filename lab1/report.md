# 第一次作业 实验报告

## 实验目的

1. 实现高效的矩阵乘法
2. 掌握主要的体系结构优化手段
3. 掌握常见的并行计算工具

## 实验过程

### 〇、上限下限分析

对项目程序进行编译，分别运行`benchmark-naive`和`benchmark-blas`，可视化结果如下：

* naive

![image-20230308205952073](https://s2.loli.net/2023/03/13/8Tq9Szw4xfHFhvd.png)

* blas

  ![image-20230308210109178](https://s2.loli.net/2023/03/13/eWItJPvq7rUd4Yb.png)

据此可以得出以下结论：

1. 相比于`benchmark-naive`，`benchmark-blas`可以达到接近100倍的加速。
2. 不同的加速方法可能在不同的矩阵尺寸上有不同的表现。例如`benchmark-naive`在矩阵尺寸较小时Gflops更高，而`benchmark-blas`在矩阵尺寸较大时Gflops更高；而`benchmark-blas`则相反。
3. `benchmark-blas`的Gflops变化有一定的周期性。当矩阵尺寸为4的整数倍时，Gflops相对较大；反之Gflops较小。

### 一、benchmark分析及其编译优化

原本代码中的benchmark实现如下：

```cpp
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
            {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

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
```

为了在不改动代码的前提下提升Gflops，可以先利用编译选项来完成一部分优化。在`makefile`中为`benchmark-blocked`添加不同的编译选项，结果如下：

![image-20230308214927085](https://s2.loli.net/2023/03/13/TynitPGmgZ9FOfc.png)

可以看出，当编译优化等级为`-O3`或`-Ofast`时，程序的性能得到了明显的提升。由于`-Ofast`的优化较为激进，为了兼顾速度和稳健性，之后统一采取`-O3`优化。

相比于基本方法，这里使用了矩阵分块的方法，将矩阵分成大小为`BLOCK_SIZE`的小块，然后分别计算每个小块的乘积，最后将结果相加。矩阵分块的优势在于，矩阵越小，越容易被装入cache，从而提高cache命中率，提高运算效率。

将`BLOCK_SIZE`视为超参，改变大小，计算结果如下：

![image-20230308220825243](https://s2.loli.net/2023/03/13/DMTFAXKvws85ErQ.png)

可见`BLOCK_SIZE`既不能太大也不能太小，之后的实验均采用默认值64。

### 二、循环展开（Unrolling）

借鉴...中的思路，我们采用循环展开（loop unrolling）的方法，每次迭代计算`UNROLLING_NUM`个元素，这样在内层循环中可以实现`UNROLLING_NUM`次数据复用。

我们假设程序中的二维矩阵是按照列优先存储的，则原始的矩阵乘法可以写作：

```cpp
C[i][j] = A[i][k] * B[k][j]
// C[i][j] = C[i + j * N]
```

循环展开后：

```cpp
C[i][j + 0] = A[i][k] * B[k][j] 
C[i][j + 1] = A[i][k] * B[k][j + 1]
// ...
C[i][j + UNROLLING_NUM - 1] = A[i][k] * B[k][j + UNROLLING_NUM - 1]
```

或者：

```cpp
C[i + 0][j] = A[i + 0][k] * B[k][j]
C[i + 1][j] = A[i + 1][k] * B[k][j]
// ...
C[i + UNROLLING_NUM - 1][j] = A[i + UNROLLING_NUM - 1][k] * B[k][j]
```

不过这会导致一个问题：当矩阵的边长不为4的整数倍时，可能会导致越界问题。为了解决这一问题，当目标矩阵的边长为`N`时，我们按照矩阵边长为`(N/4+1)*4`进行内存分配，并对边界进行补0。考虑到N远大于4，这一更改所造成的额外开销可以忽略不计。更改如下：

```diff
- buf = (float *)malloc(3 * nmax * nmax * sizeof(float));
+ int nmax_new = (nmax / 4 + 1) * 4;
+ buf = (float *)malloc(3 * nmax_new * nmax_new * sizeof(float));

- float *A = buf + 0;
- float *B = A + nmax * nmax;
- float *C = B + nmax * nmax;
+ float *A = buf + 0;
+ float *B = A + nmax_new * nmax_new;
+ float *C = B + nmax_new * nmax_new;
```

分别对矩阵`A`和矩阵`B`进行循环展开：

```cpp
// strategy 1
static inline void do_block_divide_unrolling_a(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
    {
        /* For each column j of B */
        for (int j = 0; j < N; j += 4)
        {
            /* Compute C(i,j) */
            float cij_0 = C[i + j * lda]; // C(i, j)
            float cij_1 = C[i + (j + 1) * lda]; // C(i, j + 1)
            float cij_2 = C[i + (j + 2) * lda]; // C(i, j + 2)
            float cij_3 = C[i + (j + 3) * lda]; // C(i, j + 3)

            for (int k = 0; k < K; ++k)
            {
                float aik = A[i + k * lda]; // A(i, k)
                cij_0 += aik * B[k + j * lda]; // C(i, j) += A(i, k) * B(k, j)
                cij_1 += aik * B[k + (j + 1) * lda]; // C(i, j + 1) += A(i, k) * B(k, j + 1)
                cij_2 += aik * B[k + (j + 2) * lda]; // C(i, j + 2) += A(i, k) * B(k, j + 2)
                cij_3 += aik * B[k + (j + 3) * lda]; // C(i, j + 3) += A(i, k) * B(k, j + 3)
            } 

            C[i + j * lda] = cij_0;
            C[i + (j + 1) * lda] = cij_1;
            C[i + (j + 2) * lda] = cij_2;
            C[i + (j + 3) * lda] = cij_3;
        }
    }
}

// strategy 2
static inline void do_block_divide_unrolling_b(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
    {
        /* For each column j of B */
        for (int j = 0; j < N; j += 4)
        {
            /* Compute C(i,j) */
            float cij_0 = C[i + j * lda];
            float cij_1 = C[i + (j + 1) * lda];
            float cij_2 = C[i + (j + 2) * lda];
            float cij_3 = C[i + (j + 3) * lda];

            for (int k = 0; k < K; ++k)
            {
                float aik = A[i + k * lda];
                cij_0 += aik * B[k + j * lda];
                cij_1 += aik * B[k + (j + 1) * lda];
                cij_2 += aik * B[k + (j + 2) * lda];
                cij_3 += aik * B[k + (j + 3) * lda];
            }

            C[i + j * lda] = cij_0;
            C[i + (j + 1) * lda] = cij_1;
            C[i + (j + 2) * lda] = cij_2;
            C[i + (j + 3) * lda] = cij_3;
        }
    }
}
```

计算结果如下（`BLOCK_SIZE=64`）：

![image-20230313102036789](https://s2.loli.net/2023/03/13/gYjd5zcS2TbixZ4.png)

可以看出loop unrolling对于计算速度的提升有着很明显的效果。这里值得关注的有两件事：

1. 相比于对A矩阵进行循环展开，对B矩阵进行循环展开可以获得更佳的效果，可能是因为此时的展开方法可以更有效地提升cache命中率。
2. 循环展开方法的速度呈现出周期性的波动。这可能与CPU内部的cache机制有关系。

### 三、寄存器优化（Register Blocking）

为了进一步展开循环，同时也为下一步编写SIMD代码做铺垫，我们试图在一个循环之内完成一个4X4大小的矩阵乘法。我们使用C中的关键字`register`，请求编译器尽可能的将变量存在CPU内部寄存器中，而不是通过内存寻址访问，以提高效率。在一次内部循环中，将会一次性计算4x4=16个结果，相当于做了一次更加彻底的loop unrolling。同时，代码也对不方便进行loop unrolling的边界情况进行了处理。

代码如下：

```c++
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
```

计算结果如下：

![image-20230313102306781](https://s2.loli.net/2023/03/13/zR9atUnIVgBfoKE.png)

### 四、单指令多数据（SIMD）

即Single Instruction, Multiple Data，一条指令操作多个数据．是CPU基本指令集的扩展。该扩展可以用于32bit或64bit的寄存器进行拆分，使得一个指令可以同时操作多个数据。

Intel的SIMD指令集有MMX、SSE、AVX、AVX2、AVX-512等，而ARM的SIMD指令集有NEON、SVE等。这里我们使用Intel的SSE指令集进行SIMD编程。

SSE拥有8个128位长的寄存器（XMM0~XMM7），每个寄存器可以支持4个单精度浮点数同时计算。我们分别用基础方法以及SSE指令集实现4x4乘法，代码如下：

```cpp
#include <emmintrin.h>
#include <stdio.h>
#include <time.h>

static inline void matrix_multiply_sse(float *A, float *B, float *C)
{
    __m128 row1 = _mm_load_ps(&B[0]);    // load the first row of B into an SSE register
    __m128 row2 = _mm_load_ps(&B[4]);    // load the second row of B into an SSE register
    __m128 row3 = _mm_load_ps(&B[8]);    // load the third row of B into an SSE register
    __m128 row4 = _mm_load_ps(&B[12]);   // load the fourth row of B into an SSE register
    
    for (int i = 0; i < 4; i++) {
        int offset = 4*i;
        __m128 brod1 = _mm_set1_ps(A[offset + 0]);   // load the ith element of the first row of A into an SSE register and broadcast it
        __m128 brod2 = _mm_set1_ps(A[offset + 1]);   // load the ith element of the second row of A into an SSE register and broadcast it
        __m128 brod3 = _mm_set1_ps(A[offset + 2]);   // load the ith element of the third row of A into an SSE register and broadcast it
        __m128 brod4 = _mm_set1_ps(A[offset + 3]);   // load the ith element of the fourth row of A into an SSE register and broadcast it
        
        __m128 row = _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(brod1, row1),
                _mm_mul_ps(brod2, row2)),
            _mm_add_ps(
                _mm_mul_ps(brod3, row3),
                _mm_mul_ps(brod4, row4)));    // multiply the ith row of A with all rows of B, and add the results element-wise
        
        _mm_store_ps(&C[4*i], row);    // store the result row into the ith row of C
    }
}

static inline void matrix_multiply_naive(float *A, float *B, float *C)
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) {
                sum += A[4*i + k] * B[4*k + j];
            }
            C[4*i + j] = sum;
        }
    }
}

int main()
{
    float A[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float B[16] = {
        1, 5, 9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15,
        4, 8, 12, 16
    };
    float C[16] = {0};
    
    // to test the time 
    clock_t start_f = clock();
    for (int i = 0; i < 1000000; i++)
    {
        matrix_multiply_sse(A, B, C);
    }
    clock_t end_f = clock();
    printf("time: %f ", (double)(end_f - start_f) / CLOCKS_PER_SEC);

    printf("\n");

    clock_t start_n = clock();
    for (int i = 0; i < 1000000; i++)
    {
        matrix_multiply_naive(A, B, C);
    }
    clock_t end_n = clock();
    printf("time: %f ", (double)(end_n - start_n) / CLOCKS_PER_SEC);
}
```

以上运算的时间开销如下：

| 优化 | naive      | SSE        |
| ---- | ---------- | ---------- |
| 无   | 0.065315 s | 0.223347 s |
| -Og  | 0.013843 s | 0.106731 s |
| -O1  | 0.007443 s | 0.096685 s |

可见向量化的计算相比于普通的运算方法，速度有很大的提升。对此，我们使用SSE指令优化之前的计算（同时考虑边界情况）：

```cpp
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
```

计算结果如下：

![image-20230313102739195](https://s2.loli.net/2023/03/13/nqzWUTHb6tjwsxl.png)

## 总结

![image-20230313103810821](https://s2.loli.net/2023/03/13/4Kv8uIE2Z3xeQFJ.png)

## 特别说明

为方便数据搜集和处理，在`benchmark-test.c`和`benchmark.c`文件中添加了日志功能：

```cpp
// before loop
FILE *fp;
char filename[100];
time_t t = time(NULL);
struct tm tm = *localtime(&t);
sprintf(filename, "data_%d-%d-%d_%d:%d:%d.csv", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
fp = fopen(filename, "a");
fprintf(fp, "Size,Gflops,iter,seconds\n");

// in loop
fprintf(fp, "%d,%.3g,%d,%.3f\n", n, Gflops_s, n_iterations, seconds);

// after loop
fclose(fp);
```

## 参考文献

[^1]: [深入浅出GPU优化系列：GEMM优化（一）](https://zhuanlan.zhihu.com/p/435908830)
[^2]: [通用矩阵乘（GEMM）优化与卷积计算](https://zhuanlan.zhihu.com/p/66958390)
[^3]: [Instructions函数对照表：01 mmintrin.h与MMX指令集](https://www.cnblogs.com/zyl910/archive/2012/07/19/intrin01_mmx.html)
[^4]: [Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
[^5]: [SIMD指令集](https://zhuanlan.zhihu.com/p/31271788)
[^6]: [SIMD](https://www.cnblogs.com/zyl910/archive/2012/04/26/md00.html)
[^7]: [如何加速矩阵乘法——优化GEMM (CPU单线程篇)](https://renzibei.com/2021/06/30/optimize-gemm/)
[^8]: [SIMD简介](https://zhuanlan.zhihu.com/p/55327037)
[^9]: [GCC编译优化和调试选项](http://walkerdu.com/2020/04/22/gcc_optimization/#Ofast)