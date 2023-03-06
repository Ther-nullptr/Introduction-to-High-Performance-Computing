# 第一次作业 实验报告

## 实验目的

1. 实现高效的矩阵乘法
2. 掌握主要的体系结构优化手段
3. 掌握常见的并行计算工具

## 实验过程

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

相比于基本方法，这里使用了矩阵分块的方法，将矩阵分成大小为`BLOCK_SIZE`的小块，然后分别计算每个小块的乘积，最后将结果相加。矩阵分块的优势在于，矩阵越小，越容易被装入cache，从而提高cache命中率，提高运算效率。

将`BLOCK_SIZE`视为超参，改变大小，计算结果如下：

### 二、循环展开（Unrolling）

借鉴...中的思路，我们采用循环展开（loop unrolling）的方法，每次迭代计算`UNROLLING_NUM`个元素，这样在内层循环中可以实现`UNROLLING_NUM`次数据复用。

我们假设程序中的二维矩阵是按照列优先存储的，则原始的矩阵乘法可以写作：

```cpp
C[i][j] = A[i][k] * B[k][j]
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

这里我们为了操作简便和提高效率，我们在分好块的矩阵中实现循环展开，且规定矩阵块的大小一定可以整除`UNROLLING_NUM`，这样就无需要考虑边界条件。

### 三、寄存器优化（Register Blocking）

### 四、单指令多数据（SIMD）

## 参考文献

[1^] [深入浅出GPU优化系列：GEMM优化（一）](https://zhuanlan.zhihu.com/p/435908830)
[2^] [通用矩阵乘（GEMM）优化与卷积计算](https://zhuanlan.zhihu.com/p/66958390)