# 第二次作业 实验报告

## 实验目的

1. 实现高效的矩阵乘法
2. 掌握主要的体系结构优化手段
3. 掌握常见的并行计算工具

## 实验过程

### 1 baseline分析及其编译优化

> 注：为节约时间起见，在比较不同方法时暂时将`benchmark.sh`中的Timestep设置为16。

首先看`stencil-naive.c`中模板计算的循环实现：

```cpp
for (int t = 0; t < nt; ++t)
{
    cptr_t a0 = buffer[t % 2];
    ptr_t a1 = buffer[(t + 1) % 2];
#pragma omp parallel for schedule(dynamic)
    for (int z = z_start; z < z_end; ++z)
    {
        for (int y = y_start; y < y_end; ++y)
        {
            for (int x = x_start; x < x_end; ++x)
            {
                ...
            }
        }
    }
}
```

`#pragma`将最外层的语句并行化，并采用动态调度算法将循环迭代分配给不同的线程执行。测试结果如下：

| processes | size        | performance (Gflop/s)|
| --------- | ----------- | -------------------- |
| 1         | 256x256x256 | 1.600080             |
| 2         | 256x256x256 | 3.177852             |
| 4         | 256x256x256 | 6.085051             |
| 8         | 256x256x256 | 12.603111            |
| 16        | 256x256x256 | 20.932931            |

开启`-O3`编译优化之后的测试结果如下：

| processes | size        | performance (Gflop/s)|
| --------- | ----------- | -------------------- |
| 1         | 256x256x256 | 9.363892             |
| 2         | 256x256x256 | 18.574429            |
| 4         | 256x256x256 | 36.967220            |
| 8         | 256x256x256 | 69.608565            |
| 16        | 256x256x256 | 106.569054           |


### 2 串行优化

相比于传统的2D stencil模板计算，3D stencil模板计算中对相同数据的访问通常相隔太远，需要在每次阵列扫描时将阵列元素多次带入缓存。对此，我们借鉴...一文中tiling的思路，对数据进行分块重排。

![image.png](https://s2.loli.net/2023/03/22/omQpWNdOHCLfJVl.png)

如图所示，X（I）方向和Y（J）方向被划分为大小为XX（II）和YY（JJ）的小块。阴影区域代表在单个K循环迭代中需要进行修改的点，而周围三个没有阴影的区域代表单个K循环迭代中需要访问的点（注意如图所示展示的是7-stencil的计算，而本文中需要进行27-stencil的计算，因此边界的访问情况略有不同，详见代码）。

```cpp
for (int yy = y_start; yy < y_end; yy += BLOCK_Y)
{
    int FIXED_BLOCK_Y = min(BLOCK_Y, y_end - yy); // consider the edge situation
    for (int xx = x_start; xx < x_end; xx += BLOCK_X)
    {
        int FIXED_BLOCK_X = min(BLOCK_X, x_end - xx);

        // get the small block value to write
        ptr_t a1_block = a1 + z_start * ldx * ldy + yy * ldx + xx;

        // get the small block value to read
        ptr_t a0_block_Z = a0 + z_start * ldx * ldy + yy * ldx + xx;
        ptr_t a0_block_P = a0 + (z_start + 1) * ldx * ldy + yy * ldx + xx;
        ptr_t a0_block_N = a0 + (z_start - 1) * ldx * ldy + yy * ldx + xx;

        // loop inside block
        for (int z = z_start; z < z_end; ++z)
        {
            for (int y = 0; y < FIXED_BLOCK_Y; ++y)
            {
                for (int x = 0; x < FIXED_BLOCK_X; ++x)
                {
                    a1_block[INDEX_NEW(x, y, ldx)] = \
                        ALPHA_ZZZ * a0_block_Z[INDEX_NEW(x, y, ldx)] \
                    + ALPHA_NZZ * a0_block_Z[INDEX_NEW(x - 1, y, ldx)] + ALPHA_PZZ * a0_block_Z[INDEX_NEW(x + 1, y, ldx)] \
                    + ALPHA_ZNZ * a0_block_Z[INDEX_NEW(x, y - 1, ldx)] + ALPHA_ZPZ * a0_block_Z[INDEX_NEW(x, y + 1, ldx)] \
                    + ALPHA_ZZN * a0_block_N[INDEX_NEW(x, y, ldx)] + ALPHA_ZZP * a0_block_P[INDEX_NEW(x, y, ldx)] \
                    + ALPHA_NNZ * a0_block_Z[INDEX_NEW(x - 1, y - 1, ldx)] + ALPHA_PNZ * a0_block_Z[INDEX_NEW(x + 1, y - 1, ldx)] \
                    + ALPHA_NPZ * a0_block_Z[INDEX_NEW(x - 1, y + 1, ldx)] + ALPHA_PPZ * a0_block_Z[INDEX_NEW(x + 1, y + 1, ldx)] \
                    + ALPHA_NZN * a0_block_N[INDEX_NEW(x - 1, y, ldx)] + ALPHA_PZN * a0_block_N[INDEX_NEW(x + 1, y, ldx)] \
                    + ALPHA_NZP * a0_block_P[INDEX_NEW(x - 1, y, ldx)] + ALPHA_PZP * a0_block_P[INDEX_NEW(x + 1, y, ldx)] \
                    + ALPHA_ZNN * a0_block_N[INDEX_NEW(x, y - 1, ldx)] + ALPHA_ZPN * a0_block_N[INDEX_NEW(x, y + 1, ldx)] \
                    + ALPHA_ZNP * a0_block_P[INDEX_NEW(x, y - 1, ldx)] + ALPHA_ZPP * a0_block_P[INDEX_NEW(x, y + 1, ldx)] \
                    + ALPHA_NNN * a0_block_N[INDEX_NEW(x - 1, y - 1, ldx)] + ALPHA_PNN * a0_block_N[INDEX_NEW(x + 1, y - 1, ldx)] \
                    + ALPHA_NPN * a0_block_N[INDEX_NEW(x - 1, y + 1, ldx)] + ALPHA_PPN * a0_block_N[INDEX_NEW(x + 1, y + 1, ldx)] \
                    + ALPHA_NNP * a0_block_P[INDEX_NEW(x - 1, y - 1, ldx)] + ALPHA_PNP * a0_block_P[INDEX_NEW(x + 1, y - 1, ldx)] \
                    + ALPHA_NPP * a0_block_P[INDEX_NEW(x - 1, y + 1, ldx)] + ALPHA_PPP * a0_block_P[INDEX_NEW(x + 1, y + 1, ldx)];
                }
            }
            // update the pointer of block
            a1_block = a1_block + ldx * ldy;
            a0_block_N = a0_block_Z;
            a0_block_Z = a0_block_P;
            a0_block_P = a0_block_P + ldx * ldy;
        }
    }
}
```

不过颇感意外的是，这一优化并没有明显提升效率。在`OMP_NUM_THREAD=1`的情况下，串行执行效率并没有明显提升（`9.363892` vs `9.351314`）。推测可能是编译优化起到了类似的效果。

### 3 并行优化

## 参考文献

[^1] http://people.csail.mit.edu/skamil/projects/stencilprobe/
[^2] https://blog.csdn.net/weixin_43614211/article/details/122108753