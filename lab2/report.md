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

| processes | size        | preprocess time (s) | computation time (s) | performance (Gflop/s) |
| --------- | ----------- | ------------------- | -------------------- | --------------------- |
| 1         | 256x256x256 | 0.000001            | 9.653929             | 1.473709              |
| 2         | 256x256x256 | 0.000004            | 18.884874            | 0.753358              |
| 4         | 256x256x256 | 0.000057            | 19.304891            | 0.736968              |
| 8         | 256x256x256 | 0.000032            | 18.893131            | 0.753029              |

开启`-O3`编译优化之后的测试结果如下：

| processes | size        | preprocess time (s) | computation time (s) | performance (Gflop/s) |
| --------- | ----------- | ------------------- | -------------------- | --------------------- |
| 1         | 256x256x256 | 0.000002            | 2.570787             | 5.534134              |
| 2         | 256x256x256 | 0.000006            | 4.354849             | 3.266951              |
| 4         | 256x256x256 | 0.000039            | 4.356507             | 3.265708              |
| 8         | 256x256x256 | 0.000062            | 4.356327             | 3.265843              |

### 2 OpenMPI并行化选项的优化



## 参考文献

[^1] http://people.csail.mit.edu/skamil/projects/stencilprobe/
[^2] https://blog.csdn.net/weixin_43614211/article/details/122108753