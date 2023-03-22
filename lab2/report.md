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

### 2 串行优化

相比于传统的2D stencil模板计算，3D stencil模板计算中对相同数据的访问通常相隔太远，需要在每次阵列扫描时将阵列元素多次带入缓存。对此，我们借鉴...一文中tiling的思路，对数据进行分块重排。

![image.png](https://s2.loli.net/2023/03/22/omQpWNdOHCLfJVl.png)

如图所示，X（I）方向和Y（J）方向被划分为大小为XX（II）和YY（JJ）的小块。阴影区域代表在单个K循环迭代中需要进行修改的点，而周围三个没有阴影的区域代表单个K循环迭代中需要访问的点（注意如图所示展示的是7-stencil的计算，而本文中需要进行27-stencil的计算，因此边界的访问情况略有不同，详见代码）。

## 参考文献

[^1] http://people.csail.mit.edu/skamil/projects/stencilprobe/
[^2] https://blog.csdn.net/weixin_43614211/article/details/122108753