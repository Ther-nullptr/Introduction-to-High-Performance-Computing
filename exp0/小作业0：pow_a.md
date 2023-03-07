# 小作业0：pow_a

## 源代码

`openmp_pow.cpp`:

```cpp
void pow_a(int *a, int *b, int n, int m) {
    // TODO: 使用 omp parallel for 并行这个循环
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int x = 1;
        for (int j = 0; j < m; j++)
            x *= a[i];
        b[i] = x;
    }
}
```

`mpi_pow.cpp`:

```cpp
void pow_a(int *a, int *b, int n, int m, int comm_sz /* 总进程数 */) {
    // TODO: 对这个进程拥有的数据计算 b[i] = a[i]^m
        int local_n = n / comm_sz;
        for (int i = 0; i < local_n; i++) {
            int x = 1;
        for (int j = 0; j < m; j++)
            x *= a[i];
        b[i] = x;
    }
}
```

## OpenMp分析

n=112000,m=100000时运行时间和加速比如下：

| 线程数 | 运行时间（us） | 加速比 |
| ------ | -------------- | ------ |
| 1      | 14023957       | 1      |
| 7      | 2018816        | 6.947  |
| 14     | 1012641        | 13.849 |
| 28     | 515024         | 27.230 |

## MPI分析

n=112000,m=100000时运行时间和加速比如下：

| 线程数 | 运行时间（us） | 加速比 |
| ------ | -------------- | ------ |
| 1x1    | 14005594       | 1      |
| 1x7    | 2022772        | 6.924  |
| 1x14   | 1004525        | 13.943 |
| 1x28   | 502121         | 27.893 |
| 2x28   |                |        |