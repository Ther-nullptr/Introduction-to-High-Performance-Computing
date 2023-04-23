# 第三次作业 实验报告

## 实验目的

1. 在GPU上实现高效的模板计算
2. 掌握CUDA编程

## 工程说明

本项目实现了GPU模板编程的多种优化的方式，存放于若干个git branch上：
```bash
$ git branch -v

```

## 实验过程

### 0 集群基本概况介绍

首先，编写一个小程序`get_gpu_info.cu`获取集群中GPU的相关信息：

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM number: " << devProp.multiProcessorCount << std::endl;
    std::cout << "shared memory per block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "max memory pitch: " << devProp.memPitch / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "sm architecture: " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "max block num per grid: " << devProp.maxGridSize[0] << " x " << devProp.maxGridSize[1] << " x " << devProp.maxGridSize[2] << std::endl;
    std::cout << "max thread dim per block: " << devProp.maxThreadsDim[0] << " x " << devProp.maxThreadsDim[1] << " x " << devProp.maxThreadsDim[2] << std::endl;
}
```

输出结果如下：

```bash
$ srun -n 1 get_gpu_info
GPU device 0: NVIDIA A40
SM number: 84
shared memory per block: 48 KB
max threads per block: 1024
max memory pitch: 2048 MB
sm architecture: 8.6
max block num per grid: 2147483647 x 65535 x 65535
max thread dim per block: 1024 x 1024 x 64
```

可见集群服务器使用的服务器为NVIDIA A40，SM架构为`sm_86`(Ampere架构)，每一个grid中有`2147483647 x 65535 x 65535`个block，每一个block中有`1024 x 1024 x 64`个thread，在之后我们可以据此进行一定的编译优化。

### 1 编译优化

直接编译，结果如下：

| block size | 256x256x256 | 512x512x512 | 1024x1024x1024 |
| ---------- | ----------- | ----------- | -------------- |
|   Gflops   |  478.132433 |  479.915134 |  480.479321    |

在makefile中添加编译优化选项：

```make
OPT = -O3 -arch=compute_86 -code=compute_86 --use_fast_math --extra-device-vectorization  
```

不过让人失望的是，编译优化并没有起到加速的作用：

| block size | 256x256x256 | 512x512x512 | 1024x1024x1024 |
| ---------- | ----------- | ----------- | -------------- |
|   Gflops   |  477.644443 |  479.823823 |  480.339162    |


### 2 分块策略优化

首先观察`stencil-naive.cu`中的代码：

```cpp
#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt)
{
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    dim3 grid_size(ceiling(nx, BLOCK_X), ceiling(ny, BLOCK_Y), ceiling(nz, BLOCK_Z));
    dim3 block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
    for (int t = 0; t < nt; ++t)
    {
        stencil_27_naive_kernel_1step<<<grid_size, block_size>>>(
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz,
            grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}
```

文章指出以下配置grid和block大小的原则：

1. 每个block中的线程数量是线程束大小（32）的倍数（更严格的来说，应该保持一个block的最内层维数`block.x`是32的倍数）。这是因为从硬件角度看，block是一维线程束的集合。在线程块中线程被组织为一维布局，每32个连续线程组成一个线程束。
2. 每个block至少应该有128或256个线程。

据此调整`grid_size`和`block_size`的大小：

| block size | 256x256x256 | 384x384x384 | 512x512x512 |
| ---------- | ----------- | ----------- | ----------- |
|   8,8,8    |  477.644443 |  479.823823 |  480.339162 |
|   32,4,4   |  476.651004 |  478.525451 |  479.139808 |
|   128,2,2  |  476.056412 |  478.132814 |  478.828899 |
|   4,4,4    |  477.464908 |  478.475096 |  478.794737 |
|   4,4,2    |  478.603388 |  479.216224 |  479.607900 |
|   4,2,2    |  236.320797 |  236.648849 |  236.606128 |

遗憾的是，上述建议并没有太大的指导意义，推测是因为服务器中的GPU相比原书中的GPU性能更佳，在线程调度方面做得更好。只有在单个block中的线程数少于32时，性能才会有明显下降。这里可以看到，默认的8,8,8配置几乎仍为最优解。


### 3 2.5D优化和计算重用

本节参考了[^3]的做法。

在做进一步的优化之前，有必要对源代码进行分析：

```cpp
__global__ void stencil_27_naive_kernel_1step(cptr_t in, ptr_t out, int nx, int ny, int nz, int halo_x, int halo_y, int halo_z)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;
    int ldxx = BLOCK_X + halo_x * 2;
    int ldyy = BLOCK_Y + halo_y * 2; // 分块后张量的大小

    __shared__ double local_g[(BLOCK_X + halo_x * 2) * (BLOCK_Y + halo_y * 2) * (BLOCK_Z + halo_z * 2)]; // 设置一个与块大小相等的shared memory
    if (tx < nx && ty < ny && tz < nz)
    {
        int ldx = nx + halo_x * 2;
        int ldy = ny + halo_y * 2;
        int offset_x, offset_y, offset_z;
        int x = tx + halo_x;
        int y = ty + halo_y;
        int z = tz + halo_z;
        int local_x = threadIdx.x + halo_x;
        int local_y = threadIdx.y + halo_y;
        int local_z = threadIdx.z + halo_z;
        local_g[INDEX(local_x, local_y, local_z, ldxx, ldyy)] = in[INDEX(x, y, z, ldx, ldy)];

        offset_x = (local_x == 1) ? -1 : (local_x == BLOCK_X) ? 1 : 0;
        offset_y = (local_y == 1) ? -1 : (local_y == BLOCK_Y) ? 1 : 0;
        offset_z = (local_z == 1) ? -1 : (local_z == BLOCK_Z) ? 1 : 0;

        if (offset_x)
            local_g[INDEX(local_x + offset_x, local_y, local_z, ldxx, ldyy)] = in[INDEX(x + offset_x, y, z, ldx, ldy)];
        if (offset_y)
            local_g[INDEX(local_x, local_y + offset_y, local_z, ldxx, ldyy)] = in[INDEX(x, y + offset_y, z, ldx, ldy)];
        if (offset_z)
            local_g[INDEX(local_x, local_y, local_z + offset_z, ldxx, ldyy)] = in[INDEX(x, y, z + offset_z, ldx, ldy)];
        if (offset_x && offset_y)
            local_g[INDEX(local_x + offset_x, local_y + offset_y, local_z, ldxx, ldyy)] = in[INDEX(x + offset_x, y + offset_y, z, ldx, ldy)];
        if (offset_x && offset_z)
            local_g[INDEX(local_x + offset_x, local_y, local_z + offset_z, ldxx, ldyy)] = in[INDEX(x + offset_x, y, z + offset_z, ldx, ldy)];
        if (offset_y && offset_z)
            local_g[INDEX(local_x, local_y + offset_y, local_z + offset_z, ldxx, ldyy)] = in[INDEX(x, y + offset_y, z + offset_z, ldx, ldy)];
        if (offset_x && offset_y && offset_z)
            local_g[INDEX(local_x + offset_x, local_y + offset_y, local_z + offset_z, ldxx, ldyy)] = in[INDEX(x + offset_x, y + offset_y, z + offset_z, ldx, ldy)];

        __syncthreads();

        out[INDEX(x, y, z, ldx, ldy)] = ALPHA_ZZZ * local_g[INDEX(local_x, local_y, local_z, ldxx, ldyy)] + ... + ALPHA_PPP * local_g[INDEX(local_x + 1, local_y + 1, local_z + 1, ldxx, ldyy)];
    }
}
```

首先使用blockdim, blockidx和threadidx，将分块场景下的index与实际index一一对应；然后开辟一块大小为`(BLOCK_X + halo_x * 2) * (BLOCK_Y + halo_y * 2) * (BLOCK_Z + halo_z * 2)`的shared memory，供一个block内的线程计算使用。注意在shared memory中load完毕之后需要`__syncthreads()`。

在使用shared memory时，可能存在一个潜在的问题：有的线程需要读取halo区，有的不需要，这样额外的分支可能会造成线程之间的负载不均衡。为了探究这一不均衡的分布对代码性能的影响，我们在不考虑结果正确性的情况下，把所有的分支语句删除，观察代码性能：

| block size | 256x256x256 | 384x384x384 | 512x512x512    |
| ---------- | ----------- | ----------- | -------------- |
|  w branch  |  477.644443 |  479.823823 |  480.339162    |
|  w/o branch|  477.924273 |  479.883670 |  480.356028    |

总体来说，这种线程间的负载不均衡并没有带来太大开销，所以暂时搁置优化。

shared memory在GPU中的地位类似于一级缓存，不过它是可编程的。实际上在还有比shared memory更快的存储方式：register。我们尝试将register用于stencil GPU计算。*Efficient 3D stencil computations using CUDA*一文为我们提供了一种思路：只对X,Y方向进行分块，Z方向上的计算串行完成，不过在串行完成时，大量计算可以重用。原文的伪码如下：

![image.png](https://s2.loli.net/2023/04/09/spEnNf725GtURzL.png)

不过原文也指出，这段代码并没有处理边界情况，我们需要结合实际状况进行适当改写。由于此处提供的buffer已经包含了halo区（也就是说，一个256x256x256的stencil，其实际大小是258x258x258），shared data同理。坐标逻辑如下图所示：

![33aca171b809435657de33157479c0c.jpg](https://s2.loli.net/2023/04/09/zou6BfxTqsYwylD.jpg)

因此在取用数据时，要先定义一个偏移量：

```cpp
const int in_out_shamt = (gx_with_halo + (gy_with_halo) * (nx_with_halo));         // 全局数组偏移量，避免重复的乘加运算
const int thread_shamt = (tx_with_halo + (ty_with_halo) * (thread_dim_with_halo)); // 线程数组偏移量，避免重复的乘加运算
```

每一个block负责的区域如图所示：

![03fbb1ff6bef853847460d706d1f385.jpg](https://s2.loli.net/2023/04/09/48IgYrU1NcWz6no.jpg)

每一次复制数据到shared memory中时，只需要处理一个z平面中的数据，将三维问题变成了一个二维问题：

```cpp
__device__ __forceinline__ void readBlockAndHalo(cptr_t in, ptr_t shm, uint k, const uint in_out_shamt, const uint thread_shamt, const uint nx, const uint ny, const uint tx, const uint ty)
{
    __syncthreads();
    const uint thread_dim_with_halo = (blockDim.x + 2);
    int offset_x, offset_y;
    shm[thread_shamt] = in[in_out_shamt + k * (nx * ny)];
    offset_x = (tx == 1) ? -1 : (tx == blockDim.x) ? 1 : 0;
    offset_y = (ty == 1) ? -1 : (ty == blockDim.y) ? 1 : 0;
    if (offset_x)
    {
        shm[thread_shamt + offset_x] = in[in_out_shamt + k * (nx * ny) + offset_x];
    }
    if (offset_y)
    {
        shm[thread_shamt + offset_y * thread_dim_with_halo] = in[in_out_shamt + k * (nx * ny) + offset_y * nx];
    }
    if (offset_x && offset_y)
    {
        shm[thread_shamt + offset_x + offset_y * thread_dim_with_halo] = in[in_out_shamt + k * (nx * ny) + offset_x + offset_y * nx];
    }
}
```

每次向shared memory读取完毕之后，当前线程周围的9个数据会被写入寄存器：
```cpp
__syncthreads();
r1 = shm[r1_idx], r2 = shm[r2_idx], r3 = shm[r3_idx];
r4 = shm[r4_idx], r5 = shm[r5_idx], r6 = shm[r6_idx];
r7 = shm[r7_idx], r8 = shm[r8_idx], r9 = shm[r9_idx];
```

第k个输入平面的瓦片被用作三个输出平面的模版计算的一部分：k-1、k和k+1。计算按平面进行。获得的部分模版结果被累积到寄存器t1和t2中。一旦来自三个相邻平面的部分结果被积累起来，结果就被存储在输出数组中：

```cpp
if (gx < nx && gy < ny) // 超出计算范围的线程无需工作
{
    out += in_out_shamt;
    readBlockAndHalo(in, shm, 0, in_out_shamt, thread_shamt, nx_with_halo, ny_with_halo, tx_with_halo, ty_with_halo);

    register double r1, r2, r3; // upup, upmid, updown
    register double r4, r5, r6; // midup, midmid, middown
    register double r7, r8, r9; // downup, downmid, downdown

    register double t1, t2;

    __syncthreads();
    r1 = shm[r1_idx], r2 = shm[r2_idx], r3 = shm[r3_idx];
    r4 = shm[r4_idx], r5 = shm[r5_idx], r6 = shm[r6_idx];
    r7 = shm[r7_idx], r8 = shm[r8_idx], r9 = shm[r9_idx];

    readBlockAndHalo(in, shm, 1, in_out_shamt, thread_shamt, nx_with_halo, ny_with_halo, tx_with_halo, ty_with_halo);
    t1 = ALPHA_NNN * r1 + ALPHA_NZN * r2 + ALPHA_NPN * r3 + ALPHA_ZNN * r4 + ALPHA_ZZN * r5 + ALPHA_ZPN * r6 + ALPHA_PNN * r7 + ALPHA_PZN * r8 + ALPHA_PPN * r9;

    __syncthreads();
    r1 = shm[r1_idx], r2 = shm[r2_idx], r3 = shm[r3_idx];
    r4 = shm[r4_idx], r5 = shm[r5_idx], r6 = shm[r6_idx];
    r7 = shm[r7_idx], r8 = shm[r8_idx], r9 = shm[r9_idx];
    readBlockAndHalo(in, shm, 2, in_out_shamt, thread_shamt, nx_with_halo, ny_with_halo, tx_with_halo, ty_with_halo);

    t2 = ALPHA_NNN * r1 + ALPHA_NZN * r2 + ALPHA_NPN * r3 + ALPHA_ZNN * r4 + ALPHA_ZZN * r5 + ALPHA_ZPN * r6 + ALPHA_PNN * r7 + ALPHA_PZN * r8 + ALPHA_PPN * r9;

    t1 += ALPHA_NNZ * r1 + ALPHA_NZZ * r2 + ALPHA_NPZ * r3 + ALPHA_ZNZ * r4 + ALPHA_ZZZ * r5 + ALPHA_ZPZ * r6 + ALPHA_PNZ * r7 + ALPHA_PZZ * r8 + ALPHA_PPZ * r9;

    for (uint k = 3; k < nz + 2; k++)
    {
        __syncthreads();
        r1 = shm[r1_idx], r2 = shm[r2_idx], r3 = shm[r3_idx];
        r4 = shm[r4_idx], r5 = shm[r5_idx], r6 = shm[r6_idx];
        r7 = shm[r7_idx], r8 = shm[r8_idx], r9 = shm[r9_idx];

        readBlockAndHalo(in, shm, k, in_out_shamt, thread_shamt, nx_with_halo, ny_with_halo, tx_with_halo, ty_with_halo);

        out += nx_with_halo * ny_with_halo;

        out[0] = t1 + ALPHA_NNP * r1 + ALPHA_NZP * r2 + ALPHA_NPP * r3 + ALPHA_ZNP * r4 + ALPHA_ZZP * r5 + ALPHA_ZPP * r6 + ALPHA_PNP * r7 + ALPHA_PZP * r8 + ALPHA_PPP * r9;

        t1 = t2 + ALPHA_NNZ * r1 + ALPHA_NZZ * r2 + ALPHA_NPZ * r3 + ALPHA_ZNZ * r4 + ALPHA_ZZZ * r5 + ALPHA_ZPZ * r6 + ALPHA_PNZ * r7 + ALPHA_PZZ * r8 + ALPHA_PPZ * r9;
        t2 = ALPHA_NNN * r1 + ALPHA_NZN * r2 + ALPHA_NPN * r3 + ALPHA_ZNN * r4 + ALPHA_ZZN * r5 + ALPHA_ZPN * r6 + ALPHA_PNN * r7 + ALPHA_PZN * r8 + ALPHA_PPN * r9;
    }

    __syncthreads();
    r1 = shm[r1_idx], r2 = shm[r2_idx], r3 = shm[r3_idx];
    r4 = shm[r4_idx], r5 = shm[r5_idx], r6 = shm[r6_idx];
    r7 = shm[r7_idx], r8 = shm[r8_idx], r9 = shm[r9_idx];
    out += nx_with_halo * ny_with_halo;
    out[0] = t1 + ALPHA_NNP * r1 + ALPHA_NZP * r2 + ALPHA_NPP * r3 + ALPHA_ZNP * r4 + ALPHA_ZZP * r5 + ALPHA_ZPP * r6 + ALPHA_PNP * r7 + ALPHA_PZP * r8 + ALPHA_PPP * r9;
}
```

在通常的共享存储器中存储三个输入平面的方法中，数值要么在计算前直接从共享存储器加载到寄存器中（每个格点27条加载指令），要么算术运算必须使用共享存储器中的操作数。而本方法每个格点只需使用9条加载指令。

实验结果如下：

| block size | 256x256x256 | 384x384x384 | 512x512x512    |
| ---------- | ----------- | ----------- | -------------- |
|   8,8      |  444.711849 |  470.274611 |  468.810584    |
|   16,8     |  415.873894 |  469.122651 |  468.049605    |
|   32,8     |  363.057206 |  468.509651 |  449.508509    |
|   8,4      |  441.591196 |  469.428652 |  468.452767    |
|   32,32    |  360.759357 |  406.851275 |  362.065074    |

让人倍感遗憾的是，本方法并没有起到应有的效果。

## 参考文献

[^1] [On Optimizing Complex Stencils on GPUs](https://ieeexplore.ieee.org/document/8820786)

[^2] [Diamond Tiling: Tiling Techniques to Maximize Parallelism for Stencil Computations](https://ieeexplore.ieee.org/document/7582549)

[^3] [Efficient 3D stencil computations using CUDA](https://www.sciencedirect.com/science/article/pii/S016781911300094X)

[^4] [NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)

[^5] [CUDA C编程权威指南](https://github.com/sallenkey-wei/cuda-handbook/)