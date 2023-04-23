#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define BLOCK_X 104
#define BLOCK_Y 8
#define BLOCK_Z 8

const char *version_name = "Optimized version";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type)
{
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info)
{
}

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

__global__ void stencil_27_optimized_kernel_1step(cptr_t in, ptr_t out,
                                                  int nx, int ny, int nz,
                                                  int halo_x, int halo_y, int halo_z)
{
    extern __shared__ double shm[];

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    const uint tx_with_halo = tx + halo_x;
    const uint ty_with_halo = ty + halo_y;

    const uint gx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint gy = blockIdx.y * blockDim.y + threadIdx.y;

    const uint gx_with_halo = gx + halo_x;
    const uint gy_with_halo = gy + halo_y;

    const uint nx_with_halo = nx + halo_x * 2;
    const uint ny_with_halo = ny + halo_y * 2;

    const uint thread_dim_with_halo = blockDim.x + halo_x * 2;

    const int in_out_shamt = (gx_with_halo + (gy_with_halo) * (nx_with_halo));         // 全局数组偏移量，避免重复的乘加运算
    const int thread_shamt = (tx_with_halo + (ty_with_halo) * (thread_dim_with_halo)); // 线程数组偏移量，避免重复的乘加运算

    const int r1_idx = thread_shamt - 1 - thread_dim_with_halo, r2_idx = thread_shamt - 1, r3_idx = thread_shamt - 1 + thread_dim_with_halo;
    const int r4_idx = thread_shamt - thread_dim_with_halo, r5_idx = thread_shamt, r6_idx = thread_shamt + thread_dim_with_halo;
    const int r7_idx = thread_shamt + 1 - thread_dim_with_halo, r8_idx = thread_shamt + 1, r9_idx = thread_shamt + 1 + thread_dim_with_halo;

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
}

inline int ceiling(int num, int den)
{
    return (num - 1) / den + 1;
}

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt)
{
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    dim3 grid_size(ceiling(nx, BLOCK_X), ceiling(ny, BLOCK_Y));
    dim3 block_size(BLOCK_X, BLOCK_Y);
    printf("grid.x %d grid.y %d \n", grid_size.x, grid_size.y);
    printf("block.x %d block.y %d \n", block_size.x, block_size.y);
    for (int t = 0; t < nt; ++t)
    {
        stencil_27_optimized_kernel_1step<<<grid_size, block_size, sizeof(double) * (BLOCK_X + 2) * (BLOCK_Y + 2)>>>(
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz,
            grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}