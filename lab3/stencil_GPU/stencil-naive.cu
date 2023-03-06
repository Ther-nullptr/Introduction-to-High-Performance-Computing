#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8

const char* version_name = "Optimized version";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}



__global__ void stencil_27_naive_kernel_1step(cptr_t in, ptr_t out, \
                                int nx, int ny, int nz, \
                                int halo_x, int halo_y, int halo_z) {
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y  + blockDim.y * blockIdx.y;
    int tz = threadIdx.z  + blockDim.z * blockIdx.z;
    int ldxx = BLOCK_X+2;
    int ldyy = BLOCK_Y+2;


    __shared__ double local_g[(BLOCK_X+2)*(BLOCK_Y+2)*(BLOCK_Z+2)];
    if(tx < nx && ty < ny && tz < nz) {
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


        if (offset_x)   local_g[INDEX(local_x+offset_x, local_y, local_z, ldxx, ldyy)] = in[INDEX(x+offset_x, y, z, ldx, ldy)];
        if (offset_y)   local_g[INDEX(local_x, local_y+offset_y, local_z, ldxx, ldyy)] = in[INDEX(x, y+offset_y, z, ldx, ldy)];
        if (offset_z)   local_g[INDEX(local_x, local_y, local_z+offset_z, ldxx, ldyy)] = in[INDEX(x, y, z+offset_z, ldx, ldy)];
        if (offset_x && offset_y)   local_g[INDEX(local_x+offset_x, local_y+offset_y, local_z, ldxx, ldyy)] = in[INDEX(x+offset_x, y+offset_y, z, ldx, ldy)];
        if (offset_x && offset_z)   local_g[INDEX(local_x+offset_x, local_y, local_z+offset_z, ldxx, ldyy)] = in[INDEX(x+offset_x, y, z+offset_z, ldx, ldy)];
        if (offset_y && offset_z)   local_g[INDEX(local_x, local_y+offset_y, local_z+offset_z, ldxx, ldyy)] = in[INDEX(x, y+offset_y, z+offset_z, ldx, ldy)];
        if (offset_x && offset_y && offset_z)   local_g[INDEX(local_x+offset_x, local_y+offset_y, local_z+offset_z, ldxx, ldyy)] = in[INDEX(x+offset_x, y+offset_y, z+offset_z, ldx, ldy)];
        
        __syncthreads();

        out[INDEX(x, y, z, ldx, ldy)] \
            = ALPHA_ZZZ * local_g[INDEX(local_x, local_y, local_z, ldxx, ldyy)] \
            + ALPHA_NZZ * local_g[INDEX(local_x-1, local_y, local_z, ldxx, ldyy)] \
            + ALPHA_PZZ * local_g[INDEX(local_x+1, local_y, local_z, ldxx, ldyy)] \
            + ALPHA_ZNZ * local_g[INDEX(local_x, local_y-1, local_z, ldxx, ldyy)] \
            + ALPHA_ZPZ * local_g[INDEX(local_x, local_y+1, local_z, ldxx, ldyy)] \
            + ALPHA_ZZN * local_g[INDEX(local_x, local_y, local_z-1, ldxx, ldyy)] \
            + ALPHA_ZZP * local_g[INDEX(local_x, local_y, local_z+1, ldxx, ldyy)] \
            + ALPHA_NNZ * local_g[INDEX(local_x-1, local_y-1, local_z, ldxx, ldyy)] \
            + ALPHA_PNZ * local_g[INDEX(local_x+1, local_y-1, local_z, ldxx, ldyy)] \
            + ALPHA_NPZ * local_g[INDEX(local_x-1, local_y+1, local_z, ldxx, ldyy)] \
            + ALPHA_PPZ * local_g[INDEX(local_x+1, local_y+1, local_z, ldxx, ldyy)] \
            + ALPHA_NZN * local_g[INDEX(local_x-1, local_y, local_z-1, ldxx, ldyy)] \
            + ALPHA_PZN * local_g[INDEX(local_x+1, local_y, local_z-1, ldxx, ldyy)] \
            + ALPHA_NZP * local_g[INDEX(local_x-1, local_y, local_z+1, ldxx, ldyy)] \
            + ALPHA_PZP * local_g[INDEX(local_x+1, local_y, local_z+1, ldxx, ldyy)] \
            + ALPHA_ZNN * local_g[INDEX(local_x, local_y-1, local_z-1, ldxx, ldyy)] \
            + ALPHA_ZPN * local_g[INDEX(local_x, local_y+1, local_z-1, ldxx, ldyy)] \
            + ALPHA_ZNP * local_g[INDEX(local_x, local_y-1, local_z+1, ldxx, ldyy)] \
            + ALPHA_ZPP * local_g[INDEX(local_x, local_y+1, local_z+1, ldxx, ldyy)] \
            + ALPHA_NNN * local_g[INDEX(local_x-1, local_y-1, local_z-1, ldxx, ldyy)] \
            + ALPHA_PNN * local_g[INDEX(local_x+1, local_y-1, local_z-1, ldxx, ldyy)] \
            + ALPHA_NPN * local_g[INDEX(local_x-1, local_y+1, local_z-1, ldxx, ldyy)] \
            + ALPHA_PPN * local_g[INDEX(local_x+1, local_y+1, local_z-1, ldxx, ldyy)] \
            + ALPHA_NNP * local_g[INDEX(local_x-1, local_y-1, local_z+1, ldxx, ldyy)] \
            + ALPHA_PNP * local_g[INDEX(local_x+1, local_y-1, local_z+1, ldxx, ldyy)] \
            + ALPHA_NPP * local_g[INDEX(local_x-1, local_y+1, local_z+1, ldxx, ldyy)] \
            + ALPHA_PPP * local_g[INDEX(local_x+1, local_y+1, local_z+1, ldxx, ldyy)];
    }
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

//#define BLOCK_SIZE 9

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    dim3 grid_size (ceiling(nx, BLOCK_X), ceiling(ny, BLOCK_Y), ceiling(nz, BLOCK_Z));
    dim3 block_size (BLOCK_X, BLOCK_Y, BLOCK_Z);
    for(int t = 0; t < nt; ++t) {
        stencil_27_naive_kernel_1step<<<grid_size, block_size>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}