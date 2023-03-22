#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
const char *version_name = "Optimized version";
#include "common.h"

#define BLOCK_X 256
#define BLOCK_Y 256
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define INDEX_NEW(xx, yy, ldxx) ((xx) + (ldxx) * (yy))

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type)
{
    /* Naive implementation uses Process 0 to do all computations */

    if (grid_info->p_id == 0)
    {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    }
    else
    {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1; //! 一个简单的padding
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info)
{
}

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt)
{
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    for (int t = 0; t < nt; ++t)
    {
        cptr_t restrict a0 = buffer[t % 2];
        ptr_t restrict a1 = buffer[(t + 1) % 2];

        for (int yy = y_start; yy < y_end; yy += BLOCK_Y)
        {
            int FIXED_BLOCK_Y = min(BLOCK_Y, y_end - yy); // consider the edge situation
            for (int xx = x_start; xx < x_end; xx += BLOCK_X)
            {
                int FIXED_BLOCK_X = min(BLOCK_X, x_end - xx);

                // get the small block value to write
                ptr_t a1_block = a1 + z_start * ldx * ldy + yy * ldx + xx;

                // get the small block value to read
                cptr_t a0_block_Z = a0 + z_start * ldx * ldy + yy * ldx + xx;
                cptr_t a0_block_P = a0 + (z_start + 1) * ldx * ldy + yy * ldx + xx;
                cptr_t a0_block_N = a0 + (z_start - 1) * ldx * ldy + yy * ldx + xx;

                // loop inside block
                for (int z = z_start; z < z_end; ++z)
                {
                    for (int y = 0; y < FIXED_BLOCK_Y; ++y)
                    {
                        #pragma unroll
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
    }
    return buffer[nt % 2];
}
