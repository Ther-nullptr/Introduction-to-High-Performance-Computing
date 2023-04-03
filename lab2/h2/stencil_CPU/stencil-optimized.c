#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
const char *version_name = "Optimized version";
#include "common.h"

#ifndef BLOCK_X
#define BLOCK_X 256
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 8
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define INDEX(xx, yy, zz, ldxx, ldyy) ((xx) + (ldxx) * ((yy) + (ldyy) * (zz)))
#define INDEX_NEW(xx, yy, ldxx) ((xx) + (ldxx) * (yy))

MPI_Status status;

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type)
{
    grid_info->local_size_x = grid_info->global_size_x;
    grid_info->local_size_y = grid_info->global_size_y;
    // divide the thread to p_num parts
    grid_info->local_size_z = grid_info->global_size_z / grid_info->p_num;
    if (grid_info->p_id == 0)
    {
        printf("use %d processes to divide the z\n", grid_info->p_num);
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = grid_info->local_size_z * grid_info->p_id;
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
    int rank_num = grid_info->p_num;
    int rank_id = grid_info->p_id;

    printf("p_id:%d, x_start:%d, y_start:%d, z_start:%d, x_end:%d, y_end:%d, z_end:%d\n", rank_id, x_start, y_start, z_start, x_end, y_end, z_end);

    for (int t = 0; t < nt; ++t)
    {
        ptr_t restrict a0 = buffer[t % 2];
        ptr_t restrict a1 = buffer[(t + 1) % 2];
        
        MPI_Sendrecv(&a0[INDEX(0, 0, grid_info->local_size_z, ldx, ldy)], ldx * ldy, MPI_DOUBLE, rank_id < rank_num - 1 ? rank_id + 1 : MPI_PROC_NULL, 0, \
                     &a0[INDEX(0, 0, 0, ldx, ldy)], ldx * ldy, MPI_DOUBLE, rank_id > 0 ? rank_id - 1 : MPI_PROC_NULL, 0, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&a0[INDEX(0, 0, grid_info->halo_size_z, ldx, ldy)], ldx * ldy, MPI_DOUBLE, rank_id > 0 ? rank_id - 1 : MPI_PROC_NULL, 1, \
                     &a0[INDEX(0, 0, grid_info->halo_size_z + grid_info->local_size_z, ldx, ldy)], ldx * ldy, MPI_DOUBLE, rank_id < rank_num - 1 ? rank_id + 1 : MPI_PROC_NULL, 1, MPI_COMM_WORLD, &status);

        for (int z = z_start; z < z_end; ++z)
        {
            for (int y = y_start; y < y_end; ++y)
            {
                for (int x = x_start; x < x_end; ++x)
                {
                    a1[INDEX(x, y, z, ldx, ldy)] = ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
                    + ALPHA_NZZ * a0[INDEX(x - 1, y, z, ldx, ldy)] + ALPHA_PZZ * a0[INDEX(x + 1, y, z, ldx, ldy)] \
                    + ALPHA_ZNZ * a0[INDEX(x, y - 1, z, ldx, ldy)] + ALPHA_ZPZ * a0[INDEX(x, y + 1, z, ldx, ldy)] \
                    + ALPHA_ZZN * a0[INDEX(x, y, z - 1, ldx, ldy)] + ALPHA_ZZP * a0[INDEX(x, y, z + 1, ldx, ldy)] \
                    + ALPHA_NNZ * a0[INDEX(x - 1, y - 1, z, ldx, ldy)] + ALPHA_PNZ * a0[INDEX(x + 1, y - 1, z, ldx, ldy)] \
                    + ALPHA_NPZ * a0[INDEX(x - 1, y + 1, z, ldx, ldy)] + ALPHA_PPZ * a0[INDEX(x + 1, y + 1, z, ldx, ldy)] \
                    + ALPHA_NZN * a0[INDEX(x - 1, y, z - 1, ldx, ldy)] + ALPHA_PZN * a0[INDEX(x + 1, y, z - 1, ldx, ldy)] \
                    + ALPHA_NZP * a0[INDEX(x - 1, y, z + 1, ldx, ldy)] + ALPHA_PZP * a0[INDEX(x + 1, y, z + 1, ldx, ldy)] \
                    + ALPHA_ZNN * a0[INDEX(x, y - 1, z - 1, ldx, ldy)] + ALPHA_ZPN * a0[INDEX(x, y + 1, z - 1, ldx, ldy)] \
                    + ALPHA_ZNP * a0[INDEX(x, y - 1, z + 1, ldx, ldy)] + ALPHA_ZPP * a0[INDEX(x, y + 1, z + 1, ldx, ldy)] \
                    + ALPHA_NNN * a0[INDEX(x - 1, y - 1, z - 1, ldx, ldy)] + ALPHA_PNN * a0[INDEX(x + 1, y - 1, z - 1, ldx, ldy)] \
                    + ALPHA_NPN * a0[INDEX(x - 1, y + 1, z - 1, ldx, ldy)] + ALPHA_PPN * a0[INDEX(x + 1, y + 1, z - 1, ldx, ldy)] \
                    + ALPHA_NNP * a0[INDEX(x - 1, y - 1, z + 1, ldx, ldy)] + ALPHA_PNP * a0[INDEX(x + 1, y - 1, z + 1, ldx, ldy)] \
                    + ALPHA_NPP * a0[INDEX(x - 1, y + 1, z + 1, ldx, ldy)] + ALPHA_PPP * a0[INDEX(x + 1, y + 1, z + 1, ldx, ldy)];
                }
            }
        }
    }
    return buffer[nt % 2];
}
