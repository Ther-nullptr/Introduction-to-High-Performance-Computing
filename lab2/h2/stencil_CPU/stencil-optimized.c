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

MPI_Comm cart_comm;
int up_ngb, down_ngb;
MPI_Datatype recv_from_up, send_to_up, send_to_down, recv_from_down;
MPI_Status status;

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type)
{
    /* Naive implementation uses Process 0 to do all computations */
    if (grid_info->p_id == 0)
    {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        // divide the thread to p_num parts
        grid_info->local_size_z = grid_info->global_size_z / grid_info->p_num;
        if (grid_info->p_id == 0)
        {
            printf("use %d processes to divide the z\n", grid_info->p_num);
        }
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

    extract_subarrays(grid_info);
}

void extract_subarrays(dist_grid_info_t *grid_info)
{
    const int lz = grid_info->local_size_z;
    const int hz = grid_info->halo_size_z;
    const int ly = grid_info->local_size_y;
    const int hy = grid_info->halo_size_y;
    const int lx = grid_info->local_size_x;
    const int hx = grid_info->halo_size_x;
    const int array_of_sizes = {lz + 2 * hz, ly + 2 * hy, lx + 2 * hx};
    const int array_of_subsizes = {hz, ly + 2 * hy, lx + 2 * hx};
    int array_of_starts[3];

    int dims[1] = {grid_info->p_num};
    int periods = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, &periods, 0, &cart_comm);
    MPI_Cart_shift(cart_comm, 0, 1, &down_ngb, &up_ngb);

    array_of_starts[0] = 0;
    array_of_starts[1] = 0;
    array_of_starts[2] = 0;
    MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &recv_from_up);
    MPI_Type_commit(&recv_from_up);

    array_of_starts[0] = hz;
    array_of_starts[1] = 0;
    array_of_starts[2] = 0;
    MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &send_to_up);
    MPI_Type_commit(&send_to_up);

    array_of_starts[0] = lz;
    array_of_starts[1] = 0;
    array_of_starts[2] = 0;
    MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &send_to_down);
    MPI_Type_commit(&send_to_down);

    array_of_starts[0] = hz + lz;
    array_of_starts[1] = 0;
    array_of_starts[2] = 0;
    MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &recv_from_down);
    MPI_Type_commit(&recv_from_down);
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

    for (int t = 0; t < nt; ++t)
    {
        cptr_t restrict a0 = buffer[t % 2];
        ptr_t restrict a1 = buffer[(t + 1) % 2];

        MPI_Sendrecv(a0, 1, send_to_down, down_ngb, 0, 
                     a0, 1, recv_from_up, up_ngb, 0, cart_comm, &status);
        MPI_Sendrecv(a0, 1, send_to_up, up_ngb, 1,
                     a0, 1, recv_from_down, down_ngb, 1, cart_comm, &status);

        for (int yy = y_start; yy < y_end; yy += BLOCK_Y)
        {
            for (int xx = x_start; xx < x_end; xx += BLOCK_X)
            {
                int FIXED_BLOCK_Y = min(BLOCK_Y, y_end - yy); // consider the edge situation
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
    }
    return buffer[nt % 2];
}
