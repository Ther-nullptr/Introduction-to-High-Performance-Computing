#include <stdio.h>
#include <stdlib.h>
#include "common.h"


const char* version_name = "Optimized version";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}



inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}


ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {


}