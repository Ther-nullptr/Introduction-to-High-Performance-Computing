#include <stdio.h>
#include <math.h>
#include <sys/time.h>

int main( argc, argv )
int argc;
char **argv;
{
    int     i, j, itcnt=0, t_steps=10, maxnx, maxny;
	int		num_threads;
	double	sum=0.0;
    double  *grid=NULL, *temp_grid=NULL, *other_grid=NULL;
    struct timeval  start_time, end_time;     

    // check the number of arguments
    if (argc != 4) {
        printf("Usage: program_name num_threads matrix_dim_x matrix_dim_y\n");
        return 1;
    }

    // get the number of threads
    num_threads = atoi(argv[1]);
    if (num_threads < 1) {
        printf("Illegal argument: the number of threads must be >= 1\n");
        return 2;
    }
    #ifdef _OPENMP
    /*
     *  // limit num_threads upto max_threads
     *  i = omp_get_max_threads();
     *  if (num_threads > i)
     *      num_threads = i;
     */
    #else
        num_threads = 1;
	#endif

    // get the matrix dims
    maxnx = atoll(argv[2]);
    if (maxnx < 1) {
        printf("Illegal argument: the matrix dim x must be >= 1\n");
        return 3;
    }
	maxnx += 2;

    maxny = atoll(argv[3]);
    if (maxny < 1) {
        printf("Illegal argument: the matrix dim y must be >= 1\n");
        return 4;
    }
	maxny += 2;

    grid = (double *)_mm_malloc(maxnx*maxny*sizeof(double),64);
    if( grid == NULL ) {
        printf("grid malloc error\n");
        exit(-1);
    }

    temp_grid = (double *)_mm_malloc(maxnx*maxny*sizeof(double),64);
    if( temp_grid == NULL ) {
        printf("temp_grid malloc error\n");
        exit(-1);
    }

    /* Fill the data as specified */
    for (i=0; i<maxnx; i++) {
	    for (j=0; j<maxny; j++) grid[i*maxny+j] = 1.;
    }
 
    // set the number of threads
    #ifdef _OPENMP
        omp_set_num_threads(num_threads);
    #endif

    // start timing
    gettimeofday(&start_time, NULL);

    for (itcnt=0;itcnt<t_steps;itcnt++)
    {
	    /* Compute new values */
		#pragma omp parallel for private(i,j) shared(maxnx,maxny,temp_grid,grid) schedule(static) collapse(2)
	    for (i=1; i<maxnx-1; i++) {
	        for (j=1; j<maxny-1; j++) {
		        temp_grid[i*maxny+j] = (grid[i*maxny+j+1] + grid[i*maxny+j-1] +
			            grid[(i+1)*maxny+j] + grid[(i-1)*maxny+j]) / 4.0;
	        }
        }

	    /* Only transfer the interior points */
        other_grid = temp_grid;
        temp_grid = grid;
        grid = other_grid;
        other_grid = NULL;
    }

    // stop timing
    gettimeofday(&end_time, NULL);

	sum = 0.0;
	for (i=1; i<maxnx-1; i++) {
		for (j=1; j<maxny-1; j++) {
			sum += (grid[i*maxny+j]/(double)(maxnx*maxny));
		}
	}
	
    printf("Jacobi time: %ld ms\t average %.10f\n",
        ((end_time.tv_sec - start_time.tv_sec) * 1000000 +
        (end_time.tv_usec - start_time.tv_usec)) / 1000, sum);

    _mm_free(temp_grid);
    _mm_free(grid);

    return 0;
}
