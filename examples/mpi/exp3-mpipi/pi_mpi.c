#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int i;
    int num_threads;
    long long num_steps;
    double x;
    double pi;
    double sum,tmp;
    double step;
    clock_t user_time;
    struct timeval start_time, end_time;

	MPI_Status status;
	int group_size, my_rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &group_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (my_rank==0) {

    	// check the number of arguments
    	if (argc != 2) {
        	printf("Usage: program_name num_steps\n");
        	return 1;
    	}

    	// get the number of steps
    	num_steps = atoll(argv[1]);
    	if (num_steps < 1) {
        	printf("Illegal argument: the number of steps must be >= 1\n");
        	return 2;
    	}

    	// init sum
    	sum = 0.0;

    	// calc step length
    	step = 1. / (double)num_steps;

		for (i=1;i<group_size;i++)
			MPI_Send(&num_steps, 1, MPI_LONG, i, i, MPI_COMM_WORLD);

    	// start timing
    	gettimeofday(&start_time, NULL);

		// calc partial pi
        for (i = my_rank; i < num_steps; i=i+group_size) {
        	x = (i + .5) * step;
        	sum += 4.0 / (1.+ x * x);
		}

        for (i = 1; i < group_size; i++) {
			MPI_Recv(&tmp,1,MPI_DOUBLE,i,i,MPI_COMM_WORLD,&status);
			sum = sum + tmp;
        }

    	// stop timing
    	gettimeofday(&end_time, NULL);

    	pi = sum * step;

    	// output results
    	printf("%.10f\t%ld\t%lld\n",
        	pi,
        	((end_time.tv_sec - start_time.tv_sec) * 1000000 +
        	(end_time.tv_usec - start_time.tv_usec)) / 1000,
        	num_steps);
	}
	else {
		MPI_Recv(&num_steps,1,MPI_LONG,0,my_rank,MPI_COMM_WORLD,&status);

    	// calc step length
    	step = 1. / (double)num_steps;

        for (i=my_rank;i<num_steps;i=i+group_size) {
			x = (i+0.5)*step;
			sum=sum+4.0/(1.0+x*x);
		}

        MPI_Send(&sum,1,MPI_DOUBLE,0,my_rank,MPI_COMM_WORLD);
	}

	MPI_Finalize(); 


    return 0;
}
