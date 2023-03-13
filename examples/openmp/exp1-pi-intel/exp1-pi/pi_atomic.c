#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    int i;
    int num_threads;
    long long num_steps;
    double x;
    double pi;
    double sum;
    double step;
    clock_t user_time;
    struct timeval start_time, end_time;

    // check the number of arguments
    if (argc != 3) {
        printf("Usage: program_name num_threads num_steps\n");
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

    // get the number of steps
    num_steps = atoll(argv[2]);
    if (num_steps < 1) {
        printf("Illegal argument: the number of steps must be >= 1\n");
        return 3;
    }

    // init sum
    sum = 0.0;

    // calc step length
    step = 1. / (double)num_steps;

    // set the number of threads
    #ifdef _OPENMP
        omp_set_num_threads(num_threads);
    #endif

    // start timing
    gettimeofday(&start_time, NULL);

    // calc pi
    #pragma omp parallel for private(x) shared(sum)
    for (i = 0; i < num_steps; i++) {
        x = (i + .5) * step;
        #pragma omp atomic
        sum += 4.0 / (1.+ x * x);
    }

    // stop timing
    gettimeofday(&end_time, NULL);

    pi = sum * step;

    // output results
    printf("%.10f\t%ld\t%d\t%lld\n",
        pi,
        ((end_time.tv_sec - start_time.tv_sec) * 1000000 +
        (end_time.tv_usec - start_time.tv_usec)) / 1000,
        num_threads, num_steps);

    return 0;
}

