#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

    // start timing
    gettimeofday(&start_time, NULL);

    // calc pi
    for (i = 0; i < num_steps; i++) {
        x = (i + .5) * step;
        sum += 4.0 / (1.+ x * x);
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

    return 0;
}
