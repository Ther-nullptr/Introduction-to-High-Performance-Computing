CC=icc
CFLAGS=-Wall -qopenmp -D_DEBUG -O3

all: jacobi_serial jacobi_omp_static jacobi_omp_static_1 jacobi_omp_dynamic jacobi_omp_collapse

%.o: %.c
	$(CC) $(CFLAGS) -c $<

jacobi_serial : jacobi_serial.o
	$(CC) $(CFLAGS) -o $@ $<

jacobi_omp_static : jacobi_omp_static.o
	$(CC) $(CFLAGS) -o $@ $<

jacobi_omp_static_1 : jacobi_omp_static_1.o
	$(CC) $(CFLAGS) -o $@ $<

jacobi_omp_dynamic : jacobi_omp_dynamic.o
	$(CC) $(CFLAGS) -o $@ $<

jacobi_omp_collapse : jacobi_omp_collapse.o
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm *.o jacobi_serial jacobi_omp_static jacobi_omp_static_1 jacobi_omp_dynamic jacobi_omp_collapse
