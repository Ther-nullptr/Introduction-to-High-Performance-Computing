CC=mpiicc
CFLAGS=-Wall -D_DEBUG -O3
#CFLAGS=-Wall -D_DEBUG -O3 -qopt-report

all: pi_serial pi_mpi

%.o: %.c
	$(CC) $(CFLAGS) -c $<

pi_serial : pi_serial.o
	$(CC) $(CFLAGS) -o $@ $<

pi_mpi : pi_mpi.o
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm *.o pi_serial pi_mpi
