CC=icc
CFLAGS=-Wall -qopenmp -D_DEBUG -O3

all: pi_serial pi_manual pi_manual_pad pi_atomic pi_critical pi_locked pi_reduction pi_task

%.o: %.c
	$(CC) $(CFLAGS) -c $<

pi_serial : pi_serial.o
	$(CC) $(CFLAGS) -o $@ $<

pi_manual : pi_manual.o
	$(CC) $(CFLAGS) -o $@ $<

pi_manual_pad : pi_manual_pad.o
	$(CC) $(CFLAGS) -o $@ $<

pi_atomic : pi_atomic.o
	$(CC) $(CFLAGS) -o $@ $<

pi_critical : pi_critical.o
	$(CC) $(CFLAGS) -o $@ $<

pi_locked : pi_locked.o
	$(CC) $(CFLAGS) -o $@ $<

pi_reduction : pi_reduction.o
	$(CC) $(CFLAGS) -o $@ $<

pi_task : pi_task.o
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm *.o pi_serial pi_manual pi_manual_pad pi_atomic pi_critical pi_locked pi_reduction pi_task
