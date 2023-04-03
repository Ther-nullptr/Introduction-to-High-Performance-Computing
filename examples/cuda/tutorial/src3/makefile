
all: clean cpu gpu

cpu:
	gcc -O3 mmult.c -o mmult_cpu

gpu:
	nvcc mmult.cu -o mmult_gpu

clean:
	rm -rf mmult_cpu mmult_gpu mmult.linkinfo
