
all: clean cpu gpu

cpu:
	gcc sum.c -o sum_cpu

gpu:
	nvcc sum.cu -o sum_gpu

clean:
	rm -rf sum_cpu sum_gpu sum.linkinfo
