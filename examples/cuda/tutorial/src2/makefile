
all: clean cpu gpu

cpu:
	gcc -O3 fractal.c -o fractal_cpu 

gpu:
	nvcc fractal.cu -o fractal_gpu

clean:
	rm -rf fractal_cpu fractal_gpu fractal.linkinfo fractal.ppm fractal.bmp
	
convert:
	convert fractal.ppm fractal.bmp
	rm fractal.ppm

