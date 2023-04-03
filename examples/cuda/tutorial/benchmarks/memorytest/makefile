
all: clean gpu

gpu:
	nvcc misaligned.cu -o misaligned
	nvcc strided.cu -o strided

clean:
	rm -rf misaligned misaligned.linkinfo strided strided.linkinfo

