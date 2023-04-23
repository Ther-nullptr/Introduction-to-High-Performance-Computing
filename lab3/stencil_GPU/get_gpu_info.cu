#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM number: " << devProp.multiProcessorCount << std::endl;
    std::cout << "shared memory per block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "max threads per multiprocessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "max threads per SM: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "max memory pitch: " << devProp.memPitch / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "sm architecture: " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "max block num per grid: " << devProp.maxGridSize[0] << " x " << devProp.maxGridSize[1] << " x " << devProp.maxGridSize[2] << std::endl;
   
   // print a grid (dim2)
}