#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define SNR 0.005

extern "C" void deconvolve (cufftComplex *src1, cufftComplex *src2, cufftComplex
        *dst, int size, int numThreads);

__global__ void decon_kernel (cufftComplex *src1, cufftComplex *src2,
        cufftComplex * dst, int size) {

    const int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= size)
        return;

    float a = src1[k].x;
    float b = src1[k].y;
    float c = src2[k].x;
    float d = src2[k].y;

    dst[k].x= (a*c + b*d) / ((c*c + d*d) + SNR);
    dst[k].y = (b*c - a*d) / ((c*c + d*d) + SNR); 

}

void deconvolve (cufftComplex *src1, cufftComplex *src2, cufftComplex *dst, int
        size, int numThreads) {
    int numBlocks = (size + numThreads - 1) / numThreads;
    decon_kernel<<<numBlocks, numThreads>>>(src1, src2, dst, size);

}

