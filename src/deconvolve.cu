#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

extern "C" void deconvolve (cufftComplex *src1, cufftComplex *src2, cufftComplex
        *dst, int size, int numThreads, double snr);

__global__ void decon_kernel (cufftComplex *src1, cufftComplex *src2,
        cufftComplex * dst, int size, double snr) {

    const int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= size)
        return;

    double a = src1[k].x;
    double b = src1[k].y;
    double c = src2[k].x;
    double d = src2[k].y;

    dst[k].x= (a*c + b*d) / ((c*c + d*d) + snr);
    dst[k].y = (b*c - a*d) / ((c*c + d*d) + snr); 

}

void deconvolve (cufftComplex *src1, cufftComplex *src2, cufftComplex *dst, int
        size, int numThreads, double snr) {
    int numBlocks = (size + numThreads - 1) / numThreads;
    decon_kernel<<<numBlocks, numThreads>>>(src1, src2, dst, size, snr);

}

