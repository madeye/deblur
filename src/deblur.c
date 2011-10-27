/****************************************
 deblur.c
 ****************************************/

#include <deblur.h>
#include <include.h>
#include <fftw3.h>

// cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

IplImage* deblurFilter(IplImage *img, IplImage *psf, double snr)
{
    int height = img->height;
    int width = img->width;

    double scale = 1.0 / (double)(height * width);

    fftw_init_threads();

    fftw_plan     plan_f_img, plan_f_psf;
    fftw_plan     plan_if_dst;

    fftw_complex * imgIn = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * height * width);
    fftw_complex * psfIn = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * height * width);
    fftw_complex * dstIn = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * height * width);

    fftw_complex * imgFreq = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * height * width);
    fftw_complex * psfFreq = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * height * width);
    fftw_complex * dstFreq = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * height * width);

    //copy
    for(int h = 0 , k = 0; h < height; ++h){
        for( int w = 0 ; w < width; ++w, ++k){
            imgIn[k][0] = (double) IMG_ELEM(img, h, w);
            imgIn[k][1] = 0.;
            /*printf("%f, ", imgIn[k][0]);*/
        }
        /*printf("\n");*/
    }

    //copy psf
    for(int h = 0 , k = 0; h < height; ++h){
        for( int w = 0 ; w < width; ++w, ++k){
            /*int y = height - h;*/
            /*int x = width - w;*/
            psfIn[k][0] = (double)IMG_ELEM(psf, h, w) / 256.0;
            psfIn[k][1] = 0.;
        }
    }

    fftw_plan_with_nthreads(4);

    plan_f_img = fftw_plan_dft_2d(width, height, imgIn, imgFreq, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_f_psf = fftw_plan_dft_2d(width, height, psfIn, psfFreq, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_if_dst = fftw_plan_dft_2d(width, height, dstFreq, dstIn, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan_f_img);
    fftw_execute(plan_f_psf);

    for(int h = 0, k = 0; h < height; ++h){
        for( int w = 0 ; w < width; ++w, ++k){
            double a = imgFreq[k][0];
            double b = imgFreq[k][1];
            double c = psfFreq[k][0];
            double d = psfFreq[k][1];
            /*printf("%d, %d, %d, %d", a, b, c, d);*/

            dstFreq[k][0] = (a*c + b*d) / ((c*c + d*d) + snr);
            dstFreq[k][1] = (b*c - a*d) / ((c*c + d*d) + snr); 

        }
    }

    fftw_execute(plan_if_dst);

    /*normalize(dstIn, height * width);*/

    IplImage* dst = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    for(int h = 0, k = 0 ; h < height; ++h){
        for( int w = 0; w < width; ++w, ++k){
            IMG_ELEM(dst, h, w) = dstIn[k][0] * scale;
            /*printf("%d ", dstIn[k]);*/
        }
        /*printf("\n");*/
    }

    fftw_destroy_plan(plan_f_img);
    fftw_destroy_plan(plan_f_psf);
    fftw_destroy_plan(plan_if_dst);

    fftw_free(imgIn);
    fftw_free(psfIn);
    fftw_free(dstIn);
    fftw_free(imgFreq);
    fftw_free(psfFreq);
    fftw_free(dstFreq);

    fftw_cleanup_threads();

    return dst;

}

// deblur with CUFFT
IplImage* deblurGPU(IplImage *img, IplImage *psf, double snr)
{
    int height = img->height;
    int width = img->width;

    double scale = 1.0 / (double)(height * width);

    cufftHandle     plan_f_img, plan_f_psf;
    cufftHandle     plan_if_dst;

    cufftComplex * imgDev;
    cufftComplex * psfDev;
    cufftComplex * dstDev;

    cufftComplex * imgIn = (cufftComplex *) malloc (sizeof(cufftComplex) * height * width);
    cufftComplex * psfIn = (cufftComplex *) malloc (sizeof(cufftComplex) * height * width);
    cufftComplex * dstIn = (cufftComplex *) malloc (sizeof(cufftComplex) * height * width);

    cudaMalloc ((void**) &imgDev, sizeof(cufftComplex) * height * width);
    cudaMalloc ((void**) &psfDev, sizeof(cufftComplex) * height * width);
    cudaMalloc ((void**) &dstDev, sizeof(cufftComplex) * height * width);

    //copy
    for(int h = 0 , k = 0; h < height; ++h){
        for( int w = 0 ; w < width; ++w, ++k){
            imgIn[k].x = (double) IMG_ELEM(img, h, w);
            imgIn[k].y = 0.;
        }
    }

    //copy psf
    for(int h = 0 , k = 0; h < height; ++h){
        for( int w = 0 ; w < width; ++w, ++k){
            psfIn[k].x = (double)IMG_ELEM(psf, h, w) / 256.0;
            psfIn[k].y = 0.;
        }
    }

    cudaMemcpy(imgDev, imgIn, sizeof(cufftComplex) * height * width, cudaMemcpyHostToDevice);
    cudaMemcpy(psfDev, psfIn, sizeof(cufftComplex) * height * width, cudaMemcpyHostToDevice);

    cufftPlan2d(&plan_f_img, height, width, CUFFT_C2C);
    cufftPlan2d(&plan_f_psf, height, width, CUFFT_C2C);
    cufftPlan2d(&plan_if_dst, height, width, CUFFT_C2C);

    cufftExecC2C(plan_f_img, imgDev, imgDev, CUFFT_FORWARD);
    cufftExecC2C(plan_f_psf, psfDev, psfDev, CUFFT_FORWARD);

    deconvolve(imgDev, psfDev, dstDev, height * width, 64, snr);

    cufftExecC2C(plan_if_dst, dstDev, dstDev, CUFFT_INVERSE);

    IplImage* dst = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    cudaMemcpy(dstIn, dstDev, sizeof(cufftComplex) * height * width, cudaMemcpyDeviceToHost);

    for(int h = 0, k = 0 ; h < height; ++h){
        for( int w = 0; w < width; ++w, ++k){
            IMG_ELEM(dst, h, w) = dstIn[k].x * scale;
        }
    }

    cufftDestroy(plan_f_img);
    cufftDestroy(plan_f_psf);
    cufftDestroy(plan_if_dst);

    free(imgIn);
    free(psfIn);
    free(dstIn);
    cudaFree(imgDev);
    cudaFree(psfDev);
    cudaFree(dstDev);

    return dst;

}
