/****************************************
 deblur.c
 ****************************************/

#include <deblur.h>
#include <include.h>
#include <fftw3.h>

IplImage* deblurFilter(IplImage *img, IplImage *psf)
{
    int height = img->height;
    int width = img->width;

    double scale = 1.0 / (double)(height * width);
    double snr = 0.005;

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
            imgIn[k][1] = 0.;
        }
    }

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

    return dst;

}
