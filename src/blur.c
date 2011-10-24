/****************************************
 blur.c
 ****************************************/

#include <blur.h>
#include <include.h>
#include <fftw3.h>

void normalize( fftw_complex *arr, int size )
{
    double norm = 0.0;

    // compute norm
    for( int i = 0; i < size; ++i){
        double re = arr[i][0];
        norm += re;
    }

    printf("norm = %lf\n", norm);

    // normalize
    for( int i = 0; i < size; ++i){
        arr[i][0] /= norm;
    }

    return;

}


IplImage* blur(IplImage *img, double *psf)
{
    int height = img->height;
    int width = img->width;

    fftw_plan     plan_f_img, plan_f_psf;
    fftw_plan     plan_if_dst;


    double * imgIn = (double *) malloc (sizeof(double) * height * width);
    double * psfIn = (double *) malloc (sizeof(double) * height * width);
    double * dstIn = (double *) malloc (sizeof(double) * height * width);

    fftw_complex * imgFreq = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * height * width / 2 + 1);
    fftw_complex * psfFreq = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * height * width / 2 + 1);
    fftw_complex * dstFreq = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * height * width / 2 + 1);

    //copy
    for(int h = 0 , k = 0; h < height; ++h){
        for( int w = 0 ; w < width; ++w, ++k){
            imgIn[k] = (double) IMG_ELEM(img, h, w);
        }
    }

    //copy psf
    for(int h = 0 , k = 0; h < height; ++h){
        for( int w = 0 ; w < width; ++w, ++k){
            int y = height - h;
            int x = width - w;
            psfIn[k] = psf[k];
        }
    }

    plan_f_img = fftw_plan_dft_r2c_2d(width, height, imgIn, imgFreq, FFTW_ESTIMATE);
    plan_f_psf = fftw_plan_dft_r2c_2d(width, height, psfIn, psfFreq, FFTW_ESTIMATE);
    plan_if_dst = fftw_plan_dft_c2r_2d(width, height, dstFreq, dstIn, FFTW_ESTIMATE);

    fftw_execute(plan_f_img);
    fftw_execute(plan_f_psf);

    for(int h = 0, k = 0; h < height; ++h){
        for( int w = 0 ; w < width / 2 + 1; ++w, ++k){
            double a = imgFreq[k][0];
            double b = imgFreq[k][1];
            double c = psfFreq[k][0];
            double d = psfFreq[k][1];
            /*printf("%d, %d, %d, %d", a, b, c, d);*/

            dstFreq[k][0] = (a*c - b*d) / (double)(width*height/2 + 1);
            dstFreq[k][1] = (b*c + d*a) / (double)(width*height/2 + 1);

        }
    }

    fftw_execute(plan_if_dst);

    IplImage* dst = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    for(int h = 0, k = 0 ; h < height; ++h){
        for( int w = 0; w < width; ++w, ++k){
            IMG_ELEM(dst, h, w) = dstIn[k];
            printf("%d, ", IMG_ELEM(dst, h, w));
        }
        printf("\n");
    }

    return dst;

}

IplImage* blurPSF(IplImage *img, IplImage *psf)
{
    int height = img->height;
    int width = img->width;

    double scale = 1.0 / (height * width);

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
            /*psfIn[k][0] = (double)IMG_ELEM(psf, y, x);*/
            psfIn[k][0] = (double)IMG_ELEM(psf, h, w) / 256.0;
            psfIn[k][1] = 0.;
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

            dstFreq[k][0] = (a*c - b*d) * scale;
            dstFreq[k][1] = (b*c + d*a) * scale;

        }
    }

    fftw_execute(plan_if_dst);

    /*normalize(dstIn, height * width);*/

    IplImage* dst = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    for(int h = 0, k = 0 ; h < height; ++h){
        for( int w = 0; w < width; ++w, ++k){
            IMG_ELEM(dst, h, w) = dstIn[k][0];
            /*printf("%d ", IMG_ELEM(dst, h, w));*/
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
