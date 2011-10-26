#include <stdio.h>
#include <string.h>
#include <math.h>

#include "blur.h"
#include "deblur.h"
#include "include.h"

void genPSF(double *psf, int height, int width, int radius, double stddev) {

    int side = radius * 2 + 1;

    double *kernel = (double *) malloc (sizeof(double) * side * side);
    memset(kernel, sizeof(double) * side * side, 0.0);

    // radius * radius kernel
    if (stddev != 0) {
        double xdist = 0.0, ydist = 0.0;
        for (int i = 0, k = 0; i < side; i++) {
            for (int j = 0; j < side; j++, k++) {
                xdist = abs((side - 1) / 2.0 - j);
                ydist = abs((side - 1) / 2.0 - i);
                double m = 200.0 * (1 / (pow(stddev, 2) * 2.0 * M_PI));
                double pn = -(pow(xdist, 2) + pow(ydist, 2));
                double pd = 2.0 * pow(stddev, 2);
                kernel[k] = (int) (m * exp(pn / pd));
            }
        }
        double total = 0.0;
        for (int i = 0; i < side * side; i++) {
            total += kernel[i];
        }
        for (int i = 0; i < side * side; i++) {
            kernel[i] /= total;
            kernel[i] *= 256.0;
        }
    }
    else {
        kernel[(side * side - 1) / 2] = 1.0;
    }

#ifdef PRINT_KERNEL
    for (int i = 0; i < side * side; i++) {
        if (i % side == 0)
            printf("\n");
        printf("%lf ", kernel[i]);
    }
#endif

    // left top
    for (int i = radius, h = 0; i < side; i++, h++) {
        for (int j = radius, w = 0; j < side; j++, w++) {
            psf[h * width + w] = kernel[i * side + j];
        }
    }

    // right top
    for (int i = radius, h = 0; i < side; i++, h++) {
        for (int j = radius - 1, w = width - 1; j >= 0; j--, w--) {
            psf[h * width + w] = kernel[i * side + j];
        }
    }

    // left bottom
    for (int i = radius - 1, h = height - 1; i >= 0; i--, h--) {
        for (int j = radius, w = 0; j < side; j++, w++) {
            psf[h * width + w] = kernel[i * side + j];
        }
    }

    // right bottom
    for (int i = radius - 1, h = height - 1; i >= 0; i--, h--) {
        for (int j = radius - 1, w = width - 1; j >= 0; j--, w--) {
            psf[h * width + w] = kernel[i * side + j];
        }
    }

#ifdef PRINT_KERNEL
    for (int i = 0; i < width * height; i++) {
        if (i % width == 0)
            printf("\n");
        printf ("%lf ", psf[i]);
    }
#endif

}

int main( int argc, char* argv[]){

    char c;
    int fileFlag = 0, errFlag = 0, gpuFlag = 0, kernelSize = 2;
    float stddev = 1.0f;
    char * filename;
    extern char *optarg;
    extern int optind, optopt, opterr;

    while ((c = getopt(argc, argv, ":gs:d:f:")) != -1) {
        switch(c) {
            case 'g':
                printf("Use GPU Kernel\n");
                gpuFlag = 1;
                break;
            case 's':
                kernelSize = atoi(optarg);
                printf("Kernel size: %d\n", kernelSize);
                break;
            case 'd':
                stddev = atof(optarg);
                printf("Kernel stddev: %f\n", stddev);
                break;
            case 'f':
                filename = optarg;
                fileFlag = 1;
                printf("Processing file: %s\n", filename);
                break;
            case ':':
                printf("-%c without input\n", optopt);
                errFlag++;
                break;
            case '?':
                printf("unknown arg %c\n", optopt);
                errFlag++;
                break;
        }
    }

    if (errFlag || !fileFlag) {
        goto ERROR;
    }

    IplImage* img;
    IplImage* srcImg = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);

    if (!srcImg) 
        goto ERROR;

    int side = max(srcImg->width, srcImg->height);

    if (srcImg->height != srcImg->width) {
        CvSize size = cvSize(side, side);
        img = cvCreateImage(size, IPL_DEPTH_8U, 3);
        CvPoint offset = cvPoint((side - srcImg->width) / 2, (side - srcImg->height) / 2);
        cvCopyMakeBorder(srcImg, img, offset, IPL_BORDER_REPLICATE, cvScalarAll(0));
    } else {
        img = srcImg;
    }

    IplImage* imgSplit[3];
    for(int c = 0; c < 3; ++c)
        imgSplit[c] = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvSplit(img, imgSplit[0], imgSplit[1], imgSplit[2], NULL);

    printf("Height: %d, Width: %d\n", img->height, img->width);

    double * psf = (double *) malloc (sizeof(double) * img->width * img->height);
    for (int i = 0; i < img->width * img->height; i++)
        psf[i] = 0.;

    if(!img || !psf) return 1;

    // cyclic 
    genPSF(psf, img->height, img->width, kernelSize, stddev);

    IplImage* psfImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    for(int h = 0 ; h < img->height; ++h){
        for( int w = 0; w < img->width; ++w){
            IMG_ELEM(psfImg, h, w) = psf[h * img->width + w];
        }
    }

    IplImage* bl1 = blurPSF(imgSplit[0], psfImg);
    IplImage* bl2 = blurPSF(imgSplit[1], psfImg);
    IplImage* bl3 = blurPSF(imgSplit[2], psfImg);
    IplImage* bl = cvClone(img);
    cvMerge(bl1, bl2, bl3, NULL, bl);

    IplImage* dbl1;
    IplImage* dbl2;
    IplImage* dbl3;

    if (gpuFlag) {
        dbl1 = deblurGPU(bl1, psfImg);
        dbl2 = deblurGPU(bl2, psfImg);
        dbl3 = deblurGPU(bl3, psfImg);
    } else {
        dbl1 = deblurFilter(bl1, psfImg);
        dbl2 = deblurFilter(bl2, psfImg);
        dbl3 = deblurFilter(bl3, psfImg);
    }

    IplImage* dbl = cvClone(img);
    cvMerge(dbl1, dbl2, dbl3, NULL, dbl);

    char psfFile[256], blurFile[256], deblurFile[256];
    snprintf(psfFile, 250, "%s_psf.bmp", filename);
    snprintf(blurFile, 250, "%s_blur.bmp", filename);
    snprintf(deblurFile, 250, "%s_deblur.bmp", filename);

    // ROI
    IplImage* blurROI;
    IplImage* deblurROI;

    CvRect rect;

    rect.x = (side - srcImg->width) / 2; 
    rect.y = (side - srcImg->height) / 2; 
    rect.width = srcImg->width;
    rect.height = srcImg->height;

    cvSetImageROI(bl, rect);
    cvSetImageROI(dbl, rect);

    blurROI = cvCloneImage(bl);
    deblurROI = cvCloneImage(dbl);

    cvSaveImage(psfFile, psfImg, 0);
    cvSaveImage(blurFile, blurROI, 0);
    cvSaveImage(deblurFile, deblurROI, 0);

    cvReleaseImage(&imgSplit[0]);
    cvReleaseImage(&imgSplit[1]);
    cvReleaseImage(&imgSplit[2]);

    cvReleaseImage(&psfImg);
    cvReleaseImage(&img);

    cvReleaseImage(&bl);
    cvReleaseImage(&bl1);
    cvReleaseImage(&bl2);
    cvReleaseImage(&bl3);

    cvReleaseImage(&dbl);
    cvReleaseImage(&dbl1);
    cvReleaseImage(&dbl2);
    cvReleaseImage(&dbl3);

    return 0;

ERROR:
    fprintf(stderr, "Usage: -f [/path/to/image/file]           path to the image file\n"); 
    fprintf(stderr, "       -s [2]                             kernel size\n"); 
    fprintf(stderr, "       -d [1.0]                           standard deviation\n"); 
    fprintf(stderr, "       -g                                 use GPU kernel\n"); 
    return 1;

}
