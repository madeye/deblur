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
    int fileFlag = 0, errFlag = 0, gpuFlag = 0, kernelSize = 2, blurFlag = 0;
    float stddev = 1.0f;
    double snr = 0.005;
    char * filename;
    extern char *optarg;
    extern int optind, optopt, opterr;

    while ((c = getopt(argc, argv, ":bgk:s:d:f:")) != -1) {
        switch(c) {
            case 'b':
                printf("Blur image first\n");
                blurFlag = 1;
                break;
            case 'g':
                printf("Use GPU Kernel\n");
                gpuFlag = 1;
                break;
            case 'k':
                kernelSize = atoi(optarg);
                printf("Kernel size: %d\n", kernelSize);
                break;
            case 's':
                snr = atof(optarg);
                printf("Singal-to-noise ratio: %f\n", snr);
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

    if (blurFlag) {
        imgSplit[0] = blurPSF(imgSplit[0], psfImg);
        imgSplit[1] = blurPSF(imgSplit[1], psfImg);
        imgSplit[2] = blurPSF(imgSplit[2], psfImg);
        cvMerge(imgSplit[0], imgSplit[1], imgSplit[2], NULL, img);
    }

    IplImage* dbl1;
    IplImage* dbl2;
    IplImage* dbl3;

    if (gpuFlag) {
        dbl1 = deblurGPU(imgSplit[0], psfImg, snr);
        dbl2 = deblurGPU(imgSplit[1], psfImg, snr);
        dbl3 = deblurGPU(imgSplit[2], psfImg, snr);
    } else {
        dbl1 = deblurFilter(imgSplit[0], psfImg, snr);
        dbl2 = deblurFilter(imgSplit[1], psfImg, snr);
        dbl3 = deblurFilter(imgSplit[2], psfImg, snr);
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

    cvSetImageROI(img, rect);
    cvSetImageROI(dbl, rect);

    blurROI = cvCloneImage(img);
    deblurROI = cvCloneImage(dbl);

    cvSaveImage(psfFile, psfImg, 0);
    cvSaveImage(blurFile, blurROI, 0);
    cvSaveImage(deblurFile, deblurROI, 0);

    cvReleaseImage(&imgSplit[0]);
    cvReleaseImage(&imgSplit[1]);
    cvReleaseImage(&imgSplit[2]);

    cvReleaseImage(&psfImg);
    cvReleaseImage(&img);

    cvReleaseImage(&dbl);
    cvReleaseImage(&dbl1);
    cvReleaseImage(&dbl2);
    cvReleaseImage(&dbl3);

    return 0;

ERROR:
    fprintf(stderr, "Usage: -f [/path/to/image/file]           path to the image file\n"); 
    fprintf(stderr, "       -k [2]                             kernel size\n"); 
    fprintf(stderr, "       -s [0.005]                         signal-to-noise ratio\n"); 
    fprintf(stderr, "       -d [1.0]                           standard deviation\n"); 
    fprintf(stderr, "       -g                                 use GPU kernel\n"); 
    fprintf(stderr, "       -b                                 blur image first\n"); 
    return 1;

}
