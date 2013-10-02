#include <stdio.h>
#include <string.h>
#include <math.h>

#include "blur.h"
#include "deblur.h"
#include "include.h"

/*#define PRINT_KERNEL*/
/*#define PRINT_PSF*/

void kernelToPSF (double *psf, double *kernel, int height, int width, int kheight, int kwidth) {

    int centerx = kwidth / 2;
    int centery = kheight / 2;

    // left top
    for (int i = centery, h = 0; i < kheight; i++, h++) {
        for (int j = centerx, w = 0; j < kwidth; j++, w++) {
            psf[h * width + w] = kernel[i * kwidth + j];
        }
    }

    // right top
    for (int i = centery, h = 0; i < kheight; i++, h++) {
        for (int j = centerx - 1, w = width - 1; j >= 0; j--, w--) {
            psf[h * width + w] = kernel[i * kwidth + j];
        }
    }

    // left bottom
    for (int i = centery - 1, h = height - 1; i >= 0; i--, h--) {
        for (int j = centerx, w = 0; j < kwidth; j++, w++) {
            psf[h * width + w] = kernel[i * kwidth + j];
        }
    }

    // right bottom
    for (int i = centery - 1, h = height - 1; i >= 0; i--, h--) {
        for (int j = centerx - 1, w = width - 1; j >= 0; j--, w--) {
            psf[h * width + w] = kernel[i * kwidth + j];
        }
    }
}

double readPSF(double *psf, IplImage *kernelImage, int height, int width) {

    int kwidth = kernelImage->width, kheight = kernelImage->height;
    double *kernel = (double *) malloc (sizeof(double) * kwidth * kheight);
    double total = 0.0;
    double scale = 0.0;

    for (int h = 0; h < kheight; h++) {
        for (int w = 0; w < kwidth; w++) {
            kernel[h * kwidth + w] = IMG_ELEM(kernelImage, h, w);
        }
    }

    double max = 0;
    for (int i = 0; i < kwidth * kheight; i++) {
        total += kernel[i];
        if (max < kernel[i])
            max = kernel[i];
    }

    scale = 255.0 / max;

    printf("%f\n", total);

    for (int i = 0; i < kwidth * kheight; i++) {
        kernel[i] /= total;
        kernel[i] *= scale ;
    }

#ifdef PRINT_KERNEL
    for (int i = 0; i < kwidth * kheight; i++) {
        if (i % kwidth == 0)
            printf("\n");
        printf("%lf ", kernel[i]);
    }
    printf("\n");
#endif

    kernelToPSF(psf, kernel, height, width, kheight, kwidth);

#ifdef PRINT_PSF
    for (int i = 0; i < width * height; i++) {
        if (i % width == 0)
            printf("\n");
        printf ("%lf ", psf[i]);
    }
    printf("\n");
#endif

    return scale;

}

double genPSF(double *psf, int height, int width, int radius, double stddev, double ux, double uy) {

    int side = radius * 2 + 1;

    double scale = 0;

    double *kernel = (double *) malloc (sizeof(double) * side * side);
    memset(kernel, sizeof(double) * side * side, 0.0);

    // radius * radius kernel
    if (stddev != 0) {
        double xdist = 0.0, ydist = 0.0;
        for (int i = 0, k = 0; i < side; i++) {
            for (int j = 0; j < side; j++, k++) {
                xdist = abs((side - 1) / 2.0 + ux - j);
                ydist = abs((side - 1) / 2.0 + uy - i);
                double m = 100 * 1 / sqrt(pow(stddev, 2) * 2.0 * M_PI);
                double pn = -(pow(xdist, 2) + pow(ydist, 2));
                double pd = 2.0 * pow(stddev, 2);
                kernel[k] = (m * exp(pn / pd));
            }
        }
        double total = 0.0;
        double max = 0;

        for (int i = 0; i < side * side; i++) {
            total += kernel[i];
        }

        for (int i = 0; i < side * side; i++) {
            kernel[i] /= total;
            if (max < kernel[i])
                max = kernel[i];
        }

        scale = 255.0 / max;

        for (int i = 0; i < side * side; i++) {
            kernel[i] *= scale;
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
    printf("\n");
#endif

    kernelToPSF(psf, kernel, height, width, side, side);

#ifdef PRINT_PSF
    for (int i = 0; i < width * height; i++) {
        if (i % width == 0)
            printf("\n");
        printf ("%lf ", psf[i]);
    }
    printf("\n");
#endif

    return scale;

}

int main( int argc, char* argv[]){

    char c;
    int fileFlag = 0, psfFlag = 0, errFlag = 0, gpuFlag = 0, kernelSize = 2, blurFlag = 0,
        ux = 0, uy = 0;
    float stddev = 1.0f;
    double snr = 0.005;
    char *filename, *psfname;
    extern char *optarg;
    extern int optind, optopt, opterr;

    while ((c = getopt(argc, argv, ":bgk:s:d:f:p:x:y:")) != -1) {
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
            case 'p':
                psfname = optarg;
                psfFlag = 1;
                printf("Kernel image: %s\n", psfname);
                break;
            case 'x':
                ux = atoi(optarg);
                printf("Offset X: %d\n", ux);
                break;
            case 'y':
                uy = atoi(optarg);
                printf("Offset Y: %d\n", uy);
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

    int side = max(srcImg->width, srcImg->height) + kernelSize * 2;

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

    double scale = 0;

    // cyclic 
    if (psfFlag) {
        IplImage *kernelImage = cvLoadImage(psfname, CV_LOAD_IMAGE_GRAYSCALE);
        if (!kernelImage) {
            goto ERROR;
        }
        scale = readPSF(psf, kernelImage, img->height, img->width);
    } else {
        scale = genPSF(psf, img->height, img->width, kernelSize, stddev, ux, uy);
    }

    IplImage* psfImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, 1);

    for(int h = 0 ; h < img->height; ++h){
        for( int w = 0; w < img->width; ++w){
            IMG_ELEM_DOUBLE(psfImg, h, w) = psf[h * img->width + w];
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
        dbl1 = deblurGPU(imgSplit[0], psfImg, snr, scale);
        dbl2 = deblurGPU(imgSplit[1], psfImg, snr, scale);
        dbl3 = deblurGPU(imgSplit[2], psfImg, snr, scale);
    } else {
        dbl1 = deblurFilter(imgSplit[0], psfImg, snr, scale);
        dbl2 = deblurFilter(imgSplit[1], psfImg, snr, scale);
        dbl3 = deblurFilter(imgSplit[2], psfImg, snr, scale);
    }


    IplImage* dbl = cvClone(img);
    cvMerge(dbl1, dbl2, dbl3, NULL, dbl);

    char psfFile[256], blurFile[256], deblurFile[256];

    char *pch = strchr(filename, '.');
    (*pch) = '\0';

    if (blurFlag) {
        snprintf(psfFile, 250, "%s_psf.png", filename);
        snprintf(blurFile, 250, "%s_blur.png", filename);
    }

    if (psfFlag) {
        snprintf(deblurFile, 250, "%s_%2.4f_deblur.png", filename, snr);
    } else {
        snprintf(deblurFile, 250, "%s_%d_%2.2f_%2.4f_%d_%d_deblur.png", filename, kernelSize, stddev, snr, ux, uy);
    }

    // ROI
    IplImage* blurROI;
    IplImage* deblurROI;

    CvRect rect;

    rect.x = (side - srcImg->width) / 2; 
    rect.y = (side - srcImg->height) / 2; 
    rect.width = srcImg->width;
    rect.height = srcImg->height;

    if (blurFlag) {
        cvSetImageROI(img, rect);
        blurROI = cvCloneImage(img);

        cvSaveImage(psfFile, psfImg, 0);
        cvSaveImage(blurFile, blurROI, 0);
    }

    cvSetImageROI(dbl, rect);
    deblurROI = cvCloneImage(dbl);
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
    fprintf(stderr, "Usage: -f [/path/to/image]                path to the image file\n"); 
    fprintf(stderr, "       -p [/path/to/psf]                  path to the psf file\n"); 
    fprintf(stderr, "       -k [2]                             kernel size\n"); 
    fprintf(stderr, "       -s [0.005]                         signal-to-noise ratio\n"); 
    fprintf(stderr, "       -d [1.0]                           standard deviation\n"); 
    fprintf(stderr, "       -x [0]                             center offset x\n"); 
    fprintf(stderr, "       -y [0]                             center offset y\n"); 
    fprintf(stderr, "       -g                                 use GPU kernel\n"); 
    fprintf(stderr, "       -b                                 blur image first\n"); 
    return 1;

}
