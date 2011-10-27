#ifndef __DEBLUR__
#define __DEBLUR__

#include <stdio.h>
#include <math.h>

#include <cv.h>
#include <highgui.h>

IplImage* deblurFilter(IplImage *img, IplImage *psf, double snr);
IplImage* deblurGPU(IplImage *img, IplImage *psf, double snr);

#endif
