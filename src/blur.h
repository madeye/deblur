#ifndef __BLUR__
#define __BLUR__

#include <stdio.h>
#include <math.h>

#include <cv.h>
#include <highgui.h>

IplImage* blur(IplImage *img, double *psf);
IplImage* blurPSF(IplImage *img, IplImage *psf);

#endif
