#ifndef __BLUR__
#define __BLUR__

#include <stdio.h>
#include <math.h>

#include <cv.h>
#include <highgui.h>

IplImage* deblurFilter(IplImage *img, IplImage *psf);

#endif
