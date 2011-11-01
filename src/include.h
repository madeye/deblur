#ifndef __INCLUDE_HEADER__
#define __INCLUDE_HEADER__

#define YES 1
#define NO 0

#define LOAD_DISPARITY_MAP YES

#define LEFT_CAM 0
#define CENTER_CAM 1
#define RIGHT_CAM 2

#define MAX_DISPARITY 32
#define MAX_PSF_SIZE 32
#define FFT_SIZE 512
#define CUT_OFF_SIZE FFT_SIZE //切り取る大きさはFFTのと同じにしなければならない
#define BLOCK_SIZE 16 //2^nのほうが都合が良い

#define IMG_ELEM(img, h, w) ( ((uchar*) ((img)->imageData + (h) * (img)->widthStep))[(w)] )
#define IMG_ELEM_DOUBLE(img, h, w) ( ((double*) ((img)->imageData + (h) * (img)->widthStep))[(w)] )
#define max(a,b) (((a)<(b))?(b):(a))


#endif 
