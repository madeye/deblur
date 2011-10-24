#include <stdio.h>
#include <string.h>

#include "blur.h"
#include "deblur.h"
#include "include.h"

int main( int argc, char* argv[]){


  IplImage* img = cvLoadImage("yun_512_512.bmp", CV_LOAD_IMAGE_COLOR);
  IplImage* imgSplit[3];
  for(int c = 0; c < 3; ++c)
      imgSplit[c] = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
  cvSplit(img, imgSplit[0], imgSplit[1], imgSplit[2], NULL);

  printf("Height: %d, Width: %d\n", img->height, img->width);

  double * psf = (double *) malloc (sizeof(double) * img->width * img->height);
  for (int i = 0; i < img->width * img->height; i++)
      psf[i] = 0.;

  if(!img || !psf) return 1;

  double scale = 256.0 / 15.0;

  // cyclic 
  psf[0*img->width + 0] = 3.0 * scale;
  psf[0*img->width + 1] = 2.0 * scale;
  psf[1*img->width + 0] = 2.0 * scale;
  psf[1*img->width + 1] = 1.0 * scale;
  psf[(img->height - 1) * img->width + 0] = 2.0 * scale;
  psf[(img->height - 1) * img->width + 1] = 1.0 * scale;
  psf[0*img->width + img->width - 1] = 2.0 * scale;
  psf[1*img->width + img->width - 1] = 1.0 * scale;
  psf[img->height*img->width - 1] = 1.0 * scale;

  IplImage* psfImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

  for(int h = 0 ; h < img->height; ++h){
      for( int w = 0; w < img->width; ++w){
          IMG_ELEM(psfImg, h, w) = psf[h * img->width + w];
      }
  }

  /*IplImage* bl = blur(imgSplit[0], psf);*/
  IplImage* bl1 = blurPSF(imgSplit[0], psfImg);
  IplImage* bl2 = blurPSF(imgSplit[1], psfImg);
  IplImage* bl3 = blurPSF(imgSplit[2], psfImg);
  IplImage* bl = cvMerge(bl1, bl2, bl3, NULL);

  IplImage* dbl1 = deblurFilter(bl1, psfImg);
  IplImage* dbl2 = deblurFilter(bl2, psfImg);
  IplImage* dbl3 = deblurFilter(bl3, psfImg);
  IplImage* dbl = cvMerge(dbl1, dbl2, dbl3, NULL);

  cvSaveImage("yun_512_512_psf.bmp", psfImg, 0);
  cvSaveImage("yun_512_512_blur.bmp", bl, 0);
  cvSaveImage("yun_512_512_deblur.bmp", dbl, 0);
  return 0;
}
