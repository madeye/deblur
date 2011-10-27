# Makefile of research program

##ディレクトリ
OBJ_DIR = bin
SRC_DIR = src
INCLUDE_DIR = src
LIB_DIR = lib

PKG_CONFIG_PATH := /usr/local/lib/pkgconfig/:$(PKG_CONFIG_PATH)

MAINFILE = main.c
TARGET = deblur
SRCS = ${MAINFILE} blur.c deblur.c #expsystem.c fourier.c
OBJS := ${SRCS:.c=.o}
OBJS := ${addprefix ${OBJ_DIR}/, ${OBJS}}
CUDA_SRCS = deconvolve.cu
CUDA_OBJS := ${CUDA_SRCS:.cu=.o}
CUDA_OBJS := ${addprefix ${OBJ_DIR}/, ${CUDA_OBJS}}
INCLUDE_HEADER = ${INCLUDE_DIR}/include.h

#OpenCVを追加する時
# オブジェクトファイルの生成はCVFLAGS
# リンクするときはCVLIBS
#CVFLAGS = `pkg-config --cflags opencv` 
#CVLIBS = `pkg-config --libs opencv`
CVFLAGS = `pkg-config --cflags opencv` 
CVLIBS = `pkg-config --libs opencv`

FFTWFLAGS = `pkg-config --cflags fftw3` 
FFTWLIBS = `pkg-config --libs fftw3` -lfftw3_threads

CUDAFLAGS = -I/usr/local/cuda/include
CUDALIBS = -L/usr/local/cuda/lib -lcufft
NVCCFLAGS = -arch=sm_13 --use_fast_math

#マクロ定義
CC = gcc
NVCC = nvcc
CFLAGS =-std=gnu99 \
	-I${INCLUDE_DIR}
DEBUG = -g -O2
CLIBFLAGS = -lm -lstdc++ #リンクするもの

##生成規則

# TARGET (最終的な実行可能ファイル)
${TARGET}:${OBJS} ${CUDA_OBJS}
	${CC} ${CFLAGS} -o $@ ${DEBUG} ${CLIBFLAGS} ${CVLIBS} ${FFTWLIBS} ${CUDALIBS} \
${OBJS} ${CUDA_OBJS} 

# main.c のみ別方法で
${OBJ_DIR}/${MAINFILE:.c=.o}:${MAINFILE}
	${CC} $< ${CFLAGS} -c -o $@ ${DEBUG} ${CVFLAGS} ${FFTWFLAGS} ${CUDAFLAGS}


#サフィックスルール
${OBJ_DIR}/%.o:${SRC_DIR}/%.c ${INCLUDE_HEADER} ${SRC_DIR}/%.h
	${CC} $<  ${CFLAGS} -c -o $@  ${DEBUG} ${CVFLAGS} ${FFTWFLAGS} ${CUDAFLAGS}

# CUDA COMPILER
${OBJ_DIR}/%.o:${SRC_DIR}/%.cu
	${NVCC} $< -c -o $@  ${DEBUG} ${NVCCFLAGS}

clean:
	rm -f ${TARGET} ${OBJS} ${CUDA_OBJS}
