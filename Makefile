# Makefile of research program

##ディレクトリ
OBJ_DIR = bin
SRC_DIR = src
INCLUDE_DIR = src
LIB_DIR = lib

MAINFILE = main.c
TARGET = deblur
SRCS = ${MAINFILE} blur.c deblur.c #expsystem.c fourier.c
OBJS := ${SRCS:.c=.o}
OBJS := ${addprefix ${OBJ_DIR}/, ${OBJS}}
INCLUDE_HEADER = ${INCLUDE_DIR}/include.h

#日浦先生にもらったソースのライブラリ
#HIURA_LIB_FLAG =  -Llib -Ilib
#HIURA_LIB = complex.o fourier.o matrix.o ppm.o
#HIURA_LIB := ${addprefix ${LIB_DIR}/, ${HIURA_LIB}}


#OpenCVを追加する時
# オブジェクトファイルの生成はCVFLAGS
# リンクするときはCVLIBS
#CVFLAGS = `pkg-config --cflags opencv` 
#CVLIBS = `pkg-config --libs opencv`
CVFLAGS = `pkg-config --cflags opencv` 
CVLIBS = `pkg-config --libs opencv`

#マクロ定義
CC = gcc
CFLAGS =-std=gnu99 -m64 \
	-I${INCLUDE_DIR} ${HIURA_LIB_FLAG}
DEBUG = -g -O0
CLIBFLAGS = -lm  -lstdc++ -lfftw3 #リンクするもの

##生成規則

# TARGET (最終的な実行可能ファイル)
${TARGET}:${OBJS} ${HIURA_LIB}
	${CC} ${CFLAGS} -o $@  ${DEBUG} ${CVLIBS} ${CLIBFLAGS} \
${OBJS} 

# main.c のみ別方法で
${OBJ_DIR}/${MAINFILE:.c=.o}:${MAINFILE}
	${CC} $< ${CFLAGS} -c -o $@ ${DEBUG} ${CVFLAGS}	

##OpenCVを使って確認する時(checvCV.c)
checker.out:cvCheck.c
	${CC} $< ${CFLAGS} -o $@ ${DEBUG} ${CVFLAGS} ${CVLIBS} ${CLIBFLAGS}



#サフィックスルール
${OBJ_DIR}/%.o:${SRC_DIR}/%.c ${INCLUDE_HEADER} ${SRC_DIR}/%.h
	${CC} $<  ${CFLAGS} -c -o $@  ${DEBUG} ${CVFLAGS}


clean:
	rm -f ${TARGET} ${OBJS}
