README
======

DEBLUR on CMP and GPU
---------------------
This is an open source project enabling you to deblur images much faster on CMP
and GPU using state-of-the-art parallel technology.

Compiling
---------
To compile the whole project, you should meet three prerequirements:
    1. OpenCV 2.3
    2. FFFW 3.2
    3. CUDA 2.3

Running
-------
A command line interface is provided as follow:

Usage:     -f [/path/to/image]                path to the image file
           -p [/path/to/kernel/image]         path to the kernel image
           -k [2]                             kernel size
           -s [0.005]                         signal-to-noise ratio
           -d [1.0]                           standard deviation
           -x [0]                             center offset X
           -y [0]                             center offset Y
           -e [0.0]                           enhance with a gamma value
           -g                                 use GPU kernel
           -b                                 blur image first

License
-------
DEBLUR is Copyright (c) 2011, Max Lv <max.c.lv@gmail.com>, Fudan Univ. 
DEBLUR is licensed under MIT License.

