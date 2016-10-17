#!/usr/bin/env sh

gcc -o cam cam.cpp -std=c++11 \
    -lstdc++ -lm -L../lib -ldlib -I../../dlib/dlib-19.1 \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect
