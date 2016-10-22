#!/usr/bin/env sh

gcc -o test test.cpp -std=c++11 \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect \
    -lstdc++ -lm

