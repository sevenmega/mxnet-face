#!/usr/bin/env sh

gcc -o cam cam.cpp -L../lib -ldlib -lstdc++ -lm -I../../dlib/dlib-19.1 -std=c++11 -lopencv_core -lopencv_highgui
