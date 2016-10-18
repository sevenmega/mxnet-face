#!/usr/bin/env sh

gcc -o cam cam.cpp -std=c++11 \
    -L../lib -lmxnet_predict -lstdc++ -lm -I../../mxnet/include \
    -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas \
    -L../lib -ldlib -I../../dlib/dlib-19.1 \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect
