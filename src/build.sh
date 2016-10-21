#!/usr/bin/env sh

gcc -o cam cam.cpp -std=c++11 \
    -I../../dlib -L../../dlib/dlib/build -ldlib \
    -I../../mxnet/include -L../../mxnet/amalgamation -lmxnet_predict \
    -I../../ccv/lib -L../../ccv/lib -lccv \
    -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect \
    -lpng -ljpeg \
    -lstdc++ -lm

