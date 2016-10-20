#!/usr/bin/env sh

g++ -o image-classification-predict image-classification-predict.cc -std=c++11 \
    -L../lib -lmxnet_predict -lstdc++ -lm -I../../mxnet/include \
    -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect
