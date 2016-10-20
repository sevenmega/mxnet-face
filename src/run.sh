#!/usr/bin/env sh

LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH \
    ./cam \
    ../model/dlib/shape_predictor_68_face_landmarks.dat \
    ../model/opencv/haarcascade_frontalface_alt.xml $1
