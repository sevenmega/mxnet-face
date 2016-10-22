#!/usr/bin/env sh

LD_LIBRARY_PATH=../../dlib/dlib/build:../../msgpack-c/build:$LD_LIBRARY_PATH \
    ./cam \
    ../model/dlib/shape_predictor_68_face_landmarks.dat \
    ../../ccv/samples/face.sqlite3 $1
