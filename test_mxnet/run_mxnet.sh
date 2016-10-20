#!/usr/bin/env sh

LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH ./image-classification-predict \
    ../data/my-align/larry/image-10.png
