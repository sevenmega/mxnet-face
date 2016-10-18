#!/usr/bin/env sh

model_prefix=../model/lightened_cnn/lightened_cnn
epoch=166

python classifier.py --singlerep 1 \
    --model-prefix $model_prefix \
    --epoch $epoch \
    --singleimg ../data/my-align/larry/image-10.png
