#!/usr/bin/env sh

classifier_path=../data/classifier
model_prefix=../model/lightened_cnn/lightened_cnn
epoch=166

python classifier.py --time 1 \
    --model-prefix $model_prefix \
    --epoch $epoch \
    --singleimg ./test.png
