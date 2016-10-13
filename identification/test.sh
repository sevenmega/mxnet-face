#!/usr/bin/env sh

classifier_path=../data/classifier
model_prefix=../model/lightened_cnn/lightened_cnn
epoch=166

# test on a single image
python classifier.py --infer 1 \
    --classifier-model $classifier_path/classifier.pkl \
    --model-prefix $model_prefix \
    --epoch $epoch \
    --singleimg ./test.png
