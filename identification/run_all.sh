#!/usr/bin/env sh

align_data_path=../data/my-align
feature_path=../data/feature
classifier_path=../data/classifier
model_prefix=../model/lightened_cnn/lightened_cnn
epoch=166

#python classifier.py --singlerep 1 \
#    --model-prefix $model_prefix \
#    --epoch $epoch \
#    --singleimg ./test.png
python classifier.py --feature 1 \
    --aligned-prefix $align_data_path \
    --feature-prefix $feature_path \
    --model-prefix $model_prefix \
    --epoch $epoch
python classifier.py --train 1 \
    --classifier-prefix $classifier_path \
    --feature-prefix $feature_path
python classifier.py --infer 1 \
    --classifier-model $classifier_path/classifier.pkl \
    --model-prefix $model_prefix \
    --epoch $epoch \
    --singleimg ./test.png
