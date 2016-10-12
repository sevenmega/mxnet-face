#!/usr/bin/env sh

align_data_path=../data/lfw-align
model_prefix=../model/lightened_cnn/lightened_cnn
epoch=166
# evaluate on lfw
python lfw.py --lfw-align $align_data_path --model-prefix $model_prefix --epoch $epoch
