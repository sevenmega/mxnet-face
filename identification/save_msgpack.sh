#!/usr/bin/env sh

feature_path=../data/feature

python classifier.py --msgpack 1 \
    --feature-prefix $feature_path
