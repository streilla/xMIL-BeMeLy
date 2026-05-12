#!/bin/bash

python test_model.py \
--path-args trained_models/virchow2_abmil/ \
--path-df dummy_template.csv \
--path-features path/to/features \
--patch-encoder virchow2 \

python test_model.py \
--path-args trained_models/virchow2_transmil/ \
--path-df dummy_template.csv \
--path-features path/to/features \
--patch-encoder virchow2 \

python test_model.py \
--path-args trained_models/uni2_abmil/ \
--path-df dummy_template.csv \
--path-features path/to/features \
--patch-encoder 'uni_v2' \

python test_model.py \
--path-args trained_models/uni2_transmil/ \
--path-df dummy_template.csv \
--path-features path/to/features \
--patch-encoder 'uni_v2' \