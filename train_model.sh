#!/bin/bash

python train_model.py \
--aggregation-model transmil \
--patch-encoder virchow2 \
--path-checkpoint trained_models/virchow2_transmil/ \
--path-features path/to/features \
--path-df dummy_template.csv \
--batch-size 1 \
--device cpu \
--epochs 20 \

python train_model.py \
--aggregation-model attention_mil \
--patch-encoder virchow2 \
--path-checkpoint trained_models/virchow2_abmil/ \
--path-features path/to/features \
--path-df dummy_template.csv \
--batch-size 1 \
--epochs 20 \

python train_model.py \
--aggregation-model transmil \
--patch-encoder uni_v2 \
--path-checkpoint trained_models/uni2_transmil/ \
--path-features path/to/features \
--path-df dummy_template.csv \
--batch-size 8 \
--device cpu \
--epochs 20 \

python train_model.py \
--aggregation-model attention_mil \
--patch-encoder uni_v2 \
--path-checkpoint trained_models/uni2_transmil/ \
--path-features path/to/features \
--path-df dummy_template.csv \
--batch-size 8 \
--epochs 20 \