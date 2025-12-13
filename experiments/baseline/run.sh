#!/bin/bash
set -e

EXP_NAME="baseline"
RESULT_DIR="results/$EXP_NAME"
mkdir -p $RESULT_DIR

uv run accelerate launch --config_file "experiments/$EXP_NAME/config.yaml" train_yolos_baseline_bf16.py \
    --output_dir $RESULT_DIR \
    --dataset coco2017 \
    --coco_dir ./coco \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 64 \
    --max_eval_samples 128 \
    --wandb_project dan1ar/edge-inference-research \
    --wandb_run_name $EXP_NAME
