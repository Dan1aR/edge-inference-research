#!/bin/bash
set -e
source .venv/bin/activate

EXP_NAME="baseline"
RESULT_DIR="results/$EXP_NAME"
mkdir -p $RESULT_DIR

accelerate launch --config_file "experiments/$EXP_NAME/config.yaml" train_yolos_baseline_bf16.py \
    --output_dir $RESULT_DIR \
    --dataset coco2017 \
    --coco_dir ./coco \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --warmup_ratio 0.15 \
    --max_steps 10000 \
    --lr_scheduler cosine \
    --report_to wandb \
    --precision bf16 \
    --wandb_project "devcluster-test" \
    --wandb_run_name $EXP_NAME
