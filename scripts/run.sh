#!/bin/bash

python3 -m src.run_experiment \
    --coco-root ~/dc-remote/coco \
    --precision bf16_accum \
    --batch-size 64 \
    --output results/results_bf16_accum_lin1_emb1_attn1.json
