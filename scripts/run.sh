#!/bin/bash

python3 -m src.run_experiment \
    --coco-root ~/dc-remote/coco \
    --precision bf16_accum \
    --batch-size 64 \
    --no-bf16-accum-patch-embed \
    --no-bf16-accum-attention \
    --output results/results_bf16_accum_lin1_emb0_attn0.json
