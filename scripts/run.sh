#!/bin/bash

uv run python3 -m src.run_experiment \
    --coco-root ./coco \
    --precision bf16_default \
    --batch-size 64 \
    --output results/results_bf16_default.json
