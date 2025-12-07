#!/usr/bin/env bash
set -euo pipefail

mkdir -p results logs

# 3) lin=1, patch=0, attn=1
uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --bf16-accum-linears \
  --no-bf16-accum-patch-embed \
  --bf16-accum-attention \
  --output results/bf16_accum_lin1_patch0_attn1.json \
  > logs/bf16_accum_lin1_patch0_attn1.log 2>&1

# 4) lin=1, patch=1, attn=0
uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --bf16-accum-linears \
  --bf16-accum-patch-embed \
  --no-bf16-accum-attention \
  --output results/bf16_accum_lin1_patch1_attn0.json \
  > logs/bf16_accum_lin1_patch1_attn0.log 2>&1

# 5) lin=0, patch=0, attn=1
uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --no-bf16-accum-linears \
  --no-bf16-accum-patch-embed \
  --bf16-accum-attention \
  --output results/bf16_accum_lin0_patch0_attn1.json \
  > logs/bf16_accum_lin0_patch0_attn1.log 2>&1
