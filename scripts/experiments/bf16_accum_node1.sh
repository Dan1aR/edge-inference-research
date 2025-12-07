# Node 1: 2 parallel runs
mkdir -p results logs

# 1) lin=1, patch=1, attn=1  (all BF16-accum enabled)
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --bf16-accum-linears \
  --bf16-accum-patch-embed \
  --bf16-accum-attention \
  --output results/bf16_accum_lin1_patch1_attn1.json \
  > logs/bf16_accum_lin1_patch1_attn1.log 2>&1 &

# 2) lin=0, patch=1, attn=1
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --no-bf16-accum-linears \
  --bf16-accum-patch-embed \
  --bf16-accum-attention \
  --output results/bf16_accum_lin0_patch1_attn1.json \
  > logs/bf16_accum_lin0_patch1_attn1.log 2>&1 &

