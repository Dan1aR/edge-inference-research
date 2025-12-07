# 1) lin=1, patch=1, attn=1  (all BF16-accum enabled)
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --bf16-accum-linear \
  --bf16-accum-patch-embed \
  --bf16-accum-attention \
  --output results/bf16_accum_lin1_patch1_attn1.json \
  > logs/bf16_accum_lin1_patch1_attn1.log 2>&1 &

# 2) lin=0, patch=1, attn=1
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --no-bf16-accum-linear \
  --bf16-accum-patch-embed \
  --bf16-accum-attention \
  --output results/bf16_accum_lin0_patch1_attn1.json \
  > logs/bf16_accum_lin0_patch1_attn1.log 2>&1 &

# 3) lin=1, patch=0, attn=1
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --bf16-accum-linear \
  --no-bf16-accum-patch-embed \
  --bf16-accum-attention \
  --output results/bf16_accum_lin1_patch0_attn1.json \
  > logs/bf16_accum_lin1_patch0_attn1.log 2>&1 &

# 4) lin=1, patch=1, attn=0
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --bf16-accum-linear \
  --bf16-accum-patch-embed \
  --no-bf16-accum-attention \
  --output results/bf16_accum_lin1_patch1_attn0.json \
  > logs/bf16_accum_lin1_patch1_attn0.log 2>&1 &

# 5) lin=0, patch=0, attn=1
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --no-bf16-accum-linear \
  --no-bf16-accum-patch-embed \
  --bf16-accum-attention \
  --output results/bf16_accum_lin0_patch0_attn1.json \
  > logs/bf16_accum_lin0_patch0_attn1.log 2>&1 &

# 6) lin=0, patch=1, attn=0
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --no-bf16-accum-linear \
  --bf16-accum-patch-embed \
  --no-bf16-accum-attention \
  --output results/bf16_accum_lin0_patch1_attn0.json \
  > logs/bf16_accum_lin0_patch1_attn0.log 2>&1 &

# 7) lin=1, patch=0, attn=0
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --bf16-accum-linear \
  --no-bf16-accum-patch-embed \
  --no-bf16-accum-attention \
  --output results/bf16_accum_lin1_patch0_attn0.json \
  > logs/bf16_accum_lin1_patch0_attn0.log 2>&1 &

# 8) lin=0, patch=0, attn=0  (all BF16-accum disabled within this mode)
nohup uv run python -m src.run_experiment \
  --precision bf16_accum --coco-root ./coco \
  --no-bf16-accum-linear \
  --no-bf16-accum-patch-embed \
  --no-bf16-accum-attention \
  --output results/bf16_accum_lin0_patch0_attn0.json \
  > logs/bf16_accum_lin0_patch0_attn0.log 2>&1 &
