#!/usr/bin/env bash
# E7 / C: gate-level activity simulation of all variants on the real YOLOS-tiny operands (CPU only).
# Usage: scripts/hw/array/run_power_sims.sh <var> [<var> ...]   (runs sequentially; start <= 3 in parallel)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=""
PY=${PY:-.venv/bin/python}
mkdir -p logs/hw_array
for v in "$@"; do
  nice -n 19 "$PY" scripts/hw/array/power_sim.py --var "$v" --df ws --files '*img0' > "logs/hw_array/power_ws_$v.log" 2>&1
  case "$v" in
    fp32|bf16|fp16|e8m13|kahan)
      nice -n 19 "$PY" scripts/hw/array/power_sim.py --var "$v" --df os --files '*img0' > "logs/hw_array/power_os_$v.log" 2>&1 ;;
  esac
done
