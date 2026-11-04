#!/usr/bin/env bash
# E6 hardware-side analysis, CPU only.  Outputs: results/hw/*.json, see docs/hw_analysis.md
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=""
PY=${PY:-.venv/bin/python}
nice -n 19 "$PY" scripts/hw/mac_count.py                      # step 1: MACs (analytic + instrumented)
scripts/hw/fetch_hw_deps.sh                                   # HardFloat, liberty files, yowasp-yosys
nice -n 19 "$PY" scripts/hw/run_synth.py --jobs 2             # step 4: synthesis + gate-level check
nice -n 19 "$PY" scripts/hw/run_synth.py --jobs 2 --no-patch \
    --out results/hw/synthesis_unpatched_hardfloat.json || true  # documents the HardFloat bf16 adder bug
nice -n 19 "$PY" scripts/hw/cost_model.py                     # steps 2-3: shares + cost model
