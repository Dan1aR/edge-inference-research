#!/usr/bin/env bash
# E7: array-level hardware evidence, CPU only (see docs/hw_array.md).  Total ~1.5-2 h on 3-6 cores.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=""
PY=${PY:-.venv/bin/python}
scripts/hw/fetch_hw_deps.sh && scripts/hw/array/fetch_tools.sh
nice -n 19 "$PY" scripts/hw/array/dump_operands.py                         # C: real bf16 operands
nice -n 19 "$PY" scripts/hw/array/verify_rtl.py --R 16 --C 16 --M 32 --ktiles 3 --jobs 3   # A: RTL vs model
nice -n 19 "$PY" scripts/hw/array/synth.py --jobs 3                        # B: synthesis
nice -n 19 "$PY" scripts/hw/array/semantics_check.py                       # hw vs software emulation
scripts/hw/array/run_power_sims.sh fp32 bf16 fp16 &                        # C: gate-level activity
scripts/hw/array/run_power_sims.sh e8m13 kahan chunk bf16_sep fp16_sep e8m13_sep; wait
scripts/hw/array/run_pnr.sh fmax                                           # B: P&R, achieved clock
# B/C: P&R at a common clock (5 ns; Kahan does not close at 5 ns -> 9 ns) + power with real activity
scripts/hw/array/run_pnr.sh iso "pe_ws_fp32,pe_ws_bf16,pe_ws_fp16,pe_ws_e8m13,pe_ws_chunk,pe_ws_bf16_sep,pe_ws_fp16_sep,pe_ws_e8m13_sep,pe_os_fp32,pe_os_bf16,pe_os_fp16,pe_os_e8m13,edge_top_fp32,edge_bot_fp32,edge_top_bf16,edge_bot_bf16,edge_top_fp16,edge_top_e8m13,edge_bot_e8m13,edge_bot_chunk,edge_top_chunk" 5.0
nice -n 19 "$PY" scripts/hw/array/pnr.py --blocks edge_west,edge_bot_fp16 --util 0.4 --period 5.0 --droute --tag _iso --out results/hw/array/pnr_iso_edges2.json
nice -n 19 "$PY" scripts/hw/array/pnr.py --blocks pe_ws_kahan,pe_os_kahan,edge_top_kahan,edge_bot_kahan --period 9.0 --droute --tag _iso --out results/hw/array/pnr_iso_kahan.json
nice -n 19 "$PY" scripts/hw/array/system_model.py                          # D: system level (zero-delay)
nice -n 19 "$PY" scripts/hw/array/system_model.py --mode unit --out results/hw/array/system_unitdelay.json
"$PY" scripts/hw/array/make_tables.py
