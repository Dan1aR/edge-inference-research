#!/usr/bin/env bash
# E7 / B: OpenROAD P&R runs (CPU only, nice 19, one openroad process per invocation).
#   fmax : all PEs at an unreachable 1.0 ns target, global-route parasitics -> achieved period
#   iso  : all PEs and edge units at a common clock (default 5 ns), detailed route + OpenRCX, power
#          with the gate-level activity of power_sim.py
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-.venv/bin/python}
PES_WS="pe_ws_fp32,pe_ws_bf16,pe_ws_fp16,pe_ws_e8m13,pe_ws_kahan,pe_ws_chunk,pe_ws_bf16_sep,pe_ws_fp16_sep,pe_ws_e8m13_sep"
PES_OS="pe_os_fp32,pe_os_bf16,pe_os_fp16,pe_os_e8m13,pe_os_kahan"
EDGES="edge_west,edge_top_fp32,edge_top_bf16,edge_top_fp16,edge_top_e8m13,edge_top_kahan,edge_top_chunk,edge_bot_fp32,edge_bot_bf16,edge_bot_fp16,edge_bot_e8m13,edge_bot_kahan,edge_bot_chunk"
case "$1" in
  fmax) nice -n 19 "$PY" scripts/hw/array/pnr.py --blocks "$PES_WS,$PES_OS" --period 1.0 --acts /nonexistent \
          --tag _fmax --out results/hw/array/pnr_fmax.json ;;
  iso)  nice -n 19 "$PY" scripts/hw/array/pnr.py --blocks "${2:-$PES_WS,$PES_OS,$EDGES}" --period "${3:-5.0}" --droute \
          --tag _iso --out results/hw/array/pnr_iso.json ;;
esac
