#!/usr/bin/env python
"""E6 / steps 2-3: FP32-accumulation share per accumulation regime + analytic hardware cost model.

Input : results/hw/mac_counts.json            (scripts/hw/mac_count.py)
        results/hw/synthesis.json  (optional)  (scripts/hw/run_synth.py) -> area-proxy column
        results/bf16_accum_lin*_patch*_attn*.json (optional, existing COCO AP of yolos-tiny)
Output: results/hw/regimes_cost.json

ALL HARDWARE NUMBERS HERE ARE ESTIMATES FROM AN ANALYTIC MODEL, not measurements.

Energy per MAC (Horowitz, "Computing's energy problem", ISSCC 2014, 45 nm, 0.9 V):
    16-bit FP add 0.4 pJ, 32-bit FP add 0.9 pJ, 16-bit FP mult 1.1 pJ, 32-bit FP mult 3.7 pJ,
    SRAM read (64 bit): 8 KB 10 pJ, 32 KB 20 pJ, 1 MB 100 pJ.
  bf16 x bf16 is exact in FP32 (8b x 8b significand = 16 b <= 24 b), so
    MAC with FP32 accumulation  = 16-bit mult + 32-bit add = 1.1 + 0.9 = 2.0 pJ
    MAC with BF16 accumulation  = 16-bit mult + 16-bit add = 1.1 + 0.4 = 1.5 pJ
  Horowitz's 16-bit FP is IEEE fp16 (11-bit significand); a bf16 multiplier (8-bit significand) is
  smaller.  Sensitivity variant 'bf16mult': mult energy scaled by (8/11)^2 -> 0.58 pJ.
  Only arithmetic datapath energy is modelled (no operand SRAM/DRAM, no control, no NoC).

Partial-sum storage:
  * output-stationary (OS) RxC systolic array: one accumulator register per PE (32 b vs 16 b);
  * split-K spill: if K exceeds the on-chip K-tile T, partial sums are written/read back
    (ceil(K/T)-1) times per output -> 2*(ceil(K/T)-1)*bytes per output;
  * weight-stationary (WS) RxC array (TPU-like): K is mapped onto the R rows, every output receives
    ceil(K/R) read-modify-write updates in an accumulator SRAM -> (2*ceil(K/R)-1)*bytes per output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMPONENTS = ["patch", "qkv", "attn_out", "mlp", "head", "qk", "av"]
FLAG_OF = {"patch": "patch", "qkv": "lin", "attn_out": "lin", "mlp": "lin", "head": "lin",
           "qk": "attn", "av": "attn"}

HOROWITZ = dict(add16=0.4, add32=0.9, mul16=1.1, mul32=3.7, sram8k_64b=10.0, sram32k_64b=20.0,
                sram1m_64b=100.0, source="M. Horowitz, ISSCC 2014, 45nm 0.9V")
E_MUL_BF16_SCALED = HOROWITZ["mul16"] * (8 / 11) ** 2


def regimes() -> dict:
    """regime -> set of components that use BF16 accumulation (rest: FP32 accumulation)."""
    r = {
        "B (all FP32-acc)": set(),
        "I (all BF16-acc)": set(COMPONENTS),
        "M1 (FP32-acc only qk,av)": set(COMPONENTS) - {"qk", "av"},
        "M2 (FP32-acc only av)": set(COMPONENTS) - {"av"},
        "M3 (FP32-acc only qk)": set(COMPONENTS) - {"qk"},
    }
    # all 8 combinations of the three switches of src/run_experiment.py
    for lin in (0, 1):
        for patch in (0, 1):
            for attn in (0, 1):
                on = {"lin": lin, "patch": patch, "attn": attn}
                r[f"lin{lin}_patch{patch}_attn{attn}"] = {c for c in COMPONENTS if on[FLAG_OF[c]]}
    return r


ALIASES = {
    "lin0_patch0_attn0": "= B", "lin1_patch1_attn1": "= I", "lin1_patch1_attn0": "= M1",
    "lin1_patch0_attn0": "lin-only-BF16", "lin0_patch1_attn0": "patch-only-BF16",
    "lin0_patch0_attn1": "attn-only-BF16", "lin1_patch0_attn1": "lin+attn BF16 (patch FP32)",
    "lin0_patch1_attn1": "patch+attn BF16 (lin FP32)",
}


def per_output_spill_bytes(K: int, tile: int, nbytes: int) -> int:
    return 2 * (math.ceil(K / tile) - 1) * nbytes


def ws_acc_bytes(K: int, rows: int, nbytes: int) -> int:
    return (2 * math.ceil(K / rows) - 1) * nbytes


def gemm_list(model: dict):
    """Expand components to (component, K, outputs) with per-K output counts."""
    g = model["geometry"]
    L, N, d, ff, nh, dh = g["layers"], g["seq_len"], g["hidden"], g["intermediate"], g["heads"], g["head_dim"]
    comp = model["analytic"]
    rows = [
        ("patch", comp["patch"]["K"][0], comp["patch"]["outputs"]),
        ("qkv", d, comp["qkv"]["outputs"]),
        ("attn_out", d, comp["attn_out"]["outputs"]),
        ("mlp", d, L * N * ff),   # intermediate.dense (K=d)
        ("mlp", ff, L * N * d),   # output.dense (K=ff)
        ("head", d, comp["head"]["outputs"]),
        ("qk", dh, comp["qk"]["outputs"]),
        ("av", N, comp["av"]["outputs"]),
    ]
    assert sum(o for c, k, o in rows if c == "mlp") == comp["mlp"]["outputs"]
    return rows


def analyse(model: dict, area_ratio: float | None, os_sizes, k_tiles, ws_rows):
    comp = model["analytic"]
    tot_mac = sum(comp[c]["macs"] for c in COMPONENTS)
    tot_out = sum(comp[c]["outputs"] for c in COMPONENTS)
    rows = gemm_list(model)
    e32 = HOROWITZ["mul16"] + HOROWITZ["add32"]
    e16 = HOROWITZ["mul16"] + HOROWITZ["add16"]
    e32s = E_MUL_BF16_SCALED + HOROWITZ["add32"]
    e16s = E_MUL_BF16_SCALED + HOROWITZ["add16"]
    out = {}
    for name, bf in regimes().items():
        fp = [c for c in COMPONENTS if c not in bf]
        mac32 = sum(comp[c]["macs"] for c in fp)
        out32 = sum(comp[c]["outputs"] for c in fp)
        s_mac = mac32 / tot_mac
        s_out = out32 / tot_out
        r = dict(
            fp32_acc_components=fp,
            fp32_mac_share=s_mac, fp32_output_share=s_out,
            energy_rel_horowitz=(s_mac * e32 + (1 - s_mac) * e16) / e32,
            energy_rel_horowitz_bf16mult=(s_mac * e32s + (1 - s_mac) * e16s) / e32s,
            energy_pJ_per_inference_horowitz=(mac32 * e32 + (tot_mac - mac32) * e16),
            # one partial-sum word per output: total bytes of accumulator results written once
            psum_result_bytes_rel=(out32 * 4 + (tot_out - out32) * 2) / (tot_out * 4),
        )
        if area_ratio is not None:  # energy ~ area proxy from synthesis (crude)
            r["energy_rel_area_proxy"] = s_mac + (1 - s_mac) * area_ratio
        # split-K spill traffic (any array, K tiled with tile T)
        spill = {}
        for T in k_tiles:
            base = sum(o * per_output_spill_bytes(k, T, 4) for c, k, o in rows)
            cur = sum(o * per_output_spill_bytes(k, T, 4 if c in fp else 2) for c, k, o in rows)
            spill[f"T={T}"] = dict(bytes_B=base, bytes=cur, rel=(cur / base) if base else None,
                                   sram_energy_uJ=cur / 8 * HOROWITZ["sram32k_64b"] * 1e-6)
        r["splitK_spill"] = spill
        ws = {}
        for R in ws_rows:
            base = sum(o * ws_acc_bytes(k, R, 4) for c, k, o in rows)
            cur = sum(o * ws_acc_bytes(k, R, 4 if c in fp else 2) for c, k, o in rows)
            ws[f"{R}x{R}"] = dict(acc_sram_bytes_B=base, acc_sram_bytes=cur, rel=cur / base,
                                  acc_sram_energy_uJ=cur / 8 * HOROWITZ["sram8k_64b"] * 1e-6)
        r["ws_accumulator_sram_traffic"] = ws
        osr = {}
        for R in os_sizes:
            homog = (R * R * (32 if fp else 16)) / 8
            osr[f"{R}x{R}"] = dict(
                acc_register_bytes_B=R * R * 4,
                acc_register_bytes_homogeneous=homog,
                rel_homogeneous=homog / (R * R * 4),
                note="homogeneous array: 32-b accumulators are required as soon as any GEMM uses FP32-acc",
            )
        r["os_accumulator_registers"] = osr
        out[name] = r
    return out, dict(total_macs=tot_mac, total_outputs=tot_out,
                     e_mac_fp32acc_pJ=e32, e_mac_bf16acc_pJ=e16,
                     e_mac_fp32acc_pJ_bf16mult=e32s, e_mac_bf16acc_pJ_bf16mult=e16s)


def load_ap():
    ap = {}
    for p in sorted((REPO / "results").glob("bf16_accum_lin*_patch*_attn*.json")):
        try:
            ap[p.stem.replace("bf16_accum_", "")] = json.loads(p.read_text())["metrics"]["AP"]
        except Exception:
            pass
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macs", default=str(REPO / "results/hw/mac_counts.json"))
    ap.add_argument("--synth", default=str(REPO / "results/hw/synthesis.json"))
    ap.add_argument("--os-sizes", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--k-tiles", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--ws-rows", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--out", default=str(REPO / "results/hw/regimes_cost.json"))
    args = ap.parse_args()
    macs = json.loads(Path(args.macs).read_text())
    area_ratio = None
    area_src = None
    sp = Path(args.synth)
    if sp.exists():
        s = json.loads(sp.read_text())
        d = s["designs"]
        # combinational area of the arithmetic (excluding registers) of the separate-unit PEs
        area_ratio = (d["pe_bf16acc_sep"]["nangate45"]["combinational_area_um2"]
                      / d["pe_fp32acc_sep"]["nangate45"]["combinational_area_um2"])
        area_src = "pe_bf16acc_sep / pe_fp32acc_sep combinational area (Nangate45, results/hw/synthesis.json)"
    res = dict(
        disclaimer="analytic estimate; not a measurement on hardware",
        horowitz=HOROWITZ, e_mul_bf16_scaled_pJ=E_MUL_BF16_SCALED,
        area_proxy_ratio_bf16acc_over_fp32acc=area_ratio, area_proxy_source=area_src,
        aliases=ALIASES, coco_AP_yolos_tiny_existing_results=load_ap(), models={},
    )
    for mid, model in macs["models"].items():
        reg, tot = analyse(model, area_ratio, args.os_sizes, args.k_tiles, args.ws_rows)
        res["models"][mid] = dict(totals=tot, regimes=reg)
        print(f"== {mid}: {tot['total_macs']:,} MACs, {tot['total_outputs']:,} accumulator outputs")
        print(f"{'regime':28s} {'FP32 MAC%':>9s} {'FP32 out%':>9s} {'E/E_B':>6s} {'E/E_B*':>6s} "
              f"{'psum%':>6s} {'spillT128':>9s} {'WS32':>6s} {'WS128':>6s}")
        for n, r in reg.items():
            print(f"{n:28s} {100*r['fp32_mac_share']:9.2f} {100*r['fp32_output_share']:9.2f} "
                  f"{r['energy_rel_horowitz']:6.3f} {r['energy_rel_horowitz_bf16mult']:6.3f} "
                  f"{100*r['psum_result_bytes_rel']:6.1f} {r['splitK_spill']['T=128']['rel']:9.3f} "
                  f"{r['ws_accumulator_sram_traffic']['32x32']['rel']:6.3f} "
                  f"{r['ws_accumulator_sram_traffic']['128x128']['rel']:6.3f}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
