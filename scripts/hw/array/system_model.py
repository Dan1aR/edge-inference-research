#!/usr/bin/env python
"""E7 / D: system-level energy of one YOLOS-tiny forward (512x512, batch 1) per accumulator variant and
per mixed regime, weight-stationary (WS) and output-stationary (OS) R x R arrays.

MEASURED (this study, Nangate45, OpenROAD P&R + OpenSTA power with gate-level switching activity of
real YOLOS-tiny operands): energy per MAC of the PE (incl. its registers, local clock tree, leakage) and
of the array edge units (west/top converters, bottom converter, fp32 edge adder of 'chunk'),
per matmul type (patch, qkv, fc1, fc2, qk, av); results/hw/array/pnr_iso.json.
ASSUMED (literature, explicit parameters below): SRAM access energy per bit, buffer hierarchy, 100 %
array utilisation, no DRAM unless --dram, non-GEMM ops excluded (identical for all variants).

SRAM energy: M. Horowitz, "Computing's energy problem", ISSCC 2014 (45 nm, 0.9 V; same node as
Nangate45): 64-bit read 8 KB 10 pJ, 32 KB 20 pJ, 1 MB 100 pJ; write = read assumed; linear in bits.
DRAM (optional): 1.3-2.6 nJ / 64 bit (Horowitz) -> 30 pJ/bit used.

Traffic model per GEMM (M x K) @ (K x N), array R x R:
  WS : A from the 32 KB activation buffer once per N-tile (M*K*ceil(N/R) x 16 b); weights once
       (K*N x 16 b, from the 1 MB global buffer); partial sums: every output does ceil(K/R) passes,
       each pass reads (except the first) and writes the accumulator word in the 32 KB accumulator
       SRAM ((2*ceil(K/R)-1) x acc bits; chunk: fp32 words at the edge adder); final output 16 b write
       to the 1 MB global buffer.  Energy per array cycle: R*R PE + R west + R*(top + bottom) units.
  OS : A and B streamed from 32 KB buffers once per output tile (M*K*ceil(N/R) + K*N*ceil(M/R) x 16 b);
       accumulators stay in the PE (no psum SRAM traffic); output 16 b write.  OS-chunk: bf16 OS PE +
       per chunk of C MACs one fp32 edge add (measured edge_bot_chunk energy) + fp32 read/write of the
       edge accumulator register file (8 KB-class SRAM energy) + 17-bit drain over R/2 PE hops (energy
       of a 17-bit register transfer = 17 x measured flop energy per bit toggle x 0.5 activity).
  Global buffer: every GEMM reads its activation input once and writes its output once (1 MB SRAM).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results" / "hw" / "array"
SRAM_PJ_PER_BIT = {"8KB": 10 / 64, "32KB": 20 / 64, "1MB": 100 / 64}
DRAM_PJ_PER_BIT = 30.0
ACC_BITS = {"fp32": 32, "bf16": 16, "fp16": 16, "e8m13": 22, "kahan": 32, "chunk": 32,
            "bf16_sep": 16, "fp16_sep": 16, "e8m13_sep": 22}
# measured matmul activity set used for each component
MM_OF = {"patch": ["patch"], "qkv": ["qkv"], "attn_out": ["qkv"], "mlp": ["fc1", "fc2"], "head": ["fc1"],
         "qk": ["qk"], "av": ["av"]}
COMPONENTS = ["patch", "qkv", "attn_out", "mlp", "head", "qk", "av"]


def gemms(model):
    g = model["geometry"]
    L, N, d, ff, nh, dh = g["layers"], g["seq_len"], g["hidden"], g["intermediate"], g["heads"], g["head_dim"]
    n_det, ncls = g["n_det"], g["n_class_logits"]
    out = [("patch", "patch", g["n_patch"], 768, d, 1)]
    out += [("qkv", "qkv", N, d, 3 * d, L), ("attn_out", "qkv", N, d, d, L),
            ("mlp", "fc1", N, d, ff, L), ("mlp", "fc2", N, ff, d, L),
            ("qk", "qk", N, dh, N, L * nh), ("av", "av", N, N, dh, L * nh)]
    for n_out in (d, d, ncls, d, d, 4):   # detection heads (3-layer MLPs), K = d
        out.append(("head", "fc1", n_det, d, n_out, 1))
    tot = sum(M * K * Nn * c for _, _, M, K, Nn, c in out)
    assert tot == sum(v["macs"] for v in model["analytic"].values()), tot
    return out


def load_energy(pnr: dict, period_ns: float, mode: str):
    """block -> {matmul: pJ per cycle}."""
    E = {}
    for blk, r in pnr.items():
        E[blk] = {}
        for key, p in r.get("power", {}).items():
            stem, m = key.rsplit(":", 1)
            if m != mode or "total" not in p:
                continue
            if ("_sep_" in stem) != ("_sep" in blk):   # glob ws_bf16_* also matched ws_bf16_sep_*
                continue
            parts = stem.split("_")
            # stem = ws_<var>_<mm>_... ; var may contain '_sep'
            mm = parts[3] if parts[2] == "sep" else parts[2]
            E[blk][mm] = p["total"]["total_W"] * r["period_target_ns"] * 1e3   # W * ns = nJ -> pJ: *1e3
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnr", default="pnr_iso*.json", help="glob in results/hw/array (merged)")
    ap.add_argument("--macs", default=str(REPO / "results/hw/mac_counts.json"))
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--mode", default="zero", choices=["zero", "unit"])
    ap.add_argument("--out", default=str(RES / "system.json"))
    args = ap.parse_args()
    pnr = {}
    for f in sorted(RES.glob(args.pnr)):
        for k, v in json.loads(f.read_text()).items():
            if v.get("power") or k not in pnr:
                pnr[k] = v
    model = json.loads(Path(args.macs).read_text())["models"]["hustvl/yolos-tiny"]
    R, Cc = args.R, args.chunk
    E = load_energy(pnr, None, args.mode)
    if args.mode == "unit":   # edge units were simulated zero-delay only -> use those values for them
        E0 = load_energy(pnr, None, "zero")
        for blk, d in E0.items():
            for mm, v in d.items():
                E[blk].setdefault(mm, v)
    variants = [v for v in ACC_BITS if f"pe_ws_{v}" in E and E[f"pe_ws_{v}"]]
    G = gemms(model)
    total_macs = sum(M * K * N * c for _, _, M, K, N, c in G)
    sb = SRAM_PJ_PER_BIT

    def ws_energy(var, comp_filter=None):
        """pJ per inference by component for WS."""
        pe, bot, top = E[f"pe_ws_{var}"], E[f"edge_bot_{var.replace('_sep', '')}"], E[f"edge_top_{var.replace('_sep', '')}"]
        west = E["edge_west"]
        res = {}
        for comp, mm, M, K, N, cnt in G:
            macs = M * K * N * cnt
            e_pe = pe[mm] * macs
            e_edge = (west.get(mm, west.get("qkv", 0)) + top.get(mm, 0.0) + bot[mm]) / R * macs
            passes = math.ceil(K / R)
            outs = M * N * cnt
            e_psum = outs * (2 * passes - 1) * ACC_BITS[var] * sb["32KB"]
            e_a = M * K * math.ceil(N / R) * cnt * 16 * sb["32KB"]
            e_w = K * N * cnt * 16 * sb["1MB"]
            e_glob = (M * K + M * N) * cnt * 16 * sb["1MB"]
            d = res.setdefault(comp, dict(macs=0, pe=0.0, edge=0.0, psum_sram=0.0, operand_sram=0.0, global_buf=0.0))
            d["macs"] += macs; d["pe"] += e_pe; d["edge"] += e_edge; d["psum_sram"] += e_psum
            d["operand_sram"] += e_a + e_w; d["global_buf"] += e_glob
        return res

    def os_energy(var):
        res = {}
        chunk = var == "chunk"
        pe = E["pe_os_bf16" if chunk else f"pe_os_{var}"]
        bot = E.get("edge_bot_chunk", {})
        seq_ff_pj_bit = E.get("_ff_bit_pj", 0.0)
        for comp, mm, M, K, N, cnt in G:
            macs = M * K * N * cnt
            e_pe = pe[mm] * macs
            e_edge = 0.0
            if chunk:
                nchunks = M * N * cnt * math.ceil(K / Cc)
                e_edge = nchunks * (bot[mm] + 2 * 32 * sb["8KB"] + 17 * (R / 2) * 0.5 * seq_ff_pj_bit)
            e_ops = (M * K * math.ceil(N / R) + K * N * math.ceil(M / R)) * cnt * 16 * sb["32KB"]
            e_glob = (M * K + M * N) * cnt * 16 * sb["1MB"]
            d = res.setdefault(comp, dict(macs=0, pe=0.0, edge=0.0, psum_sram=0.0, operand_sram=0.0, global_buf=0.0))
            d["macs"] += macs; d["pe"] += e_pe; d["edge"] += e_edge; d["operand_sram"] += e_ops; d["global_buf"] += e_glob
        return res

    # flop energy per bit transfer (for the OS-chunk drain): from the bf16 OS PE sequential power share
    E["_ff_bit_pj"] = 0.0
    if "pe_os_bf16" in pnr:
        r = pnr["pe_os_bf16"]
        ps = [p["sequential"]["total_W"] for k, p in r["power"].items() if k.endswith(":" + args.mode) and "sequential" in p]
        if ps:
            E["_ff_bit_pj"] = sum(ps) / len(ps) * r["period_target_ns"] * 1e3 / 51   # pJ per flop per cycle

    table = {}
    for df, fn, vs in (("ws", ws_energy, variants),
                       ("os", os_energy, [v for v in ["fp32", "bf16", "fp16", "e8m13", "kahan", "chunk"]
                                          if f"pe_os_{'bf16' if v == 'chunk' else v}" in E])):
        per = {v: fn(v) for v in vs}
        table[df] = {}
        for v, comps in per.items():
            tot = {k: sum(c[k] for c in comps.values()) for k in ("pe", "edge", "psum_sram", "operand_sram", "global_buf")}
            tot["array"] = tot["pe"] + tot["edge"]
            tot["onchip_total"] = tot["array"] + tot["psum_sram"] + tot["operand_sram"] + tot["global_buf"]
            table[df][v] = dict(total_pJ=tot, per_component=comps,
                                array_pJ_per_mac=tot["array"] / total_macs)
        # mixed regimes: FP32-acc only in a subset of components, the rest with the given variant
        base = table[df]["fp32"]["total_pJ"]["onchip_total"]
        base_arr = table[df]["fp32"]["total_pJ"]["array"]
        regimes = {}
        for v in vs:
            for name, fp in (("all", set()), ("M1_fp32_qk_av", {"qk", "av"}), ("M2_fp32_av", {"av"})):
                comp_e = 0.0
                comp_arr = 0.0
                for c in COMPONENTS:
                    src = per["fp32"][c] if c in fp else per[v][c]
                    comp_e += src["pe"] + src["edge"] + src["psum_sram"] + src["operand_sram"] + src["global_buf"]
                    comp_arr += src["pe"] + src["edge"]
                regimes[f"{v}:{name}"] = dict(onchip_rel=comp_e / base, array_rel=comp_arr / base_arr,
                                             onchip_saving_pct=100 * (1 - comp_e / base),
                                             array_saving_pct=100 * (1 - comp_arr / base_arr),
                                             onchip_mJ=comp_e * 1e-9)
        table[df]["_regimes"] = regimes
    params = 6.5e6
    dram = params * 16 * DRAM_PJ_PER_BIT
    out = dict(assumptions=dict(sram_pJ_per_bit=SRAM_PJ_PER_BIT, dram_pJ_per_bit=DRAM_PJ_PER_BIT, R=R, chunk=Cc,
                                activity_mode=args.mode, acc_bits=ACC_BITS, matmul_proxy=MM_OF,
                                dram_weights_once_pJ=dram, note="DRAM weight fetch is identical for all variants"),
               total_macs=total_macs, dataflows=table)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for df in ("ws", "os"):
        print(f"== {df.upper()} {R}x{R}, activity {args.mode}")
        print(f"{'variant':10s} {'array pJ/MAC':>12s} {'array rel':>9s} {'on-chip mJ':>10s} {'rel':>6s} {'+DRAM rel':>9s}")
        b = table[df]["fp32"]["total_pJ"]
        for v, r in table[df].items():
            if v.startswith("_"):
                continue
            t = r["total_pJ"]
            print(f"{v:10s} {r['array_pJ_per_mac']:12.3f} {t['array']/b['array']:9.3f} {t['onchip_total']*1e-9:10.3f}"
                  f" {t['onchip_total']/b['onchip_total']:6.3f} {(t['onchip_total']+dram)/(b['onchip_total']+dram):9.3f}")
        for k, r in table[df]["_regimes"].items():
            if not k.endswith(":all"):
                print(f"   {k:22s} on-chip saving {r['onchip_saving_pct']:5.1f}%  array saving {r['array_saving_pct']:5.1f}%")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
