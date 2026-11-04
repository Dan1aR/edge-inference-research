#!/usr/bin/env python
"""E7: print the markdown tables of docs/hw_array.md from results/hw/array/*.json."""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[3] / "results" / "hw" / "array"
VARS = ["fp32", "bf16", "fp16", "e8m13", "kahan", "chunk", "bf16_sep", "fp16_sep", "e8m13_sep"]
MMS = ["patch", "qkv", "fc1", "fc2", "qk", "av"]


def load(n):
    p = RES / n
    return json.loads(p.read_text()) if p.exists() else {}


def mm_of(key):
    stem = key.rsplit(":", 1)[0]
    p = stem.split("_")
    return p[3] if p[2] == "sep" else p[2]


def energy(r, mode):
    out = {}
    for k, p in r.get("power", {}).items():
        if k.endswith(":" + mode) and "total" in p and ("_sep_" in k) == ("_sep" in r.get("block", "")):
            out[mm_of(k)] = p["total"]["total_W"] * r["period_target_ns"] * 1e3
    return out


def main():
    syn = load("synth.json").get("blocks", {})
    iso = {}
    for f in sorted(RES.glob("pnr_iso*.json")):
        for k, v in json.loads(f.read_text()).items():
            if v.get("power") or k not in iso:
                iso[k] = v
    fmx = load("pnr_fmax.json")
    print("### synthesis (Nangate45 / sky130hd, um2)\n")
    print("| variant | PE_WS N45 | FF | PE_OS N45 | PE_WS sky130 | arr16 N45 | arr32 N45 | arr16/fp32 | arr32/fp32 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    a16 = syn.get("arr16_fp32", {}).get("nangate45_area", {}).get("area_um2", 1)
    a32 = syn.get("arr32_fp32", {}).get("nangate45_area", {}).get("area_um2", 1)
    for v in VARS:
        pe = syn.get(f"pe_ws_{v}", {})
        if not pe:
            continue
        os_ = syn.get(f"pe_os_{v}", {}).get("nangate45_area", {}).get("area_um2")
        r16 = syn.get(f"arr16_{v}", {}).get("nangate45_area", {}).get("area_um2")
        r32 = syn.get(f"arr32_{v}", {}).get("nangate45_area", {}).get("area_um2")
        f = lambda x: f"{x:,.0f}" if x else "—"  # noqa: E731
        g = lambda x, b: f"{x / b:.3f}" if x else "—"  # noqa: E731
        print(f"| {v} | {f(pe['nangate45_area']['area_um2'])} | {pe['nangate45_area']['flipflops']} | {f(os_)} | "
              f"{f(pe['sky130hd_area']['area_um2'])} | {f(r16)} | {f(r32)} | {g(r16, a16)} | {g(r32, a32)} |")
    for v in ("west", "top", "bot"):
        pass
    print("\n### P&R (OpenROAD, Nangate45)\n")
    print("| block | cell area iso | WNS iso (ns) | period target fmax run | achieved period (ns) | fmax MHz | cell area fmax run |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for b in sorted(set(iso) | set(fmx)):
        i, f = iso.get(b, {}), fmx.get(b, {})
        print(f"| {b} | {i.get('cell_area_um2', '—')} | {i.get('wns_ns', '—')} | {f.get('period_target_ns', '—')} | "
              f"{round(f['achieved_period_ns'], 3) if f.get('achieved_period_ns') else '—'} | "
              f"{round(f['fmax_MHz']) if f.get('fmax_MHz') else '—'} | {f.get('cell_area_um2', '—')} |")
    for mode in ("zero", "unit"):
        print(f"\n### energy per MAC, PE only (pJ), activity {mode}-delay, iso-frequency P&R\n")
        base = energy(iso.get("pe_ws_fp32", {}), mode)
        for df in ("ws", "os"):
            print(f"\n{df.upper()}\n")
            print("| variant | " + " | ".join(MMS) + " | mean | ratio to fp32 (per matmul) |")
            print("|---|" + "---:|" * (len(MMS) + 2))
            b = energy(iso.get(f"pe_{df}_fp32", {}), mode)
            for v in VARS:
                e = energy(iso.get(f"pe_{df}_{v}", {}), mode)
                if not e:
                    continue
                vals = [e.get(m) for m in MMS]
                ok = [x for x in vals if x]
                rat = [e[m] / b[m] for m in MMS if m in e and m in b]
                print(f"| {v} | " + " | ".join(f"{x:.2f}" if x else "—" for x in vals) +
                      f" | {sum(ok) / len(ok):.2f} | " + " ".join(f"{x:.2f}" for x in rat) + " |")


if __name__ == "__main__":
    main()
