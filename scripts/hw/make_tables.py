#!/usr/bin/env python
"""E6: print the markdown tables used in docs/hw_analysis.md from results/hw/*.json."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R = REPO / "results/hw"
COMPS = ["patch", "qkv", "attn_out", "mlp", "head", "qk", "av"]
REG_ROWS = [
    ("B (all FP32-acc)", "B — всё FP32-acc", "lin0_patch0_attn0"),
    ("I (all BF16-acc)", "I — всё BF16-acc", "lin1_patch1_attn1"),
    ("M1 (FP32-acc only qk,av)", "M1 — FP32-acc только qk, av", "lin1_patch1_attn0"),
    ("M2 (FP32-acc only av)", "M2 — FP32-acc только av", None),
    ("M3 (FP32-acc only qk)", "M3 — FP32-acc только qk", None),
    ("lin1_patch0_attn0", "lin-only-BF16 (FP32: patch, qk, av)", "lin1_patch0_attn0"),
    ("lin0_patch0_attn1", "attn-only-BF16 (FP32: patch, все Linear)", "lin0_patch0_attn1"),
    ("lin0_patch1_attn0", "patch-only-BF16", "lin0_patch1_attn0"),
    ("lin1_patch0_attn1", "lin+attn BF16 (FP32: только patch)", "lin1_patch0_attn1"),
    ("lin0_patch1_attn1", "patch+attn BF16 (FP32: Linear)", "lin0_patch1_attn1"),
]


def fmt_int(x):
    return f"{x:,}".replace(",", " ")


def main():
    macs = json.loads((R / "mac_counts.json").read_text())
    cost = json.loads((R / "regimes_cost.json").read_text())
    syn = json.loads((R / "synthesis.json").read_text())
    for mid, m in macs["models"].items():
        g = m["geometry"]
        a = m["analytic"]
        tm, to = m["totals"]["macs"], m["totals"]["outputs"]
        print(f"\n#### {mid} (d={g['hidden']}, L={g['layers']}, heads={g['heads']}, d_h={g['head_dim']}, "
              f"d_ff={g['intermediate']}, N={g['seq_len']})\n")
        print("| компонент | K | MAC | доля MAC | выходов (аккумуляторов) | доля выходов |")
        print("|---|---|---:|---:|---:|---:|")
        for c in COMPS:
            v = a[c]
            print(f"| {c} | {'/'.join(map(str, v['K']))} | {fmt_int(v['macs'])} | {100*v['macs']/tm:.2f}% | "
                  f"{fmt_int(v['outputs'])} | {100*v['outputs']/to:.2f}% |")
        print(f"| **всего** | | **{fmt_int(tm)}** | 100% | **{fmt_int(to)}** | 100% |")
        print(f"\nИнструментирование: {fmt_int(m['totals']['instrumented_macs'])} MAC; FlopCounter eager/2 = "
              f"{fmt_int(m['totals']['flopcounter_eager_flops']//2)}; sdpa(math)/2 = "
              f"{fmt_int(m['totals']['flopcounter_sdpa_flops']//2)}; совпадение: {m['agree']}")
    ap = cost["coco_AP_yolos_tiny_existing_results"]
    for mid, mm in cost["models"].items():
        print(f"\n#### {mid}\n")
        print("| режим | FP32-acc MAC | FP32-acc выходов | E/E_B (Horowitz) | E/E_B (bf16-mult) | "
              "площадь/E (синтез-прокси) | psum-байт | split-K T=128 | WS 32×32 | WS 128×128 |"
              + (" AP (tiny) |" if "tiny" in mid else ""))
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if "tiny" in mid else ""))
        for key, label, apkey in REG_ROWS:
            r = mm["regimes"][key]
            row = (f"| {label} | {100*r['fp32_mac_share']:.2f}% | {100*r['fp32_output_share']:.2f}% | "
                   f"{r['energy_rel_horowitz']:.3f} | {r['energy_rel_horowitz_bf16mult']:.3f} | "
                   f"{r['energy_rel_area_proxy']:.3f} | {100*r['psum_result_bytes_rel']:.1f}% | "
                   f"{r['splitK_spill']['T=128']['rel']:.3f} | "
                   f"{r['ws_accumulator_sram_traffic']['32x32']['rel']:.3f} | "
                   f"{r['ws_accumulator_sram_traffic']['128x128']['rel']:.3f} |")
            if "tiny" in mid:
                row += f" {ap[apkey]:.4f} |" if apkey and apkey in ap else " — |"
            print(row)
        t = mm["totals"]
        b = mm["regimes"]["B (all FP32-acc)"]
        print(f"\nB: datapath {b['energy_pJ_per_inference_horowitz']/1e9:.2f} mJ/инференс; "
              f"split-K spill T=64/128/256: " + ", ".join(
                  f"{v['bytes_B']/2**20:.1f} MiB ({v['sram_energy_uJ']/1e3:.2f} mJ)" for v in b["splitK_spill"].values())
              + "; WS acc-SRAM 32/128: " + ", ".join(
                  f"{v['acc_sram_bytes_B']/2**20:.1f} MiB ({v['acc_sram_energy_uJ']/1e3:.2f} mJ)"
                  for v in b["ws_accumulator_sram_traffic"].values()))
    print("\n#### synthesis\n")
    print("| блок | 2-вх. вентилей (generic) | FF | Nangate45, мкм² | из них FF | NAND2-экв. | sky130hd, мкм² |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for d, v in syn["designs"].items():
        n, s = v["nangate45"], v["sky130hd"]
        print(f"| `{d}` | {v['generic']['cells']} | {n['flipflops']} | {n['area_um2']:.0f} | "
              f"{n['sequential_area_um2']:.0f} | {n['nand2_eq']:.0f} | {s['area_um2']:.0f} |")
    print()
    for k in syn["ratios"]["nangate45"]:
        print(f"| {k} | {syn['ratios']['nangate45'][k]:.2f} | {syn['ratios']['sky130hd'][k]:.2f} |")
    print()
    for t, c in syn["functional_check"].items():
        print(t, c["vectors"], c["mismatches"])
    un = R / "synthesis_unpatched_hardfloat.json"
    if un.exists():
        u = json.loads(un.read_text())
        print("unpatched:", {t: c["mismatches"] for t, c in u["functional_check"].items()})


if __name__ == "__main__":
    main()
