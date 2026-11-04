"""Block D final table: method x inference arithmetic, mean +- sd over seeds, n, paired p vs ref_I and vs ref_B.

Source: results/x7/<method>_s<seed>_<arith>.json (scripts/eval_grid.py); the pretrained model is
results/x7/pretrained_<arith>.json (deterministic; compared by a one-sample t-test of the reference's seeds).
p: paired t-test over common seeds, same arithmetic, same val split. Writes results/x7/summary_<val>.md and
results/x7/summary.json for val in (full, valA, valB). Also: --dir results/x7v --methods B,I for VisDrone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = (42, 123, 456, 789, 1011)
ARITHS = ("none", "all", "cbest", "chunk128", "e8m13")
METHODS = ["ref_I", "ref_B", "I_lr5e-6", "I_lr2e-5", "I_ema99", "bitfit_lr1e-4", "bitfit_lr1e-3", "kd1", "kd5",
           "kdonly", "kd1_feat", "sr", "cbest", "chunk128", "e8m13"]


def ap(path, val):
    if not path.exists():
        return None
    m = json.load(open(path))["metrics"]
    return m["AP"] if val == "full" else m[val]["AP"]


def test(x, y):
    """x: {seed: v}; y: {seed: v} or a constant. Returns (mean diff pp, p, n)."""
    if isinstance(y, dict):
        s = [k for k in x if k in y]
        d = np.array([x[k] - y[k] for k in s]) * 100
    else:
        d = (np.array(list(x.values())) - y) * 100
    if len(d) < 2:
        return (float(d.mean()) if len(d) else None), None, len(d)
    if np.allclose(d, 0):
        return 0.0, 1.0, len(d)
    return float(d.mean()), float(stats.ttest_1samp(d, 0).pvalue), len(d)


def fmt_p(t, self_ref):
    if self_ref:
        return "опорный"
    d, p, n = t
    if d is None:
        return "нет общих сидов"
    return f"{d:+.2f} (p={p:.3f})" if p is not None else f"{d:+.2f} (n=1, p не опр.)"


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dir", default="results/x7")
    ap_.add_argument("--methods", default=",".join(METHODS))
    ap_.add_argument("--refs", default="ref_I,ref_B")
    a = ap_.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    methods = a.methods.split(",")
    ref_i, ref_b = a.refs.split(",")
    out = {}
    for val in ("full", "valA", "valB"):
        data = {m: {ar: {s: v for s in SEEDS if (v := ap(d / f"{m}_s{s}_{ar}.json", val)) is not None}
                    for ar in ARITHS} for m in methods}
        pre = {ar: ap(d / f"pretrained_{ar}.json", val) for ar in ARITHS}
        L = [f"# Блок D: метод × арифметика инференса, AP ({val}); auto: scripts/x7_summary.py", "",
             f"Ячейка: среднее ± sd по сидам [n]; Δ pp и парный p против {ref_i} и против {ref_b} в той же арифметике.", "",
             "| метод | " + " | ".join(ARITHS) + " |", "|---|" + "---|" * len(ARITHS)]
        res = {}
        for m in methods:
            cells, rc = [], {}
            for ar in ARITHS:
                x = data[m][ar]
                if not x:
                    cells.append("нет данных (в очереди)")
                    continue
                v = np.array(list(x.values()))
                sd = f"{v.std(ddof=1):.4f}" if len(v) > 1 else "—"
                ti, tb = test(x, data[ref_i][ar]), test(x, data[ref_b][ar])
                rc[ar] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "n": len(v),
                          "per_seed": x, "vs_ref_I": ti, "vs_ref_B": tb}
                cells.append(f"{v.mean():.4f} ± {sd} [{len(v)}]<br>I: {fmt_p(ti, m == ref_i)}<br>B: {fmt_p(tb, m == ref_b)}")
            res[m] = rc
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        if any(v is not None for v in pre.values()):
            cells = []
            for ar in ARITHS:
                if pre[ar] is None:
                    cells.append("нет данных (в очереди)")
                    continue
                ti, tb = test(data[ref_i][ar], pre[ar]), test(data[ref_b][ar], pre[ar])
                ti = (-ti[0], ti[1], ti[2]) if ti[0] is not None else ti
                tb = (-tb[0], tb[1], tb[2]) if tb[0] is not None else tb
                cells.append(f"{pre[ar]:.4f} (детерм., 1 прогон)<br>I: {fmt_p(ti, False)}<br>B: {fmt_p(tb, False)}")
            L.append("| исходная | " + " | ".join(cells) + " |")
            res["pretrained"] = pre
        L.append("")
        (d / f"summary_{val}.md").write_text("\n".join(L))
        out[val] = res
        if val == "full":
            print("\n".join(L))
    (d / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
