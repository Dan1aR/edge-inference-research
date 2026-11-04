"""X2+X3 summary over all available seeds: final-step AP on full val and val-B, paired differences.

Pairs: every variant vs ref_I (runs/e8/lr1e-5_I_s*, same seed); arithmetic variants (cbest, chunk128, e8m13)
additionally vs B weights deployed in the same arithmetic (results/x2b/B_s<seed>_<arith>.json, full val).
Writes results/x3/summary.json and prints a table. Paired t-test across seeds (scipy).
"""
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = ("42", "123", "456", "789", "1011")
ARITH = ("cbest", "chunk128", "e8m13")
runs = defaultdict(dict)
for f in glob.glob("runs/x2/*/summary.json") + glob.glob("runs/e8/lr1e-5_[BI]_s*/summary.json"):
    name, seed = os.path.basename(os.path.dirname(f)).rsplit("_s", 1)
    v = json.load(open(f))["val"]["coco_val"]["final"]
    runs[name.replace("lr1e-5_", "ref_")][seed] = {"full": v["AP"], "valB": v.get("valB", {}).get("AP", v.get("valB"))}
deployed = defaultdict(dict)
for f in glob.glob("results/x2b/B_s*_*.json"):
    seed, arith = Path(f).stem[len("B_s"):].split("_", 1)
    deployed[arith][seed] = {"full": json.load(open(f))["metrics"]["AP"]}


def paired(a, b, key="full"):
    s = [x for x in SEEDS if x in a and x in b and a[x][key] is not None and b[x][key] is not None]
    if not s:
        return None
    d = np.array([a[x][key] - b[x][key] for x in s]) * 100
    p = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 else None
    return {"n": len(s), "mean_pp": float(d.mean()), "sd_pp": float(d.std(ddof=1)) if len(d) > 1 else None,
            "p": p, "per_seed_pp": d.round(3).tolist()}


out = {}
print(f"{'variant':15s} {'n':>2s} {'AP full':>8s} {'vs ref_I (pp, p)':>20s} {'vs B deployed (pp, p)':>24s}")
for name in sorted(runs):
    r = runs[name]
    ap = np.mean([r[s]["full"] for s in r])
    row = {"seeds": r, "mean_full": float(ap), "vs_ref_I": paired(r, runs["ref_I"]),
           "vs_ref_I_valB": paired(r, runs["ref_I"], "valB")}
    if name in ARITH:
        row["vs_B_deployed"] = paired(r, deployed[name])
        row["B_deployed_mean"] = float(np.mean([v["full"] for v in deployed[name].values()])) if deployed[name] else None
    out[name] = row
    f = lambda x: "-" if not x else f"{x['mean_pp']:+.2f}" + (f" (p={x['p']:.3f})" if x["p"] is not None else "")  # noqa
    print(f"{name:15s} {len(r):2d} {ap:8.4f} {f(row['vs_ref_I']):>20s} {f(row.get('vs_B_deployed')):>24s}")
Path("results/x3").mkdir(parents=True, exist_ok=True)
Path("results/x3/summary.json").write_text(json.dumps(out, indent=1))
