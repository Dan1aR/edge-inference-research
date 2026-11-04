"""Screening table for X2: final val-A AP (selection criterion) and full-val AP per variant and seed."""
import glob
import json
import os
from collections import defaultdict

rows = defaultdict(dict)
for f in glob.glob("runs/x2/*/summary.json") + glob.glob("runs/e8/lr1e-5_[BI]_s*/summary.json"):
    run = os.path.basename(os.path.dirname(f))
    name, seed = run.rsplit("_s", 1)
    name = name.replace("lr1e-5_", "ref_")
    if seed not in ("42", "123"):
        continue
    s = json.load(open(f))
    rows[name][seed] = (s["selection_curve_valA"][-1]["valA_AP"], s["val"]["coco_val"]["final"]["AP"])
ref = rows.get("ref_I", {})
print(f"{'variant':16s} {'valA s42':>9s} {'valA s123':>9s} {'mean dA vs ref_I':>16s} {'full s42':>9s} {'full s123':>9s}")
for name in sorted(rows):
    r = rows[name]
    d = [r[s][0] - ref[s][0] for s in r if s in ref]
    fmt = lambda s, i: f"{r[s][i]:.4f}" if s in r else "   -   "  # noqa: E731
    print(f"{name:16s} {fmt('42', 0):>9s} {fmt('123', 0):>9s} {100 * sum(d) / max(len(d), 1):+16.2f} {fmt('42', 1):>9s} {fmt('123', 1):>9s}")
