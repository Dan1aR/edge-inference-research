"""Phase X7: complete grid for block D — every fine-tuning method x 5 seeds x 5 inference arithmetics.

Methods: ref_I, ref_B (runs/e8) and the 13 X2 variants (runs/x2); seeds 42, 123, 456, 789, 1011.
Arithmetics: none, all, cbest, chunk128, e8m13 (scripts/eval_grid.py), each on full val, val-A, val-B.
Also: the pretrained model in the same 5 arithmetics (deterministic, one run).
Prints job lines for "--before x6_" (see __main__).
"""
import re
import sys
from pathlib import Path

SEEDS = (42, 123, 456, 789, 1011)
src = Path(__file__).with_name("make_jobs_x2.py").read_text().split("lines = []")[0]
ns = {}
exec(src, ns)
PY, BASE, VARIANTS = ns["PY"], ns["BASE"], ns["VARIANTS"]
queued = {l.split("\t")[0] for l in Path("queue/jobs.tsv").read_text().splitlines() if l.strip()}
G = "NUM_WORKERS=6 .venv/bin/python scripts/eval_grid.py"


def ckpt(method, seed):
    if method in ("ref_I", "ref_B"):
        return f"runs/e8/lr1e-5_{method[-1]}_s{seed}/final.pt"
    return f"runs/x2/{method}_s{seed}/final.pt"


methods = ["ref_I", "ref_B"] + list(VARIANTS)
existing = [(m, s) for m in methods for s in SEEDS if Path(ckpt(m, s)).exists()]
train, grid_old, grid_new = [], [], []
for name in VARIANTS:
    for s in SEEDS:
        if Path(ckpt(name, s)).exists() or f"x2_{name}_s{s}" in queued or f"x3_{name}_s{s}" in queued:
            continue
        train.append(f"x7t_{name}_s{s}\t{PY} --run_dir runs/x2/{name}_s{s} --seed {s} {BASE} {VARIANTS[name]}")
for m in methods:
    for s in SEEDS:
        line = f"x7g_{m}_s{s}\t{G} --ckpt {ckpt(m, s)} --prefix results/x7/{m}_s{s}"
        (grid_old if (m, s) in existing else grid_new).append(line)
before_x6 = [f"x7g_pretrained\t{G} --prefix results/x7/pretrained"] + grid_old + train + grid_new

# YOLOS-small: ref_I / ref_B (recipe of E5: batch 32 x grad_accum 2, LR 1e-5, 300 steps), 5 seeds, same 5 arithmetics.
# Seeds 42/123/456 exist in runs/e5; 789 and 1011 are added. Grid outputs: results/x7/small_<method>_s<seed>_<arith>.
def small_block():
    e5 = {l.split("\t")[0]: l.split("\t", 1)[1] for l in Path("queue/jobs.tsv").read_text().splitlines()
          if l.startswith("e5_small_")}
    lines = []
    for s in (789, 1011):
        for v in ("B", "I"):
            if f"e5_small_{v}_s{s}" in e5 or f"x7s_{v}_s{s}" in queued:
                continue
            cmd = re.sub(r"--seed \d+", f"--seed {s}", e5[f"e5_small_{v}_s42"]).replace(f"{v}_s42", f"{v}_s{s}")
            lines.append(f"x7s_{v}_s{s}\t{cmd}")
    lines.append(f"x7sg_pretrained\t{G} --model small --batch_size 32 --prefix results/x7/small_pretrained")
    for s in SEEDS:
        for v in ("I", "B"):
            lines.append(f"x7sg_ref_{v}_s{s}\t{G} --model small --batch_size 32 --ckpt runs/e5/{v}_s{s}/final.pt "
                         f"--prefix results/x7/small_ref_{v}_s{s}")
    return lines


if __name__ == "__main__":  # d: before x6_; s: YOLOS-small refs, before x6_ (after block d)
    block = sys.argv[1]
    lines = {"d": before_x6, "s": small_block()}[block]
    print("\n".join(lines))
    print(f"{block}: {len(lines)} jobs", file=sys.stderr)
