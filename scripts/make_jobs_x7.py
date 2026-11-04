"""Phase X7: complete grid for block D — every fine-tuning method x 5 seeds x 5 inference arithmetics.

Methods: ref_I, ref_B (runs/e8) and the 13 X2 variants (runs/x2); seeds 42, 123, 456, 789, 1011.
Arithmetics: none, all, cbest, chunk128, e8m13 (scripts/eval_grid.py), each on full val, val-A, val-B.
Also: the pretrained model in the same 5 arithmetics (deterministic, one run), and the VisDrone transfer
(X5, B and I) extended from 3 to 5 seeds and evaluated in the 5 arithmetics on VisDrone val.
Prints two blocks: lines for "--before x6_" and lines for "--after x5_vd_I_s456" (see __main__).
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

# VisDrone transfer: 2 more seeds of B and I, then all x5 checkpoints in the 5 arithmetics
x5 = {l.split("\t")[0]: l.split("\t", 1)[1] for l in Path("queue/jobs.tsv").read_text().splitlines()
      if l.startswith("x5_vd_")}
after_x5 = []
for s in (789, 1011):
    for v in ("B", "I"):
        cmd = re.sub(r"--seed \d+", f"--seed {s}", x5[f"x5_vd_{v}_s42"]).replace(f"vd_{v}_s42", f"vd_{v}_s{s}")
        after_x5.append(f"x5_vd_{v}_s{s}\t{cmd}")
for s in SEEDS:
    for v in ("B", "I"):
        after_x5.append(f"x7v_{v}_s{s}\t{G} --ckpt runs/x5/vd_{v}_s{s}/final.pt --prefix results/x7v/{v}_s{s} "
                        f"--dataset visdrone_val_native --size 800x800 --new_head_from visdrone_train_native "
                        f"--batch_size 16")

if __name__ == "__main__":
    block = sys.argv[1]
    print("\n".join(before_x6 if block == "d" else after_x5))
    print(f"{block}: {len(before_x6 if block == 'd' else after_x5)} jobs "
          f"(train {len(train)}, grid old {len(grid_old)}, grid new {len(grid_new)})", file=sys.stderr)
