"""Completeness matrix of the extension: what is done, queued, failed or missing. Writes results/completeness.md.

Block D (fine-tuning): method x seed x inference arithmetic, from results/x7/<method>_s<seed>_<arith>.json
(eval_grid: full val + val-A + val-B in one file). Inference-only blocks are deterministic (no seeds) and are
listed per group of queue jobs.
"""
import re
from collections import defaultdict
from pathlib import Path

SEEDS = (42, 123, 456, 789, 1011)
ARITHS = ("none", "all", "cbest", "chunk128", "e8m13")
METHODS = ["ref_I", "ref_B", "I_lr5e-6", "I_lr2e-5", "I_ema99", "bitfit_lr1e-4", "bitfit_lr1e-3", "kd1", "kd5",
           "kdonly", "kd1_feat", "sr", "cbest", "chunk128", "e8m13"]
jobs = [l.split("\t")[0] for l in Path("queue/jobs.tsv").read_text().splitlines() if l.strip() and not l.startswith("#")]
done = {p.name for p in Path("queue/done").iterdir()}
failed = {p.name for p in Path("queue/failed").iterdir()}


def jstat(name):
    if name in done:
        return "done"
    if name in failed:
        return "FAILED"
    return "queued" if name in jobs else "missing"


SYM = {"done": "✓", "queued": "q", "FAILED": "F", "missing": "·"}


def ckpt_job(m, s):
    if m in ("ref_I", "ref_B"):
        return f"e8_lr1e-5_{m[-1]}_s{s}"
    return next((j for j in (f"x2_{m}_s{s}", f"x3_{m}_s{s}", f"x7t_{m}_s{s}") if j in jobs), f"x7t_{m}_s{s}")


L = ["# Матрица полноты (auto: scripts/completeness.py)", "",
     "✓ готово · q в очереди · F упало · «·» нет и не запланировано. Каждая ячейка арифметики = 3 val (full, val-A,",
     "val-B) из одного прогона `scripts/eval_grid.py`. Символы в ячейке — сиды 42, 123, 456, 789, 1011.", "",
     "## D: дообучение YOLOS-tiny (COCO), метод × арифметика инференса", "",
     "| метод | обучение | " + " | ".join(ARITHS) + " |", "|---|---|" + "---|" * len(ARITHS)]
tot = defaultdict(int)
for m in METHODS:
    tr = "".join(SYM[jstat(ckpt_job(m, s))] for s in SEEDS)
    cells = []
    for a in ARITHS:
        c = ""
        for s in SEEDS:
            st = "done" if Path(f"results/x7/{m}_s{s}_{a}.json").exists() else jstat(f"x7g_{m}_s{s}")
            st = "queued" if st == "done" and not Path(f"results/x7/{m}_s{s}_{a}.json").exists() else st
            tot[st] += 1
            c += SYM[st]
        cells.append(c)
    L.append(f"| {m} | {tr} | " + " | ".join(cells) + " |")
pre = ["✓" if Path(f"results/x7/pretrained_{a}.json").exists() else SYM[jstat("x7g_pretrained")] for a in ARITHS]
L.append("| исходная (без дообучения, детерм.) | — | " + " | ".join(pre) + " |")
L += ["", f"Ячеек метод×сид×арифметика: {sum(tot.values())}; " + ", ".join(f"{k}: {v}" for k, v in tot.items()), ""]

L += ["## D-small: YOLOS-small (COCO), ref_I и ref_B (рецепт E5), метод × арифметика", "",
      "| метод | обучение | " + " | ".join(ARITHS) + " |", "|---|---|" + "---|" * len(ARITHS)]
for v in ("I", "B"):
    tr = "".join(SYM[jstat(f"e5_small_{v}_s{s}" if f"e5_small_{v}_s{s}" in jobs else f"x7s_{v}_s{s}")] for s in SEEDS)
    cells = ["".join("✓" if Path(f"results/x7/small_ref_{v}_s{s}_{a}.json").exists() else SYM[jstat(f"x7sg_ref_{v}_s{s}")]
                     for s in SEEDS) for a in ARITHS]
    L.append(f"| ref_{v} | {tr} | " + " | ".join(cells) + " |")
pre = ["✓" if Path(f"results/x7/small_pretrained_{a}.json").exists() else SYM[jstat("x7sg_pretrained")] for a in ARITHS]
L += ["| исходная (детерм.) | — | " + " | ".join(pre) + " |", ""]

L += ["## Инференс (детерминированный; сидов нет — неопределённость через парный бутстреп по изображениям)", "",
      "| группа | заданий | ✓ | q | F |", "|---|---:|---:|---:|---:|"]
groups = defaultdict(lambda: defaultdict(int))
for j in jobs:
    if j.startswith(("x2", "x3_", "x7", "x5_", "e", "x0")):
        continue
    mm = re.match(r"^(x1c|x1d|x1s|x6|x1|x4)_(.*)$", j)
    if not mm:
        continue
    pref, rest = mm.groups()
    if pref == "x4":
        dsm = re.match(r"^(tiny|small|base|rtdetr_r18|rtdetr_r50|detr_r50|deit-tiny|deit-small|deit-base)_"
                       r"(coco_val|coco_r\d+|voc07_test|visdrone_val|bdd_val|coco_o|crowdhuman_val|inv2)_", rest)
        key = f"X4 {dsm.group(1)} / {dsm.group(2)}" if dsm else f"X4 {rest}"
    elif pref == "x1":
        key = f"X1 {rest.split('_')[0]} / coco (пространство аккумуляторов)"
    else:
        model = rest.split("errmodel_")[-1] if pref == "x6" else next(
            (t for t in rest.split("_") if t in ("tiny", "small", "base")), rest)
        key = {"x1c": "X1c компоненты в FP32 (контроль A5)", "x1d": "D1 декомпозиция ошибки",
               "x1s": "A5 чувствительность + бюджеты", "x6": "A6 модель ошибки"}[pref] + f" ({model})"
    groups[key][jstat(j)] += 1
for k in sorted(groups):
    g = groups[k]
    L.append(f"| {k} | {sum(g.values())} | {g['done']} | {g['queued']} | {g['FAILED']} |")
L.append("")
Path("results/completeness.md").write_text("\n".join(L))
print("\n".join(L))
