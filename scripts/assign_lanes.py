"""Assign pending queue jobs to GPU lanes from measured peaks (results/gpu_peak.jsonl), see scripts/queue_runner.py.

A pending job gets ``LANE=light`` if its class (rules below, each tied to measured gpu_peak tags) has a
measured peak <= LIMIT GiB for every measured arithmetic; trainings and unmeasured classes stay main.
Evals of checkpoints produced by pending trainings get ``AFTER=<train job>``.
Usage: python scripts/assign_lanes.py [--apply]   (default: dry run, prints the assignment)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "queue"
LIMIT = 6.0  # CoS budget: heavy <= 28 + VAP server <= 11.5 + light <= 6 GB
peaks = {}
for line in (ROOT / "results/gpu_peak.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["status"] == "ok" or r["status"].startswith("after_truncation"):
        peaks[r["tag"]] = r["peak_gib"]  # later re-measurements override
    else:
        peaks.setdefault(r["tag"], None)


def cls_peak(*tags):
    """Max peak over the tags; None if any is missing or failed."""
    v = [peaks.get(t) for t in tags]
    return None if any(x is None for x in v) else max(v)


AR = ("none", "all", "chunk128")
RULES = [  # (name regex, command regex or None, measured tags)
    (r"^x7g_|^x7sg_", None, ("grid_tiny", "grid_small")),
    (r"^x1_base_", r"--batch_size 32", tuple(f"base_{a}" for a in AR)),
    (r"^x4_rtdetr_r18_", None, tuple(f"rtdetr_r18_{a}" for a in AR)),
    (r"^x4_rtdetr_r50_", None, tuple(f"rtdetr_r50_{a}" for a in AR)),
    (r"^x4_detr_", None, tuple(f"detr_r50_{a}" for a in AR)),
    (r"^x4_small_", r"^(?!.*--size)", tuple(f"small64_{a}" for a in AR)),
    (r"^x4_tiny_", r"^(?!.*--size)", ("grid_tiny",)),
    (r"^x4_(tiny|small)_.*", r"--size 1024x1024", tuple(f"{m}1024_{a}" for m in ("tiny", "small") for a in AR)),
    (r"^x4_deit", None, ("deit_base_none", "deit_base_all", "deit_base_chunk32", "deit_base_fp8_tc")),
    (r"^x6_errmodel_(tiny|small)$", None, ("errm_small",)),
    (r"^x6_errmodel_base$", None, ("errm_base",)),
    (r"^x6_errmodel_rtdetr_r18$", None, ("errm_rtdetr_r18",)),
    (r"^x6_errmodel_detr_r50$", None, ("errm_detr_r50",)),
]
TOKEN = re.compile(r"^\s*(LANE|AFTER)=(\S+)\s+")


def strip(cmd):
    while m := TOKEN.match(cmd):
        cmd = cmd[m.end():]
    return cmd


def main():
    apply = "--apply" in sys.argv
    done = {p.name for p in (Q / "done").iterdir()} | {p.name for p in (Q / "failed").iterdir()}
    lines = (Q / "jobs.tsv").read_text().splitlines()
    pending = {l.split("\t", 1)[0] for l in lines if l.strip() and not l.startswith("#")} - done
    out, stat = [], Counter()
    for line in lines:
        if not line.strip() or line.startswith("#"):
            out.append(line)
            continue
        name, cmd = line.split("\t", 1)
        if name not in pending:
            out.append(line)
            continue
        cmd = strip(cmd)
        lane, peak = "main", None
        for nre, cre, tags in RULES:
            if re.search(nre, name) and (cre is None or re.search(cre, cmd)):
                peak = cls_peak(*tags)
                lane = "light" if peak is not None and peak <= LIMIT else "main"
                break
        deps = []
        m = re.search(r"--ckpt (\S+)/final\.pt", cmd)
        if m:  # eval of a checkpoint: depend on the pending train job that writes it
            rd = m.group(1)
            deps = [l.split("\t", 1)[0] for l in lines if l.strip() and f"--run_dir {rd} " in l + " "
                    and l.split("\t", 1)[0] in pending and l.split("\t", 1)[0] != name]
        pre = (f"LANE={lane} " if lane != "main" else "") + (f"AFTER={','.join(deps)} " if deps else "")
        stat[(re.sub(r"_s\d+$|_(none|all|e8m13|chunk\S*|lin,patch|conv|attn|fp32|fp8\S*|av|all-\S+|center\S*)$", "",
                     name)[:28], lane, peak)] += 1
        out.append(f"{name}\t{pre}{cmd}")
    for (k, lane, peak), n in sorted(stat.items(), key=lambda x: x[0][0]):
        print(f"{n:4d}  {k:30s} {lane:6s} peak={peak}")
    if apply:
        (Q / "jobs.tsv").write_text("\n".join(out) + "\n")
        print("applied")


if __name__ == "__main__":
    main()
