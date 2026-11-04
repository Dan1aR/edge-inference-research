"""Paired image-bootstrap CIs for the extension's inference results (CPU only).

Usage: python scripts/bootstrap_x.py <results_dir> <prefix> <ann_file> <out.json> [--refs none,all] [--n_boot 500]
Every <prefix>*.dets.npz in results_dir is compared with each reference config on identical resamples.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.bootstrap import PerImageEval, paired_bootstrap, summarize_delta  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("res_dir")
    p.add_argument("prefix")
    p.add_argument("ann")
    p.add_argument("out")
    p.add_argument("--refs", default="none,all")
    p.add_argument("--n_boot", type=int, default=500)
    a = p.parse_args()
    res_dir = Path(a.res_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.ann)
    files = sorted(res_dir.glob(f"{a.prefix}*.dets.npz"))
    names = [f.name[len(a.prefix): -len(".dets.npz")] for f in files]
    evals = {}
    for n, f in zip(names, files):
        evals[n] = PerImageEval(gt, dict(np.load(f)))
    order = list(evals)
    point = {n: float(evals[n].ap(np.ones((1, len(evals[n].img_ids))))[0]) for n in order}
    samples = paired_bootstrap([evals[n] for n in order], n_boot=a.n_boot)
    col = {n: i for i, n in enumerate(order)}
    result = {"prefix": a.prefix, "n_boot": a.n_boot, "ap": point,
              "single_se": {n: float(samples[:, col[n]].std(ddof=1)) for n in order}, "deltas": {}}
    for r in [r for r in a.refs.split(",") if r in col]:
        for n in order:
            if n != r:
                d = samples[:, col[n]] - samples[:, col[r]]
                result["deltas"][f"{n} - {r}"] = summarize_delta(d, point[n] - point[r])
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("written", a.out, len(order), "configs")


if __name__ == "__main__":
    main()
