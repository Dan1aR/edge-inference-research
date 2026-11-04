"""Paired image-bootstrap CIs for inference-ablation AP differences (CPU only).

Usage: python scripts/bootstrap_e2.py <results_dir> <ann_file> <out.json> [n_boot]
Compares every *.dets.npz in results_dir against reference configs ("none" and "fp32"),
plus a few specific pairs, on identical image resamples.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.bootstrap import PerImageEval, paired_bootstrap, summarize_delta  # noqa: E402

PAIRS = [("lin+patch_fma", "all_fma"), ("all-av_fma", "all_fma"), ("all_rp", "all_fma")]


def main():
    res_dir, ann, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    n_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann)
    names = sorted(p.name[: -len(".dets.npz")] for p in res_dir.glob("*.dets.npz"))
    evals = {}
    for n in names:
        evals[n] = PerImageEval(gt, dict(np.load(res_dir / f"{n}.dets.npz")))
        print("evaluated", n, flush=True)
    order = list(evals)
    point = {n: float(evals[n].ap(np.ones((1, len(evals[n].img_ids))))[0]) for n in order}
    samples = paired_bootstrap([evals[n] for n in order], n_boot=n_boot)
    col = {n: i for i, n in enumerate(order)}
    result = {"n_boot": n_boot, "ap": point, "single_se": {n: float(samples[:, col[n]].std(ddof=1)) for n in order},
              "deltas": {}}
    refs = [r for r in ("none", "fp32") if r in col]
    pairs = [(n, r) for r in refs for n in order if n != r]
    pairs += [(a, b) for a, b in PAIRS if a in col and b in col]
    pairs += [(n, "all_fma") for n in order if n.startswith("all_chunk") and "all_fma" in col]
    for a, b in pairs:
        d = samples[:, col[a]] - samples[:, col[b]]
        result["deltas"][f"{a} - {b}"] = summarize_delta(d, point[a] - point[b])
    out.write_text(json.dumps(result, indent=2))
    print("written", out)


if __name__ == "__main__":
    main()
