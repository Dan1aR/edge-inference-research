"""AP on val-A / val-B halves of COCO val2017 from saved detections (CPU).

Usage: python scripts/valb_ap.py results/x1/tiny_coco_{none,all,sel01}.dets.npz ...
Needed for configs selected on val-A (A5 budgets): report them on val-B only.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import DATASETS, coco_metrics, load_dataset, load_gt, subset_dets, val_split  # noqa: E402

ds = load_dataset("coco_val")
with contextlib.redirect_stdout(io.StringIO()):
    gt = load_gt(DATASETS["coco_val"][1])
ia, ib = val_split(ds)
ids_a, ids_b = [ds.ids[i] for i in ia], [ds.ids[i] for i in ib]
res = {}
for f in sys.argv[1:]:
    dets = dict(np.load(f))
    with contextlib.redirect_stdout(io.StringIO()):
        res[Path(f).name.replace(".dets.npz", "")] = {
            "valA": coco_metrics(gt, subset_dets(dets, ids_a))["AP"], "valB": coco_metrics(gt, subset_dets(dets, ids_b))["AP"]}
    print(Path(f).name, {k: round(v, 4) for k, v in res[Path(f).name.replace('.dets.npz', '')].items()}, flush=True)
print(json.dumps(res))
