"""Evaluate one checkpoint in several inference arithmetics (one model load), full val + val-A/val-B halves.

Arithmetics (names used in results and summaries):
  none      standard BF16 inference, FP32 accumulation
  all       BF16 accumulation in every matmul (sequential, RNE)
  cbest     all + centering av:b, patch:a, qk:a
  chunk128  all + chunks of 128 merged in FP32
  e8m13     all with an e8m13 accumulator
Usage: python scripts/eval_grid.py --ckpt runs/x2/kd1_s42/final.pt --prefix results/x7/kd1_s42
       [--ariths none,all,cbest,chunk128,e8m13] [--model tiny] [--dataset coco_val] [--size 800x800]
       [--new_head_from visdrone_train_native]
Writes <prefix>_<arith>.json (+ .dets.npz); existing outputs are skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import generic_arith as ga  # noqa: E402
from src.experiment import (DATASETS, MODEL_ZOO, eval_loader, evaluate, load_dataset, load_gt, load_model,  # noqa: E402
                            val_split, write_json)
from src.triton_kernels.bf16acc_core import set_accum_config  # noqa: E402

ARITHS = {  # name -> (component spec, accumulator config, centering)
    "none": ("none", {}, ""),
    "all": ("all", {}, ""),
    "cbest": ("all", {}, "av:b,patch:a,qk:a"),
    "chunk128": ("all", {"chunk": 128, "chunk_merge": "fp32"}, ""),
    "e8m13": ("all", {"fmt": "e8m13"}, ""),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--prefix", required=True)
    p.add_argument("--ariths", default=",".join(ARITHS))
    p.add_argument("--model", default="tiny")
    p.add_argument("--dataset", default="coco_val")
    p.add_argument("--size", default=None)
    p.add_argument("--new_head_from", default=None, help="train dataset whose categories define the head")
    p.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()
    todo = [x for x in a.ariths.split(",") if not Path(f"{a.prefix}_{x}.json").exists()]
    if not todo:
        print("all outputs exist")
        return
    if a.size:
        h, w = map(int, a.size.split("x"))
        MODEL_ZOO[a.model] = (MODEL_ZOO[a.model][0], {"height": h, "width": w}, MODEL_ZOO[a.model][2])
    kw = {"use_pretrained_backbone": False} if a.model.startswith("detr") else {}
    if a.new_head_from:
        cats = sorted(load_gt(DATASETS[a.new_head_from][1]).cats.values(), key=lambda c: c["id"])
        kw.update(num_labels=len(cats), id2label={i: c["name"] for i, c in enumerate(cats)},
                  label2id={c["name"]: i for i, c in enumerate(cats)}, ignore_mismatched_sizes=True)
    model, processor, ctrl = load_model(a.model, a.ckpt, model_kwargs=kw)
    model.eval()
    ds = load_dataset(a.dataset)
    gt = load_gt(DATASETS[a.dataset][1])
    ia, ib = val_split(ds)
    subsets = {"valA": [ds.ids[i] for i in ia], "valB": [ds.ids[i] for i in ib]}
    loader = eval_loader(ds, processor, a.batch_size)
    Path(a.prefix).parent.mkdir(parents=True, exist_ok=True)
    for name in todo:
        spec, acc, center = ARITHS[name]
        set_accum_config(**acc)
        ctrl.center = ga.parse_center(center)
        t0 = time.time()
        m = evaluate(model, loader, processor, ctrl, spec, gt, Path(f"{a.prefix}_{name}.dets.npz"), subsets)
        write_json(Path(f"{a.prefix}_{name}.json"), {
            "model": a.model, "ckpt": a.ckpt, "dataset": a.dataset, "arith_name": name, "arith": spec, "accum": acc,
            "center": center, "size": a.size, "metrics": m, "seconds": round(time.time() - t0, 1),
            "gpu": torch.cuda.get_device_name()})
        print(name, json.dumps({k: round(m[k]["AP"] if isinstance(m[k], dict) else m[k], 4)
                                for k in ("AP", "valA", "valB")}), flush=True)
        ctrl.center = {}
        set_accum_config()


if __name__ == "__main__":
    main()
