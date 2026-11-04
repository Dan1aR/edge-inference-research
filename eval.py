"""Evaluate YOLOS under a given arithmetic (FP32 / BF16 / BF16-accum per component).

Example:
  python eval.py --arith all --out results/e2/tiny_coco_all_fma.json
  python eval.py --arith all --chunk 32 --chunk_merge fp32 --out ...
  python eval.py --model small --dataset voc07_test --arith lin,patch --out ...
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from src import generic_arith as ga
from src.experiment import eval_loader, evaluate, load_dataset, load_gt, load_model, write_json, DATASETS, MODEL_ZOO
from src.triton_kernels.bf16acc_core import FORMATS, TC_FORMATS, set_accum_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny", choices=list(MODEL_ZOO))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dataset", default="coco_val", choices=list(DATASETS))
    p.add_argument("--arith", default="all", help='"fp32", "none" or component spec')
    p.add_argument("--round_product", action="store_true")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--chunk_merge", default="bf16", choices=["bf16", "acc", "fp32"])
    p.add_argument("--fmt", default="bf16", choices=list(FORMATS) + list(TC_FORMATS), help="accumulator format")
    p.add_argument("--rounding", default="rne", choices=["rne", "sr"])
    p.add_argument("--kahan", action="store_true")
    p.add_argument("--center", default="", help='centering, e.g. "av:b,patch:a"')
    p.add_argument("--exclude", default="", help="file with op ids (one per line) kept in standard arithmetic")
    p.add_argument("--impl", default="generic", choices=["generic", "legacy"])
    p.add_argument("--size", default=None, help='override input size, e.g. "800x800"')
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--no_dets", action="store_true", help="do not save detections npz")
    args = p.parse_args()

    cfg = set_accum_config(args.round_product, args.chunk, args.chunk_merge, args.fmt, args.rounding, args.kahan)
    if args.size:
        h, w = map(int, args.size.split("x"))
        MODEL_ZOO[args.model] = (MODEL_ZOO[args.model][0], {"height": h, "width": w}, MODEL_ZOO[args.model][2])
    kw = {"use_pretrained_backbone": False} if args.model.startswith("detr") else {}
    model, processor, ctrl = load_model(args.model, args.ckpt, impl=args.impl, model_kwargs=kw)
    if args.impl == "generic":
        ctrl.center = ga.parse_center(args.center)
        if args.exclude:
            ctrl.exclude = set(Path(args.exclude).read_text().split())
    elif args.center or args.exclude:
        raise ValueError("--center/--exclude need --impl generic")
    ds = load_dataset(args.dataset)
    if args.max_images:
        ds_eval = Subset(ds, range(args.max_images))
    else:
        ds_eval = ds
    gt = load_gt(DATASETS[args.dataset][1])
    out = Path(args.out)
    t0 = time.time()
    metrics = evaluate(model, eval_loader(ds_eval, processor, args.batch_size), processor, ctrl, args.arith, gt,
                       dets_path=None if args.no_dets else out.with_suffix(".dets.npz"))
    result = {
        "model": args.model, "ckpt": args.ckpt, "dataset": args.dataset, "arith": args.arith,
        "accum": cfg.describe(), "center": args.center, "exclude": args.exclude, "size": args.size,
        "metrics": metrics, "seconds": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(),
    }
    write_json(out, result)
    print(f"{out.name}: AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f} ({result['seconds']}s)")


if __name__ == "__main__":
    main()
