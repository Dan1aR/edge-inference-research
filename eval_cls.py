"""Top-1/top-5 of DeiT/ViT classifiers on ImageNetV2 (matched-frequency) under a given arithmetic.

Example: python eval_cls.py --model facebook/deit-small-patch16-224 --arith all --out results/x4/deit_s_all.json
Saves per-image correctness (for a paired bootstrap) next to the json.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src import generic_arith as ga
from src.experiment import HF_LOCAL, arith_context, num_workers, write_json
from src.triton_kernels.bf16acc_core import FORMATS, TC_FORMATS, set_accum_config

ROOT = Path(__file__).resolve().parent / "datasets/imagenetv2"


class ImageNetV2(Dataset):
    def __init__(self, processor):
        with open(ROOT / "matched-frequency.csv") as f:
            self.items = [(r["path"], int(r["label"])) for r in csv.DictReader(f)]
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(ROOT / path).convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0], label


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/deit-small-patch16-224")
    p.add_argument("--arith", default="all")
    p.add_argument("--fmt", default="bf16", choices=list(FORMATS) + list(TC_FORMATS))
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--chunk_merge", default="acc")
    p.add_argument("--center", default="")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    set_accum_config(chunk=a.chunk, chunk_merge=a.chunk_merge, fmt=a.fmt)
    model = AutoModelForImageClassification.from_pretrained(a.model, attn_implementation="eager", cache_dir=str(HF_LOCAL))
    processor = AutoImageProcessor.from_pretrained(a.model, cache_dir=str(HF_LOCAL))
    ctrl = ga.install_arith(model)
    ctrl.center = ga.parse_center(a.center)
    model = model.cuda().eval()
    loader = DataLoader(ImageNetV2(processor), batch_size=a.batch_size, num_workers=num_workers())
    t0 = time.time()
    top1, top5 = [], []
    with torch.no_grad():
        for x, y in loader:
            with arith_context(a.arith, ctrl):
                logits = model(pixel_values=x.cuda()).logits.float()
            pred = logits.topk(5, -1).indices.cpu()
            top1.append((pred[:, 0] == y).numpy())
            top5.append((pred == y[:, None]).any(-1).numpy())
    top1, top5 = np.concatenate(top1), np.concatenate(top5)
    out = Path(a.out)
    np.savez_compressed(out.with_suffix(".correct.npz"), top1=top1, top5=top5)
    res = {"model": a.model, "arith": a.arith, "fmt": a.fmt, "chunk": a.chunk, "chunk_merge": a.chunk_merge,
           "center": a.center, "top1": float(top1.mean()), "top5": float(top5.mean()), "n": int(len(top1)),
           "seconds": round(time.time() - t0, 1)}
    write_json(out, res)
    print(f"{out.name}: top1={res['top1']:.4f} top5={res['top5']:.4f} ({res['seconds']}s)")


if __name__ == "__main__":
    main()
