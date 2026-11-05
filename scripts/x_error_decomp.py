"""D1: decomposition of the accumulator error of every op into a systematic part (per-output-channel mean
over tokens/images, absorbable by a bias) and the rest, for given checkpoints, on N val images.

Usage: python scripts/x_error_decomp.py OUT.json [--model tiny] [--n 256] [--ckpt name=path ...]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import arith_context, eval_loader, load_dataset, load_model  # noqa: E402


def run(model_name, ckpt, n, spec):
    model, processor, ctrl = load_model(model_name, ckpt)
    model.eval()
    loader = eval_loader(Subset(load_dataset("coco_val"), range(n)), processor, 32)
    ctrl.diag = {}
    with torch.no_grad():
        for batch in loader:
            with arith_context(spec, ctrl):
                model(pixel_values=batch["pixel_values"].cuda())
    out = {}
    fixed = {op: st for op, st in ctrl.diag.items() if st["sum"] is not None}  # ops with a fixed channel axis
    for op, st in ctrl.diag.items():
        sys_energy = float((st["sum"] / st["n"]).pow(2).sum()) * st["n"] if op in fixed else None
        out[op] = {"component": op.split("#")[1], "rel_err": (st["sq"] / st["sig"]) ** 0.5,
                   "systematic_fraction": None if sys_energy is None else
                   (sys_energy / st["sq"] if st["sq"] > 0 else 0.0)}
    tot_sq = sum(st["sq"] for st in fixed.values())
    tot_sys = sum(float((st["sum"] / st["n"]).pow(2).sum()) * st["n"] for st in fixed.values())
    return {"ops": out, "systematic_fraction_all_ops": tot_sys / tot_sq}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--model", default="tiny")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--arith", default="all")
    p.add_argument("--ckpt", nargs="*", default=[], help="name=path; the pretrained model is always included")
    a = p.parse_args()
    res = {"pretrained": run(a.model, None, a.n, a.arith)}
    for item in a.ckpt:
        name, path = item.split("=")
        res[name] = run(a.model, path, a.n, a.arith)
    for k, v in res.items():
        print(k, "systematic fraction of error energy:", round(v["systematic_fraction_all_ops"], 4))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
