"""A6: cheap output-level error metrics per arithmetic config, to test whether they predict the AP drop.

For N val images, the model output under each config is compared with the standard BF16 output
(FP32 accumulation): mean KL of the per-query class distributions, mean L1 of the boxes, relative
error of the last hidden state, and the mean relative error of all accumulator ops (diag mode).
The AP of the same configs comes from results/x1 and results/x4 (analysis in scripts/aggregate_x.py).

Usage: python scripts/x_error_model.py --model tiny --out results/x6/errmodel_tiny.json
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import generic_arith as ga  # noqa: E402
from src.experiment import MODEL_ZOO, arith_context, eval_loader, load_dataset, load_model  # noqa: E402
from src.triton_kernels.bf16acc_core import set_accum_config  # noqa: E402

CONFIGS = {  # name -> eval.py-style args (subset of X1/X4 names)
    "all": "--arith all", "all_e8m10": "--arith all --fmt e8m10", "all_e8m13": "--arith all --fmt e8m13",
    "all_e8m16": "--arith all --fmt e8m16", "all_fp16": "--arith all --fmt fp16", "all_kahan": "--arith all --kahan",
    "all_sr": "--arith all --rounding sr",
    **{f"all_chunk{c}_{m}": f"--arith all --chunk {c} --chunk_merge {m}" for c in (8, 16, 32, 64, 128, 256)
       for m in ("acc", "fp32")},
    "all_cav": '--arith all --center "av:b"', "all_cpatch": '--arith all --center "patch:a"',
    "all_cbest": '--arith all --center "av:b,patch:a,qk:a"', "all_clin": '--arith all --center "lin:a"',
    "lin+patch": "--arith lin,patch", "attn": "--arith attn", "lin": "--arith lin", "patch": "--arith patch",
    "av": "--arith av", "qk": "--arith qk", "conv": "--arith conv", "all-conv": "--arith all-conv",
    "all-av": "--arith all-av",
}


def parse(args: str):
    t = shlex.split(args)
    get = lambda k, d=None: t[t.index(k) + 1] if k in t else d  # noqa: E731
    acc = dict(chunk=int(get("--chunk", 0)), chunk_merge=get("--chunk_merge", "acc"), fmt=get("--fmt", "bf16"),
               rounding=get("--rounding", "rne"), kahan="--kahan" in t)
    return get("--arith"), acc, get("--center", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--size", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.size:
        h, w = map(int, a.size.split("x"))
        MODEL_ZOO[a.model] = (MODEL_ZOO[a.model][0], {"height": h, "width": w}, MODEL_ZOO[a.model][2])
    kw = {"use_pretrained_backbone": False} if a.model.startswith("detr") else {}
    model, processor, ctrl = load_model(a.model, model_kwargs=kw)
    model.eval()
    batches = list(eval_loader(Subset(load_dataset("coco_val"), range(a.n)), processor, 16))

    def run(spec, acc, center, diag=False):
        set_accum_config(**acc)
        ctrl.center = ga.parse_center(center)
        ctrl.diag = {} if diag else None
        outs = []
        with torch.no_grad():
            for b in batches:
                extra = {"pixel_mask": b["pixel_mask"].cuda()} if "pixel_mask" in b else {}
                with arith_context(spec, ctrl):
                    o = model(pixel_values=b["pixel_values"].cuda(), **extra)
                outs.append((o.logits.float(), o.pred_boxes.float(), o.last_hidden_state.float()))
        stats = ctrl.diag
        ctrl.diag, ctrl.center = None, {}
        set_accum_config()
        return outs, stats

    ref, _ = run("none", {}, "")
    res = {}
    for name, args in CONFIGS.items():
        spec, acc, center = parse(args)
        if not (ga.parse_components(spec) & ctrl.ops_components()):
            continue
        outs, stats = run(spec, acc, center, diag=True)
        kl = l1 = hs = 0.0
        for (lr, br, hr), (lo, bo, ho) in zip(ref, outs):
            kl += F.kl_div(F.log_softmax(lo, -1), F.softmax(lr, -1), reduction="none").sum(-1).mean().item()
            l1 += (bo - br).abs().mean().item()
            hs += ((ho - hr).norm() / hr.norm()).item()
        n = len(ref)
        op_err = [((s["sq"] / s["sig"]) ** 0.5) for s in stats.values()] if stats else []
        res[name] = {"kl": kl / n, "box_l1": l1 / n, "hidden_rel": hs / n,
                     "op_rel_err_mean": sum(op_err) / max(len(op_err), 1),
                     "op_rel_err_max": max(op_err) if op_err else 0.0}
        print(name, {k: round(v, 5) for k, v in res[name].items()}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "n": a.n, "size": a.size, "configs": res}, indent=1))


if __name__ == "__main__":
    main()
