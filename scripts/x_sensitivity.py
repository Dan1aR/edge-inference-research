"""A5: per-op sensitivity and budgeted selection of ops kept in FP32 accumulation.

1. inventory of ops (record mode): component, K, accumulator outputs, MACs;
2. AP on a val-A subset with everything in BF16-acc, and with one op at a time restored to FP32-acc;
3. greedy selection by (AP gain) / (share of accumulator outputs) under budgets; writes exclude files
   (op ids kept in FP32) to be evaluated on the full val / val-B by eval.py --exclude.

Selection uses val-A only. Usage: python scripts/x_sensitivity.py --model tiny --n 1000 --out results/x1/sens_tiny
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import (DATASETS, eval_loader, evaluate, load_dataset, load_gt, load_model,  # noqa: E402
                            processor_kwargs, val_split)

BUDGETS = (0.01, 0.03, 0.05, 0.10, 0.25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny")
    p.add_argument("--n", type=int, default=1000, help="images of val-A used for the sensitivities")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = {"use_pretrained_backbone": False} if a.model.startswith("detr") else {}
    model, processor, ctrl = load_model(a.model, model_kwargs=kw)
    model.eval()
    ds = load_dataset("coco_val")
    gt = load_gt(DATASETS["coco_val"][1])
    idx_a, _ = val_split(ds)
    loader = eval_loader(Subset(ds, idx_a[: a.n]), processor, a.batch_size)

    # 1. inventory on one batch of real images (shapes are fixed for square-resize models)
    ctrl.record, ctrl.ops = True, {}
    batch = next(iter(loader))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(pixel_values=batch["pixel_values"][:1].cuda())
    ctrl.record = False
    ops = {k: vars(v) for k, v in ctrl.ops.items()}
    tot_out = sum(v["outputs"] for v in ops.values())
    tot_mac = sum(v["macs"] for v in ops.values())
    for v in ops.values():
        v["out_share"], v["mac_share"] = v["outputs"] / tot_out, v["macs"] / tot_mac

    # 2. sensitivities
    res_file = out / "sensitivity.json"
    sens = json.loads(res_file.read_text()) if res_file.exists() else {}
    if "__all__" not in sens:
        sens["__all__"] = evaluate(model, loader, processor, ctrl, "all", gt)["AP"]
        sens["__none__"] = evaluate(model, loader, processor, ctrl, "none", gt)["AP"]
    for op in ops:
        if op in sens:
            continue
        ctrl.exclude = {op}
        sens[op] = evaluate(model, loader, processor, ctrl, "all", gt)["AP"]
        ctrl.exclude = set()
        res_file.write_text(json.dumps(sens, indent=1))
        print(op, round(sens[op] - sens["__all__"], 4), flush=True)
    res_file.write_text(json.dumps(sens, indent=1))

    # 3. greedy selection by gain per unit of accumulator-output share
    gains = {op: sens[op] - sens["__all__"] for op in ops}
    order = sorted(ops, key=lambda o: gains[o] / max(ops[o]["out_share"], 1e-9), reverse=True)
    sel = {}
    for budget in BUDGETS:
        chosen, used = [], 0.0
        for op in order:
            if gains[op] <= 0:
                continue
            if used + ops[op]["out_share"] <= budget:
                chosen.append(op)
                used += ops[op]["out_share"]
        (out / f"exclude_out{int(budget * 100):02d}.txt").write_text("\n".join(chosen) + "\n")
        sel[str(budget)] = {"ops": chosen, "out_share": used,
                            "mac_share": sum(ops[o]["mac_share"] for o in chosen),
                            "sum_of_single_gains": sum(gains[o] for o in chosen)}
    (out / "inventory.json").write_text(json.dumps({"ops": ops, "gains": gains, "selection": sel,
                                                    "processor": str(processor_kwargs())}, indent=1))
    for b, v in sel.items():
        print(f"budget {b}: {len(v['ops'])} ops, outputs {v['out_share']:.3f}, MACs {v['mac_share']:.3f}, "
              f"sum gains {v['sum_of_single_gains']:.4f}")


if __name__ == "__main__":
    main()
