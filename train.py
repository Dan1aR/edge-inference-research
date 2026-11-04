"""Fine-tune YOLOS under a (progressive) BF16-accumulation schedule with STE backward.

Protocol (identical for all methods):
  * train on COCO train2017; AdamW, cosine LR with 10% warmup, FP32 master weights, autocast bf16;
  * primary result: the checkpoint at the fixed final step, evaluated on the full COCO val2017;
  * checkpoint selection (secondary result): val2017 is split once into halves A/B (seed 0);
    every --eval_every steps AP on val-A is computed in the *target* arithmetic, the best
    checkpoint is kept and reported on val-B only. (A held-out part of train2017 cannot be used:
    the pretrained YOLOS checkpoints were trained on all of train2017.)

Examples:
  baseline (BF16, FP32 accum):  --schedule "" --target none --extra_evals all
  immediate full BF16-accum:    --schedule "all@0" --target all
  progressive (paper):          --schedule "patch@0,attn@60,mlp@120,qkvo@180,head@240" --target all
  mixed, FP32 accum in attn:    --schedule "all-attn@0" --target all-attn
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Subset
from transformers import get_scheduler

import torch.nn.functional as F

from src import generic_arith as ga
from src.experiment import (
    DATASETS, MODEL_ZOO, eval_loader, evaluate, load_dataset, load_gt, load_model, train_loader, val_split, write_json,
)
from src.generic_arith import ProgressiveSchedule
from src.generic_arith import parse_components
from src.triton_kernels.bf16acc_core import set_accum_config
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model", default="tiny", choices=list(MODEL_ZOO))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--schedule", default="", help='e.g. "all@0" or "patch@0,attn@60,..." ("" = BF16 baseline)')
    p.add_argument("--target", default="all", help="arithmetic used for selection and val eval")
    p.add_argument("--extra_evals", default="", help="';'-separated extra arithmetics for the final val eval")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--val_datasets", default="coco_val", help="extra datasets besides coco_val, e.g. voc07_test")
    p.add_argument("--max_val_images", type=int, default=None, help="debug only")
    # --- extension (block D) ---
    p.add_argument("--fmt", default="bf16", help="accumulator format (train and eval)")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--chunk_merge", default="acc")
    p.add_argument("--center", default="", help='centering in train and eval, e.g. "av:b,patch:a"')
    p.add_argument("--train_rounding", default="rne", choices=["rne", "sr"],
                   help="rounding of the accumulator in training forward passes (eval always RNE)")
    p.add_argument("--trainable", default="all", choices=["all", "bitfit"],
                   help="bitfit: only biases and LayerNorm parameters are trained")
    p.add_argument("--kd", type=float, default=0.0, help="weight of distillation from the BF16 teacher (0 = off)")
    p.add_argument("--kd_T", type=float, default=1.0)
    p.add_argument("--kd_only", action="store_true", help="drop the ground-truth loss (pure distillation)")
    p.add_argument("--kd_feat", type=float, default=0.0, help="weight of last-hidden-state MSE to the teacher")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights used for evaluation (0 = off)")
    # --- domain transfer (new head on another dataset) ---
    p.add_argument("--train_ds", default="coco_train")
    p.add_argument("--val_ds", default="coco_val", help="primary validation set (split into val-A/val-B)")
    p.add_argument("--new_head", action="store_true",
                   help="re-initialise the class head for the categories of --train_ds (contiguous ids)")
    p.add_argument("--size", default=None, help='input size override, e.g. "800x800"')
    return p.parse_args()


class RemapCategories(torch.utils.data.Dataset):
    """Maps dataset category ids to contiguous class indices of a new detection head."""

    def __init__(self, ds, cat_map):
        self.ds, self.cat_map, self.ids = ds, cat_map, ds.ids

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        ex["annotations"] = [dict(a, category_id=self.cat_map[a["category_id"]]) for a in ex["annotations"]
                             if not a.get("iscrowd", 0)]
        return ex


def kd_loss(student, teacher, T: float, w_feat: float):
    """Per-query distillation: KL of class distributions + L1/GIoU-free box L1 + optional feature MSE.

    Queries/detection tokens of YOLOS and DETR have a fixed index, and student and teacher start from
    the same weights, so query i of the student is matched to query i of the teacher.
    """
    s_log = F.log_softmax(student.logits.float() / T, -1)
    t_prob = F.softmax(teacher.logits.float() / T, -1)
    kl = F.kl_div(s_log, t_prob, reduction="none").sum(-1).mean() * T * T
    box = F.l1_loss(student.pred_boxes.float(), teacher.pred_boxes.float())
    loss = kl + 5.0 * box
    if w_feat > 0:
        loss = loss + w_feat * F.mse_loss(student.last_hidden_state.float(), teacher.last_hidden_state.float())
    return loss


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "args.json", vars(args))
    set_seed(args.seed)

    kw = {"use_pretrained_backbone": False} if args.model.startswith("detr") else {}
    if args.size:
        h, w = map(int, args.size.split("x"))
        MODEL_ZOO[args.model] = (MODEL_ZOO[args.model][0], {"height": h, "width": w}, MODEL_ZOO[args.model][2])
    cat_map = None
    if args.new_head:
        cats = sorted(load_gt(DATASETS[args.train_ds][1]).cats.values(), key=lambda c: c["id"])
        cat_map = {c["id"]: i for i, c in enumerate(cats)}
        kw.update(num_labels=len(cats), id2label={i: c["name"] for i, c in enumerate(cats)},
                  label2id={c["name"]: i for i, c in enumerate(cats)}, ignore_mismatched_sizes=True)
    model, processor, ctrl = load_model(args.model, model_kwargs=kw)
    ctrl.center = ga.parse_center(args.center)
    eval_acc = dict(chunk=args.chunk, chunk_merge=args.chunk_merge, fmt=args.fmt)
    set_accum_config(**eval_acc)
    teacher = None
    if args.kd > 0:
        teacher, _, tctrl = load_model(args.model, model_kwargs=kw)
        teacher.eval().requires_grad_(False)
    if args.trainable == "bitfit":
        for n, prm in model.named_parameters():
            prm.requires_grad_(n.endswith(".bias") or "layernorm" in n.lower() or "norm" in n.split(".")[-2].lower())
    ema = None
    if args.ema > 0:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.is_floating_point()}
    schedule = ProgressiveSchedule(args.schedule)
    target = args.target
    target_comps = set() if target in ("none", "bf16") else parse_components(target)
    if schedule.components_at(args.steps - 1) & ctrl.ops_components() != target_comps & ctrl.ops_components():
        raise ValueError(f"schedule ends with {sorted(schedule.components_at(args.steps - 1))}, "
                         f"target is {sorted(target_comps)}")
    device = next(model.parameters()).device

    train_ds = load_dataset(args.train_ds)
    if cat_map is not None:
        train_ds = RemapCategories(train_ds, cat_map)
    dl = train_loader(train_ds, processor, args.batch_size, args.seed)
    val_ds = load_dataset(args.val_ds)
    val_gt = load_gt(DATASETS[args.val_ds][1])
    idx_a, idx_b = val_split(val_ds)
    if args.max_val_images:
        idx_a, idx_b = idx_a[: args.max_val_images // 2], idx_b[: args.max_val_images // 2]
    ids_a = [val_ds.ids[i] for i in idx_a]
    ids_b = [val_ds.ids[i] for i in idx_b]
    sel_dl = eval_loader(Subset(val_ds, idx_a), processor, args.eval_batch_size)

    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_sched = get_scheduler("cosine", optimizer, num_warmup_steps=int(args.warmup_ratio * args.steps),
                             num_training_steps=args.steps)

    log_f = (run_dir / "train_log.jsonl").open("w")
    curve = []
    best = {"AP": -1.0, "step": None, "state": None}

    def swap_ema():
        """Exchange live weights and EMA weights in place (call twice to restore)."""
        if ema is None:
            return
        sd = model.state_dict()
        for k, v in ema.items():
            live = sd[k].detach().clone().float()
            sd[k].copy_(v)
            ema[k] = live

    def selection_eval(step: int) -> None:
        t = time.time()
        swap_ema()
        m = evaluate(model, sel_dl, processor, ctrl, target, val_gt)
        swap_ema()
        rec = {"step": step, "valA_AP": m["AP"], "valA_AP50": m["AP50"], "valA_AP75": m["AP75"],
               "seconds": round(time.time() - t, 1)}
        curve.append(rec)
        print(json.dumps(rec), flush=True)
        if m["AP"] > best["AP"]:
            swap_ema()
            best.update(AP=m["AP"], step=step, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            swap_ema()

    selection_eval(0)
    it = iter(dl)
    t_train = 0.0
    for step in range(args.steps):
        comps = schedule.components_at(step)
        ctrl.set(comps)
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            labels = [{k: v.to(device) for k, v in lab.items()} for lab in batch["labels"]]
            pv = batch["pixel_values"].to(device)
            extra = {"pixel_mask": batch["pixel_mask"].to(device)} if "pixel_mask" in batch else {}
            if args.train_rounding == "sr":
                set_accum_config(rounding="sr", **eval_acc)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(pixel_values=pv, labels=labels, output_hidden_states=args.kd_feat > 0, **extra)
                loss = 0.0 if args.kd_only else out.loss
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(pixel_values=pv, output_hidden_states=args.kd_feat > 0, **extra)
                    loss = loss + args.kd * kd_loss(out, t_out, args.kd_T, args.kd_feat)
                loss = loss / args.grad_accum
            set_accum_config(**eval_acc)
            loss.backward()
            loss_sum += loss.item()
        lr_used = lr_sched.get_last_lr()[0]
        optimizer.step()
        lr_sched.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            with torch.no_grad():
                sd = model.state_dict()
                for k, v in ema.items():
                    v.mul_(args.ema).add_(sd[k].float(), alpha=1 - args.ema)
        torch.cuda.synchronize()
        t_train += time.time() - t0
        log_f.write(json.dumps({"step": step + 1, "loss": loss_sum, "lr": lr_used,
                                "components": sorted(comps), "sec": round(time.time() - t0, 3)}) + "\n")
        log_f.flush()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            selection_eval(step + 1)
    log_f.close()

    swap_ema()  # evaluate (and save) the EMA weights if EMA is on
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(final_state, run_dir / "final.pt")

    summary = {
        "args": vars(args),
        "target_components": sorted(target_comps),
        "full_target_from_step": schedule.full_at() if args.schedule else None,
        "selection_curve_valA": curve,
        "best_step": best["step"],
        "train_seconds": round(t_train, 1),
        "val": {},
    }
    subsets = {"valA": ids_a, "valB": ids_b}
    extra = [a for a in args.extra_evals.split(";") if a]
    datasets = [args.val_ds] + [d for d in args.val_datasets.split(",") if d and d not in ("coco_val", args.val_ds)]
    for ds_name in datasets:
        gt = val_gt if ds_name == args.val_ds else load_gt(DATASETS[ds_name][1])
        ds = val_ds if ds_name == args.val_ds else load_dataset(ds_name)
        if args.max_val_images:
            ds = Subset(ds, range(args.max_val_images))
        loader = eval_loader(ds, processor, args.eval_batch_size)
        sub = subsets if ds_name == args.val_ds else None
        res = {}
        for tag, state in (("final", final_state), ("best", best["state"])):
            if tag == "best" and best["step"] == args.steps:
                res["best"] = res["final"]
                continue
            model.load_state_dict(state)
            res[tag] = evaluate(model, loader, processor, ctrl, target, gt, run_dir / f"dets_{ds_name}_{tag}.npz", sub)
            if tag == "final":
                for arith in extra:
                    res[f"final@{arith}"] = evaluate(model, loader, processor, ctrl, arith, gt,
                                                     run_dir / f"dets_{ds_name}_final@{arith}.npz", sub)
        summary["val"][ds_name] = res
        print(ds_name, json.dumps({k: round(v["AP"], 4) for k, v in res.items()}), flush=True)
    write_json(run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
