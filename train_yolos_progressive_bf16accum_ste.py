from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Set

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import (
    YolosForObjectDetection,
    YolosImageProcessor,
    get_scheduler,
)

from src.data.coco import DetectionDataConfig, build_dataloaders
from src.eval.map import evaluate_map
from src.patching import apply_attention_patch, apply_triton_linear_patches, ProgressiveSchedule
from src.utils.checkpoints import save_checkpoint
from src.utils.logging import JsonlLogger, log_metrics, maybe_init_wandb
from src.utils.seed import set_seed


DEFAULT_SCHEDULE = "attn@0.2,mlp@0.4,qkv@0.6,head@0.8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progressive bf16-accum/STE training")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="coco2017", choices=["coco2017", "cppe5"])
    parser.add_argument("--coco_dir", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--ramp_mode", type=str, default="none", choices=["none", "blocks"])
    parser.add_argument("--blocks_per_stage", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    return parser.parse_args()


def _count_blocks(model) -> int:
    indices = set()
    for name, _ in model.named_modules():
        if ".layer." in name:
            parts = name.split(".layer.")
            if len(parts) > 1:
                remainder = parts[1]
                if remainder and remainder[0].isdigit():
                    try:
                        idx = int(remainder.split(".")[0])
                        indices.add(idx)
                    except ValueError:
                        continue
    return max(indices) + 1 if indices else 0


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    train_cfg = DetectionDataConfig(
        dataset=args.dataset,
        coco_dir=args.coco_dir,
        split="train",
        max_samples=args.max_train_samples,
    )
    eval_cfg = DetectionDataConfig(
        dataset=args.dataset,
        coco_dir=args.coco_dir,
        split="validation",
        max_samples=args.max_eval_samples,
    )
    train_dl, eval_dl = build_dataloaders(
        processor=processor,
        train_config=train_cfg,
        eval_config=eval_cfg,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = len(train_dl) // args.gradient_accumulation_steps
    max_steps = args.max_steps if args.max_steps > 0 else args.num_train_epochs * num_update_steps_per_epoch
    warmup_steps = args.warmup_steps
    if warmup_steps == 0 and args.warmup_ratio > 0:
        warmup_steps = int(max_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    train_dl, eval_dl, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dl, eval_dl, model, optimizer, lr_scheduler
    )

    schedule = ProgressiveSchedule.from_string(args.schedule)
    num_blocks_total = _count_blocks(model)

    train_logger = JsonlLogger(output_dir / "logs" / "train.jsonl")
    eval_logger = JsonlLogger(output_dir / "logs" / "eval.jsonl")
    wandb_run = maybe_init_wandb(
        args.report_to,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
    )

    (output_dir / "hparams.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    enabled_groups: Set[str] = set()
    max_block_enabled = None

    global_step = 0
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            current_enabled = set(schedule.enabled_groups(global_step, max_steps))
            newly_enabled = current_enabled - enabled_groups
            if newly_enabled:
                if "attn" in newly_enabled:
                    apply_attention_patch(accelerator.unwrap_model(model))
                linear_groups = newly_enabled & {"mlp", "qkv", "head"}
                if linear_groups:
                    block_limit = None
                    if args.ramp_mode == "blocks" and num_blocks_total:
                        block_limit = schedule.enabled_blocks(
                            global_step, max_steps, blocks_per_stage=args.blocks_per_stage, num_blocks_total=num_blocks_total
                        )
                        block_limit = block_limit - 1 if block_limit > 0 else None
                    patched = apply_triton_linear_patches(
                        accelerator.unwrap_model(model),
                        enabled_groups=current_enabled & {"mlp", "qkv", "head"},
                        max_block=block_limit,
                    )
                    if accelerator.is_main_process and patched:
                        print(f"[patch] enabled {len(patched)} linears (groups={current_enabled})")
                enabled_groups |= newly_enabled
                if args.ramp_mode == "blocks" and num_blocks_total:
                    max_block_enabled = schedule.enabled_blocks(
                        global_step, max_steps, blocks_per_stage=args.blocks_per_stage, num_blocks_total=num_blocks_total
                    )

            with accelerator.accumulate(model):
                pixel_values = batch["pixel_values"]
                labels = batch["labels"]
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16, enabled=True):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                log_metrics(
                    logger=train_logger,
                    metrics={"loss": loss.item(), "enabled_groups": list(sorted(enabled_groups))},
                    step=global_step,
                    epoch=epoch,
                    lr=lr_scheduler.get_last_lr()[0],
                    wandb_run=wandb_run,
                )

            global_step += 1
            completed_steps += 1

            if args.max_steps and completed_steps >= args.max_steps:
                break

            if eval_dl is not None and global_step % args.eval_steps == 0 and accelerator.is_main_process:
                metrics = evaluate_map(
                    accelerator.unwrap_model(model),
                    eval_dl,
                    image_processor=processor,
                    device=accelerator.device,
                    use_autocast=True,
                )
                metrics.update({"enabled_groups": list(sorted(enabled_groups)), "max_block": max_block_enabled})
                log_metrics(
                    logger=eval_logger,
                    metrics=metrics,
                    step=global_step,
                    epoch=epoch,
                    lr=lr_scheduler.get_last_lr()[0],
                    wandb_run=wandb_run,
                )
                save_checkpoint(
                    output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    step=global_step,
                    tag=f"step-{global_step}",
                    accelerator=accelerator,
                )
        if args.max_steps and completed_steps >= args.max_steps:
            break

    if accelerator.is_main_process:
        summary = {"final_step": global_step, "enabled_groups": list(sorted(enabled_groups))}
        (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
