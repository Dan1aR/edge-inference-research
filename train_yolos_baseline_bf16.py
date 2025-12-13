from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    YolosForObjectDetection,
    YolosImageProcessor,
    get_scheduler,
)

from src.data.coco import DetectionDataConfig, build_dataloaders
from src.eval.map import evaluate_map
from src.utils.checkpoints import save_checkpoint
from src.utils.logging import JsonlLogger, log_metrics, maybe_init_wandb
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline YOLOS bf16 training")
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
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def maybe_get_grad_scaler(precision: str) -> Optional[torch.cuda.amp.GradScaler]:
    if precision == "fp16":
        return torch.cuda.amp.GradScaler()
    return None


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

    train_logger = JsonlLogger(output_dir / "logs" / "train.jsonl")
    eval_logger = JsonlLogger(output_dir / "logs" / "eval.jsonl")
    wandb_run = maybe_init_wandb(
        args.report_to,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
    )

    scaler = maybe_get_grad_scaler(args.precision)

    hparams = vars(args)
    (output_dir / "hparams.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")

    global_step = 0
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                pixel_values = batch["pixel_values"]
                labels = batch["labels"]
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=(torch.bfloat16 if args.precision == "bf16" else torch.float16), enabled=args.precision in {"bf16", "fp16"}):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    accelerator.backward(loss)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                log_metrics(
                    logger=train_logger,
                    metrics={"loss": loss.item()},
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
                    use_autocast=args.precision in {"bf16", "fp16"},
                )
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
        summary = {"final_step": global_step}
        (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
