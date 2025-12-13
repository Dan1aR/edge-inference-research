from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    output_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    step: int,
    tag: str,
    accelerator=None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / tag
    ckpt_dir.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model) if accelerator is not None else model
    torch.save(model_to_save.state_dict(), ckpt_dir / "pytorch_model.bin")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    meta = {"step": step, "tag": tag}
    (ckpt_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return ckpt_dir
