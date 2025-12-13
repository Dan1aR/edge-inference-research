from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class JsonlLogger:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record = {**record, "timestamp": datetime.utcnow().isoformat()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def maybe_init_wandb(report_to: str, *, project: Optional[str], run_name: Optional[str], config: Dict[str, Any]) -> Optional[Any]:
    if report_to != "wandb":
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - explicit failure when requested
        raise ImportError(
            "wandb is required when report_to='wandb'. Install wandb and ensure you are logged in."
        ) from exc
    wandb.login()
    wandb.init(project=project, name=run_name, config=config, mode=os.environ.get("WANDB_MODE"))
    return wandb


def log_metrics(
    *,
    logger: JsonlLogger,
    metrics: Dict[str, Any],
    step: int,
    epoch: int,
    lr: float,
    wandb_run: Optional[Any] = None,
) -> None:
    payload = {"step": step, "epoch": epoch, "lr": lr, **metrics}
    logger.log(payload)
    if wandb_run is not None:
        wandb_run.log(payload, step=step)
