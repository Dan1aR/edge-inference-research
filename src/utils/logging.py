from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import underdeep as U


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
    # wandb.login()
    # wandb.init(project=project, name=run_name, config=config, mode=os.environ.get("WANDB_MODE"))
    if not run_name:
        raise ValueError(f"--wandb_run_name can't be empty if using report_to='wandb'")

    try:
        # 1. Импортируйте пакет Underdeep
        client = U.Client(project=project)
        # 3. Придумайте код для нового эксперимента и впишите его в параметр code
        new_experiment = client.experiments.add(code=run_name)
    except U.common.utils.UnderdeepException:
        # skipping creation of already existing exp
        pass
    except Exception as exc:
        # failing for everything else
        raise exc

    client = U.Client(experiment=f"{project}/{run_name}")
    run = client.init_run(parameters=config)
    return run


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
    _bad_keys = [k for k, v in payload.items() if not (isinstance(v, float) or isinstance(v, int))]
    for k in _bad_keys:
        payload.pop(k)
    logger.log(payload)
    if wandb_run is not None:
        wandb_run.log(payload, step=step)
    # else:
    #     raise ValueError("Local log is disabled, you can uncomment `logger.log(payload)` ")
