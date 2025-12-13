from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import YolosImageProcessor


def _labels_to_targets(labels: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    targets: List[Dict[str, torch.Tensor]] = []
    for label in labels:
        targets.append({"boxes": label["boxes"], "labels": label["class_labels"]})
    return targets


def evaluate_map(
    model,
    dataloader,
    *,
    image_processor: YolosImageProcessor,
    device,
    threshold: float = 0.0,
    use_autocast: bool = True,
):
    metric = MeanAveragePrecision()
    model.eval()
    autocast_ctx = (
        torch.autocast(device_type=str(device), dtype=torch.bfloat16)
        if use_autocast
        else torch.cpu.amp.autocast(enabled=False)
    )
    with torch.no_grad():
        for batch in dataloader:
            # print(f"Eval {batch['target_sizes']=}")
            pixel_values = batch["pixel_values"].to(device)
            # target_sizes = torch.stack(batch["target_sizes"]).to(device)
            target_sizes = batch["target_sizes"]
            labels = [
                {"boxes": l["boxes"].to(device), "class_labels": l["class_labels"].to(device)}
                for l in batch["labels"]
            ]
            with autocast_ctx:
                outputs = model(pixel_values=pixel_values, labels=labels)
            processed = image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)
            preds = [
                {
                    "boxes": p["boxes"].cpu(),
                    "scores": p["scores"].cpu(),
                    "labels": p["labels"].cpu(),
                }
                for p in processed
            ]
            metric.update(
                preds,
                _labels_to_targets(
                    [
                        {
                            "boxes": label["boxes"].cpu(),
                            "class_labels": label["class_labels"].cpu(),
                        }
                        for label in labels
                    ]
                ),
            )
    computed = metric.compute()
    return {k: v.item() for k, v in computed.items()}
