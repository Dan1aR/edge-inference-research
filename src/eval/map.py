from __future__ import annotations

"""Evaluation utilities for YOLOS models.

This module combines mAP computation used during training with COCO evaluation
routines originally implemented in :mod:`src.eval_coco`.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import YolosForObjectDetection, YolosImageProcessor

from ..config import PrecisionMode
from ..data import TorchvisionCocoDetection


def evaluate_map(
    model,
    dataloader,
    *,
    image_processor: YolosImageProcessor,
    device,
    threshold: float = 0.0,
    use_autocast: bool = True,
):
    model.eval()

    def _unwrap_dataset(ds):
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        return ds

    coco_ds = _unwrap_dataset(dataloader.dataset)
    if not hasattr(coco_ds, "coco"):
        raise ValueError("evaluate_map expects a COCO-style dataset with a 'coco' attribute")

    coco_gt = coco_ds.coco
    yolos_to_coco = build_category_mapping(model, coco_gt)

    all_predictions: list[dict[str, Any]] = []
    processed_image_ids: list[int] = []

    autocast_ctx = (
        torch.autocast(device_type=str(device), dtype=torch.bfloat16)
        if use_autocast
        else torch.cpu.amp.autocast(enabled=False)
    )

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            image_ids = batch["image_ids"]
            original_sizes = batch["original_sizes"]
            target_sizes = torch.tensor(original_sizes, device=device)

            with autocast_ctx:
                outputs = model(pixel_values=pixel_values)

            results = image_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes
            )

            batch_predictions = convert_predictions_to_coco_format(
                results=results,
                image_ids=image_ids,
                original_sizes=original_sizes,
                yolos_to_coco_mapping=yolos_to_coco,
            )

            all_predictions.extend(batch_predictions)
            processed_image_ids.extend(image_ids)

    coco_metrics = run_coco_eval(coco_gt, all_predictions, image_ids=processed_image_ids)

    # Provide both COCO-style keys and the legacy mAP/mAR aliases used by the training logs
    combined: dict[str, Any] = {
        "AP": coco_metrics["AP"],
        "AP50": coco_metrics["AP50"],
        "AP75": coco_metrics["AP75"],
        "AP_small": coco_metrics["AP_small"],
        "AP_medium": coco_metrics["AP_medium"],
        "AP_large": coco_metrics["AP_large"],
        "AR_1": coco_metrics["AR_1"],
        "AR_10": coco_metrics["AR_10"],
        "AR_100": coco_metrics["AR_100"],
        "AR_small": coco_metrics["AR_small"],
        "AR_medium": coco_metrics["AR_medium"],
        "AR_large": coco_metrics["AR_large"],
        "num_predictions": len(all_predictions),
        "num_images": len(processed_image_ids),
        # Legacy aliases
        "map": coco_metrics["AP"],
        "map_50": coco_metrics["AP50"],
        "map_75": coco_metrics["AP75"],
        "map_small": coco_metrics["AP_small"],
        "map_medium": coco_metrics["AP_medium"],
        "map_large": coco_metrics["AP_large"],
        "mar_1": coco_metrics["AR_1"],
        "mar_10": coco_metrics["AR_10"],
        "mar_100": coco_metrics["AR_100"],
        "mar_small": coco_metrics["AR_small"],
        "mar_medium": coco_metrics["AR_medium"],
        "mar_large": coco_metrics["AR_large"],
    }

    return combined


def build_category_mapping(
    model: YolosForObjectDetection,
    coco_gt: COCO,
) -> dict[int, int]:
    """
    Build a mapping from YOLOS label indices to COCO category IDs.

    YOLOS uses its own label indices that need to be mapped back to
    COCO's category_id for proper evaluation.

    Args:
        model: YOLOS model with id2label config
        coco_gt: COCO ground truth object

    Returns:
        Dictionary mapping YOLOS label index -> COCO category_id
    """
    id2label = model.config.id2label
    name_to_cat_id = {cat["name"]: cid for cid, cat in coco_gt.cats.items()}

    yolos_idx_to_coco_id: dict[int, int] = {}
    for idx, name in id2label.items():
        idx = int(idx)
        if name in name_to_cat_id:
            yolos_idx_to_coco_id[idx] = name_to_cat_id[name]

    return yolos_idx_to_coco_id


def convert_predictions_to_coco_format(
    results: list[dict],
    image_ids: list[int],
    original_sizes: list[tuple[int, int]],
    yolos_to_coco_mapping: dict[int, int],
) -> list[dict[str, Any]]:
    """
    Convert YOLOS post-processed results to COCO prediction format.
    """
    predictions = []

    for result, image_id, _ in zip(results, image_ids, original_sizes):
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]  # xyxy format

        for score, label, box in zip(scores, labels, boxes):
            label_idx = label.item()
            if label_idx not in yolos_to_coco_mapping:
                continue

            x1, y1, x2, y2 = box.tolist()
            width = x2 - x1
            height = y2 - y1

            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": yolos_to_coco_mapping[label_idx],
                    "bbox": [x1, y1, width, height],
                    "score": score.item(),
                }
            )

    return predictions


def run_coco_eval(
    coco_gt: COCO,
    predictions: list[dict[str, Any]],
    image_ids: list[int] | None = None,
) -> dict[str, float]:
    """
    Run COCO evaluation and return metrics.
    """
    if len(predictions) == 0:
        print("Warning: No predictions to evaluate!")
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    if image_ids is not None:
        coco_eval.params.imgIds = image_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR_1": coco_eval.stats[6],
        "AR_10": coco_eval.stats[7],
        "AR_100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }

    return metrics


def evaluate_coco(
    model: YolosForObjectDetection,
    processor: YolosImageProcessor,
    dataloader: DataLoader,
    dataset: TorchvisionCocoDetection,
    device: torch.device,
    precision_mode: PrecisionMode,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate model on COCO validation set.
    """
    model.eval()
    coco_gt = dataset.coco

    yolos_to_coco = build_category_mapping(model, coco_gt)

    all_predictions = []
    processed_image_ids = []

    print(f"\nRunning inference in {precision_mode.value} mode...")

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            image_ids = batch["image_ids"]
            original_sizes = batch["original_sizes"]

            if precision_mode in (PrecisionMode.BF16_DEFAULT, PrecisionMode.BF16_ACCUM):
                pixel_values = pixel_values.to(torch.bfloat16)

            outputs = model(pixel_values=pixel_values)

            target_sizes = torch.tensor(original_sizes, device=device)

            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )

            batch_predictions = convert_predictions_to_coco_format(
                results=results,
                image_ids=image_ids,
                original_sizes=original_sizes,
                yolos_to_coco_mapping=yolos_to_coco,
            )
            all_predictions.extend(batch_predictions)
            processed_image_ids.extend(image_ids)

    print(f"Total predictions: {len(all_predictions)}")
    print(f"Processed images: {len(processed_image_ids)}")

    metrics = run_coco_eval(coco_gt, all_predictions, image_ids=processed_image_ids)

    result = {
        "precision_mode": precision_mode.value,
        "num_images": len(dataset),
        "num_predictions": len(all_predictions),
        "metrics": metrics,
    }

    return result


def save_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_metrics(metrics: dict[str, float]) -> None:
    """Print evaluation metrics in a formatted way."""
    print("\n" + "=" * 50)
    print("COCO Detection Metrics")
    print("=" * 50)
    print(f"  AP @ IoU=0.50:0.95  = {metrics['AP']:.4f}")
    print(f"  AP @ IoU=0.50       = {metrics['AP50']:.4f}")
    print(f"  AP @ IoU=0.75       = {metrics['AP75']:.4f}")
    print(f"  AP (small)          = {metrics['AP_small']:.4f}")
    print(f"  AP (medium)         = {metrics['AP_medium']:.4f}")
    print(f"  AP (large)          = {metrics['AP_large']:.4f}")
    print("-" * 50)
    print(f"  AR @ maxDets=1      = {metrics['AR_1']:.4f}")
    print(f"  AR @ maxDets=10     = {metrics['AR_10']:.4f}")
    print(f"  AR @ maxDets=100    = {metrics['AR_100']:.4f}")
    print(f"  AR (small)          = {metrics['AR_small']:.4f}")
    print(f"  AR (medium)         = {metrics['AR_medium']:.4f}")
    print(f"  AR (large)          = {metrics['AR_large']:.4f}")
    print("=" * 50)
