"""
COCO Evaluation Logic

This module provides functions for evaluating object detection models
on the COCO dataset using pycocotools.
"""
import json
import torch
from torch.utils.data import DataLoader
from transformers import YolosForObjectDetection, YolosImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from pathlib import Path
from typing import Any

from .config import PrecisionMode
from .data import CocoYolosDataset


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
    # Get YOLOS id2label mapping
    id2label = model.config.id2label
    
    # Build name -> COCO category_id mapping
    name_to_cat_id = {cat["name"]: cid for cid, cat in coco_gt.cats.items()}
    
    # Build YOLOS idx -> COCO category_id mapping
    yolos_idx_to_coco_id = {}
    for idx, name in id2label.items():
        idx = int(idx)  # Ensure integer key
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
    
    Args:
        results: List of post_process_object_detection outputs
                 Each contains 'scores', 'labels', 'boxes' (xyxy format)
        image_ids: List of COCO image IDs
        original_sizes: List of (height, width) tuples
        yolos_to_coco_mapping: YOLOS label -> COCO category_id mapping
    
    Returns:
        List of COCO-format prediction dictionaries
    """
    predictions = []
    
    for result, image_id, orig_size in zip(results, image_ids, original_sizes):
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]  # xyxy format
        
        for score, label, box in zip(scores, labels, boxes):
            label_idx = label.item()
            
            # Skip if label not in mapping (shouldn't happen for valid detections)
            if label_idx not in yolos_to_coco_mapping:
                continue
            
            # Convert xyxy to xywh
            x1, y1, x2, y2 = box.tolist()
            width = x2 - x1
            height = y2 - y1
            
            predictions.append({
                "image_id": image_id,
                "category_id": yolos_to_coco_mapping[label_idx],
                "bbox": [x1, y1, width, height],
                "score": score.item(),
            })
    
    return predictions


def run_coco_eval(
    coco_gt: COCO,
    predictions: list[dict[str, Any]],
    image_ids: list[int] | None = None,
) -> dict[str, float]:
    """
    Run COCO evaluation and return metrics.
    
    Args:
        coco_gt: COCO ground truth object
        predictions: List of COCO-format prediction dictionaries
        image_ids: Optional list of image IDs to evaluate on.
                   If None, evaluates on all images with predictions.
    
    Returns:
        Dictionary of evaluation metrics
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
    
    # Load predictions into COCO format
    coco_dt = coco_gt.loadRes(predictions)
    
    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    
    # Limit evaluation to specific images if provided
    if image_ids is not None:
        coco_eval.params.imgIds = image_ids
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        "AP": coco_eval.stats[0],      # AP @ IoU=0.50:0.95
        "AP50": coco_eval.stats[1],    # AP @ IoU=0.50
        "AP75": coco_eval.stats[2],    # AP @ IoU=0.75
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR_1": coco_eval.stats[6],    # AR maxDets=1
        "AR_10": coco_eval.stats[7],   # AR maxDets=10
        "AR_100": coco_eval.stats[8],  # AR maxDets=100
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }
    
    return metrics


def evaluate_coco(
    model: YolosForObjectDetection,
    processor: YolosImageProcessor,
    dataloader: DataLoader,
    dataset: CocoYolosDataset,
    device: torch.device,
    precision_mode: PrecisionMode,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate model on COCO validation set.
    
    Args:
        model: YOLOS model to evaluate
        processor: YOLOS image processor for post-processing
        dataloader: DataLoader for COCO validation
        dataset: Dataset instance (for access to COCO object)
        device: Device to run inference on
        precision_mode: Current precision mode (for metadata)
        threshold: Score threshold for filtering predictions
    
    Returns:
        Dictionary containing metrics and metadata
    """
    model.eval()
    coco_gt = dataset.coco
    
    # Build category mapping
    yolos_to_coco = build_category_mapping(model, coco_gt)
    
    all_predictions = []
    processed_image_ids = []
    
    print(f"\nRunning inference in {precision_mode.value} mode...")
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            image_ids = batch["image_ids"]
            original_sizes = batch["original_sizes"]
            
            # Handle precision mode for input
            if precision_mode in (PrecisionMode.BF16_DEFAULT, PrecisionMode.BF16_ACCUM):
                pixel_values = pixel_values.to(torch.bfloat16)
            
            # Forward pass
            # Note: YOLOS (ViT-based) doesn't support pixel_mask, so we use
            # fixed-size images to ensure consistent results across batch sizes
            outputs = model(pixel_values=pixel_values)
            
            # Convert original_sizes to tensor for post-processing
            # Format: (height, width) for each image
            target_sizes = torch.tensor(original_sizes, device=device)
            
            # Post-process to get boxes in original image coordinates
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )
            
            # Convert to COCO format
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
    
    # Run COCO evaluation only on processed images
    metrics = run_coco_eval(coco_gt, all_predictions, image_ids=processed_image_ids)
    
    # Add metadata
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
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save JSON file
    """
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

