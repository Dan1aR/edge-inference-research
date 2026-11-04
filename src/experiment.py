"""Shared pieces of the revised experimental pipeline (eval.py / train.py).

Arithmetic specs used throughout:
  "fp32"           - no autocast, plain FP32 model;
  "none" / "bf16"  - autocast bf16, standard kernels (FP32 accumulation);
  component spec   - autocast bf16 + BF16 accumulation in the given components,
                     e.g. "all", "lin,patch", "all-av" (see src.target_arith).
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForObjectDetection, YolosForObjectDetection, YolosImageProcessor

from .config import FIXED_IMAGE_SIZE
from .data import TorchvisionCocoDetection, collate_fn_builder
from . import generic_arith
from .target_arith import TargetArithmetic, install_target_arithmetic, parse_components

REPO = Path(__file__).resolve().parents[1]
HF_LOCAL = REPO / "hf_models"  # weights of the models added in the extension (kept inside the project)
MODEL_IDS = {"tiny": "hustvl/yolos-tiny", "small": "hustvl/yolos-small"}
# name -> (HF id, input size, pad) ; YOLOS/RT-DETR: fixed square resize, DETR: aspect-preserving + padding
MODEL_ZOO = {
    "tiny": ("hustvl/yolos-tiny", FIXED_IMAGE_SIZE, False),
    "small": ("hustvl/yolos-small", FIXED_IMAGE_SIZE, False),
    "base": ("hustvl/yolos-base", FIXED_IMAGE_SIZE, False),
    "detr_r50": ("facebook/detr-resnet-50", {"shortest_edge": 800, "longest_edge": 1333}, True),
    "rtdetr_r18": ("PekingU/rtdetr_r18vd", {"height": 640, "width": 640}, False),
    "rtdetr_r50": ("PekingU/rtdetr_r50vd", {"height": 640, "width": 640}, False),
}
DS = REPO / "datasets"
DATASETS = {
    "coco_val": (REPO / "coco/val2017", REPO / "coco/annotations/instances_val2017.json"),
    "coco_train": (REPO / "coco/train2017", REPO / "coco/annotations/instances_train2017.json"),
    "voc07_test": (REPO / "voc/images", REPO / "voc/annotations/instances_voc07test_cocoids.json"),
    "visdrone_val": (DS / "visdrone/VisDrone2019-DET-val/images", DS / "visdrone/annotations/instances_val_cocoids.json"),
    "visdrone_val_native": (DS / "visdrone/VisDrone2019-DET-val/images", DS / "visdrone/annotations/instances_val_native.json"),
    "visdrone_train_native": (DS / "visdrone/VisDrone2019-DET-train/images", DS / "visdrone/annotations/instances_train_native.json"),
    "bdd_val": (DS / "bdd100k/images/val", DS / "bdd100k/annotations/instances_val_cocoids.json"),
    "coco_o": (DS / "coco_o", DS / "coco_o/annotations/instances_all.json"),
    "crowdhuman_val": (DS / "crowdhuman/images/val", DS / "crowdhuman/annotations/instances_val_cocoids.json"),
}
PROCESSOR_KWARGS = {"size": FIXED_IMAGE_SIZE, "do_convert_annotations": True, "do_pad": False}
CURRENT = {"model": "tiny"}  # set by load_model; selects the processor kwargs of eval/train loaders


def processor_kwargs(name: Optional[str] = None) -> Dict[str, Any]:
    _, size, pad = MODEL_ZOO[name or CURRENT["model"]]
    return {"size": size, "do_convert_annotations": True, "do_pad": pad}
METRIC_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]


def num_workers() -> int:
    return int(os.environ.get("NUM_WORKERS", max(1, (os.cpu_count() or 2) - 2)))


def load_model(name: str = "tiny", ckpt: Optional[str] = None, device: str = "cuda", impl: str = "generic",
               model_kwargs: Optional[Dict[str, Any]] = None):
    """impl="generic": :mod:`src.generic_arith` (all models); "legacy": :mod:`src.target_arith` (YOLOS only)."""
    hf_id = MODEL_ZOO[name][0]
    cache = {} if name in MODEL_IDS else {"cache_dir": str(HF_LOCAL)}
    CURRENT["model"] = name
    if name in MODEL_IDS:
        model = YolosForObjectDetection.from_pretrained(hf_id, attn_implementation="eager", **(model_kwargs or {}))
        processor = YolosImageProcessor.from_pretrained(hf_id)
    else:
        model = AutoModelForObjectDetection.from_pretrained(hf_id, attn_implementation="eager", **cache,
                                                            **(model_kwargs or {}))
        processor = AutoImageProcessor.from_pretrained(hf_id, **cache)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
    if impl == "legacy":
        ctrl = install_target_arithmetic(model)
    else:
        ctrl = generic_arith.install_arith(model)
    return model.to(device), processor, ctrl


def load_dataset(name: str) -> TorchvisionCocoDetection:
    images, ann = DATASETS[name]
    return TorchvisionCocoDetection(images, ann)


def holdout_split(ds: TorchvisionCocoDetection, n_holdout: int = 2000, seed: int = 0) -> Tuple[List[int], List[int]]:
    """Fixed split of train2017 indices into (train, holdout); independent of the run seed."""
    order = np.random.default_rng(seed).permutation(len(ds))
    hold = sorted(order[:n_holdout].tolist())
    train = sorted(order[n_holdout:].tolist())
    return train, hold


def val_split(ds: TorchvisionCocoDetection, seed: int = 0) -> Tuple[List[int], List[int]]:
    """Fixed split of val2017 into halves A (checkpoint selection) and B (reporting the selected model).

    A held-out part of train2017 cannot be used for selection: the pretrained YOLOS checkpoints
    were trained on all of train2017.
    """
    order = np.random.default_rng(seed).permutation(len(ds))
    half = len(ds) // 2
    return sorted(order[:half].tolist()), sorted(order[half:].tolist())


def subset_dets(dets: Dict[str, np.ndarray], image_ids: Sequence[int]) -> Dict[str, np.ndarray]:
    keep_ids = np.asarray(sorted(image_ids), dtype=np.int64)
    mask = np.isin(dets["image_id"], keep_ids)
    out = {k: v[mask] for k, v in dets.items() if k != "image_ids_evaluated"}
    out["image_ids_evaluated"] = keep_ids[np.isin(keep_ids, dets["image_ids_evaluated"])]
    return out


def eval_loader(ds, processor, batch_size: int = 64) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers(),
                      collate_fn=collate_fn_builder(processor, processor_kwargs=processor_kwargs()))


def train_loader(ds, processor, batch_size: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
                      num_workers=num_workers(), persistent_workers=True, prefetch_factor=2,
                      collate_fn=collate_fn_builder(processor, processor_kwargs=processor_kwargs()))


def arith_context(arith: str, ctrl):
    """Configure the controller for an arithmetic spec and return the autocast context."""
    parse = generic_arith.parse_components if isinstance(ctrl, generic_arith.Arith) else parse_components
    if arith == "fp32":
        ctrl.set(())
        return contextlib.nullcontext()
    ctrl.set(() if arith in ("none", "bf16") else parse(arith))
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _category_mapping(model, coco_gt) -> Dict[int, int]:
    name_to_cat = {cat["name"]: cid for cid, cat in coco_gt.cats.items()}
    mapping = {}
    for idx, name in model.config.id2label.items():
        if name in name_to_cat:
            mapping[int(idx)] = name_to_cat[name]
    return mapping


@torch.no_grad()
def predict(model, loader, processor, ctrl, arith: str, coco_gt) -> Dict[str, np.ndarray]:
    """Run the model and return all detections (threshold 0, as in COCO protocol)."""
    model.eval()
    mapping = _category_mapping(model, coco_gt)
    lut = np.full(max(int(k) for k in model.config.id2label) + 2, -1, dtype=np.int64)
    for k, v in mapping.items():
        lut[k] = v
    device = next(model.parameters()).device
    cols: Dict[str, list] = {"image_id": [], "category_id": [], "bbox": [], "score": []}
    image_ids: List[int] = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        extra = {"pixel_mask": batch["pixel_mask"].to(device)} if "pixel_mask" in batch else {}
        with arith_context(arith, ctrl):
            outputs = model(pixel_values=pixel_values, **extra)
        outputs.logits = outputs.logits.float()
        outputs.pred_boxes = outputs.pred_boxes.float()
        target_sizes = torch.tensor(batch["original_sizes"], device=device)
        results = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)
        for img_id, res in zip(batch["image_ids"], results):
            labels = res["labels"].cpu().numpy()
            cats = lut[labels]
            keep = cats >= 0
            boxes = res["boxes"].float().cpu().numpy()[keep]
            xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
            cols["image_id"].append(np.full(keep.sum(), img_id, dtype=np.int64))
            cols["category_id"].append(cats[keep])
            cols["bbox"].append(xywh.astype(np.float32))
            cols["score"].append(res["scores"].float().cpu().numpy()[keep].astype(np.float32))
        image_ids.extend(int(i) for i in batch["image_ids"])
    dets = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    dets["image_ids_evaluated"] = np.asarray(image_ids, dtype=np.int64)
    return dets


def coco_metrics(coco_gt, dets: Dict[str, np.ndarray]) -> Dict[str, float]:
    from faster_coco_eval import COCOeval_faster

    anns = [
        {"image_id": int(i), "category_id": int(c), "bbox": [float(x) for x in b], "score": float(s)}
        for i, c, b, s in zip(dets["image_id"], dets["category_id"], dets["bbox"], dets["score"])
    ]
    if not anns:
        return {k: 0.0 for k in METRIC_NAMES}
    coco_dt = coco_gt.loadRes(anns)
    ev = COCOeval_faster(coco_gt, coco_dt, iouType="bbox", print_function=lambda *a, **k: None)
    ev.params.imgIds = sorted(set(int(i) for i in dets["image_ids_evaluated"]))
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {k: float(v) for k, v in zip(METRIC_NAMES, ev.stats[:12])}


def load_gt(ann_file) :
    from faster_coco_eval import COCO

    return COCO(str(ann_file))


def evaluate(model, loader, processor, ctrl, arith: str, coco_gt, dets_path: Optional[Path] = None,
             subsets: Optional[Dict[str, Sequence[int]]] = None) -> Dict[str, Any]:
    """COCO metrics of all predictions; ``subsets`` adds metrics restricted to given image-id sets."""
    dets = predict(model, loader, processor, ctrl, arith, coco_gt)
    metrics = coco_metrics(coco_gt, dets)
    for name, ids in (subsets or {}).items():
        metrics[name] = coco_metrics(coco_gt, subset_dets(dets, ids))
    if dets_path is not None:
        dets_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dets_path, **dets)
    metrics["num_images"] = int(len(dets["image_ids_evaluated"]))
    return metrics


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
