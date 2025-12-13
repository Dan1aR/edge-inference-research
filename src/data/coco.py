from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import datasets
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CocoDetection
from transformers import YolosImageProcessor

from .transforms import build_transform


@dataclass
class DetectionDataConfig:
    dataset: str
    coco_dir: Optional[str] = None
    split: str = "train"
    max_samples: Optional[int] = None
    transform: Optional[str] = None


def _ensure_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image)


def _format_annotations(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "annotations" in example:
        return example["annotations"]
    if "objects" in example:
        anns = []
        for obj in example["objects"]:
            anns.append({"bbox": obj.get("bbox", obj.get("bboxes", [0, 0, 1, 1])), "category_id": int(obj.get("category_id", obj.get("id", 0)))})
        return anns
    return []


class TorchvisionCocoDetection(Dataset):
    """Thin wrapper to expose torchvision CocoDetection in HF-like dict format."""

    def __init__(self, images_dir: Path, annotations_file: Path):
        if not images_dir.exists():
            raise FileNotFoundError(f"COCO images dir not found: {images_dir}")
        if not annotations_file.exists():
            raise FileNotFoundError(f"COCO annotations file not found: {annotations_file}")
        self.ds = CocoDetection(root=str(images_dir), annFile=str(annotations_file))
        self.ids = self.ds.ids

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, targets = self.ds[idx]
        anns: List[Dict[str, Any]] = []
        for ann in targets:
            anns.append({"bbox": ann.get("bbox", [0, 0, 1, 1]), "category_id": int(ann.get("category_id", 0))})

        return {
            "image": image,
            "annotations": anns,
            "image_id": self.ids[idx],
        }


def load_dataset(config: DetectionDataConfig):
    if config.dataset == "coco2017":
        if config.coco_dir is None:
            raise ValueError("coco_dir is required for coco2017 mode")
        split = "val" if config.split.startswith("val") else "train"
        coco_dir = Path(config.coco_dir)
        images_dir = coco_dir / f"{split}2017"
        annotations_file = coco_dir / "annotations" / f"instances_{split}2017.json"
        ds: Dataset = TorchvisionCocoDetection(images_dir=images_dir, annotations_file=annotations_file)
        if config.max_samples is not None:
            ds = Subset(ds, range(min(config.max_samples, len(ds))))
        return ds
    elif config.dataset == "cppe5":
        ds = datasets.load_dataset("cppe-5")
        split = "validation" if config.split.startswith("val") else config.split
        subset = ds[split]
        if config.max_samples is not None:
            subset = subset.select(range(config.max_samples))
        return subset
    else:
        raise ValueError(f"unknown dataset {config.dataset}")


def collate_fn_builder(image_processor: YolosImageProcessor, *, transform: Optional[str] = None):
    augment = build_transform(transform)

    def _collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images: List[Image.Image] = []
        annotations: List[Dict[str, Any]] = []
        sizes: List[Tuple[int, int]] = []
        for example in batch:
            img = _ensure_pil(example["image"])
            images.append(img)
            anns = _format_annotations(example)
            annotations.append({"image_id": example.get("image_id", None), "annotations": anns})
            sizes.append((img.height, img.width))

        images = augment(images)
        processed = image_processor(images=images, annotations=annotations, return_tensors="pt")
        processed["target_sizes"] = torch.tensor(sizes, dtype=torch.long)
        processed["original_sizes"] = sizes
        return processed

    return _collate


def build_dataloaders(
    *,
    processor: YolosImageProcessor,
    train_config: DetectionDataConfig,
    eval_config: Optional[DetectionDataConfig],
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_ds = load_dataset(train_config)
    train_dl = DataLoader(
        train_ds,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_builder(processor, transform=train_config.transform),
    )

    eval_dl = None
    if eval_config is not None:
        eval_ds = load_dataset(eval_config)
        eval_dl = DataLoader(
            eval_ds,
            batch_size=per_device_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_builder(processor, transform=eval_config.transform),
        )
    return train_dl, eval_dl
