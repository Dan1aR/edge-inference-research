from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import datasets
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CocoDetection
from transformers import YolosImageProcessor

import albumentations as A

from .config import FIXED_IMAGE_SIZE

ImageLike = Image.Image | torch.Tensor | Any


@dataclass
class DetectionDataConfig:
    dataset: str
    coco_dir: Optional[str] = None
    split: str = "train"
    max_samples: Optional[int] = None
    transform: Optional[str] = None


def build_transform(pipeline: str | None = None) -> Callable[[Sequence[ImageLike]], List[ImageLike]]:
    """Build an augmentation pipeline.

    Args:
        pipeline: Optional pipeline identifier. When ``None`` or ``"none"`` a
            no-op transform is returned.

    Returns:
        Callable that accepts a sequence of images and returns a list of transformed
        images.
    """

    if pipeline is None or pipeline.lower() == "none":
        def _noop(images: Sequence[ImageLike]) -> List[ImageLike]:
            return [img for img in images]

        return _noop

    if pipeline.lower() == "basic":
        aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            ]
        )

        def _apply(images: Sequence[ImageLike]) -> List[ImageLike]:
            outputs: List[ImageLike] = []
            for img in images:
                img_np = np.array(img) if isinstance(img, Image.Image) else np.array(img)
                transformed = aug(image=img_np)["image"]
                outputs.append(Image.fromarray(transformed))
            return outputs

        return _apply

    def _default(images: Sequence[ImageLike]) -> List[ImageLike]:
        return [img for img in images]

    return _default


def _ensure_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    return Image.fromarray(image)


def _format_annotations(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "annotations" in example:
        return example["annotations"]
    if "objects" in example:
        anns = []
        for obj in example["objects"]:
            bbox = obj.get("bbox", obj.get("bboxes", [0, 0, 1, 1]))
            bbox = [float(x) for x in bbox]
            w = max(0.0, bbox[2])
            h = max(0.0, bbox[3])
            anns.append(
                {
                    "bbox": bbox,
                    "category_id": int(obj.get("category_id", obj.get("id", 0))),
                    "area": float(obj.get("area", w * h)),
                    "iscrowd": int(obj.get("iscrowd", 0)),
                }
            )
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
            bbox = ann.get("bbox", [0, 0, 1, 1])  # COCO is [x, y, w, h]
            bbox = [float(x) for x in bbox]
            w = max(0.0, bbox[2])
            h = max(0.0, bbox[3])
            anns.append(
                {
                    "bbox": bbox,
                    "category_id": int(ann.get("category_id", 0)),
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    # optional but harmless:
                    **({"id": int(ann["id"])} if "id" in ann else {}),
                }
            )

        return {
            "image": image,
            "annotations": anns,
            "image_id": self.ids[idx],
        }

    @property
    def coco(self):  # pragma: no cover - passthrough for evaluation
        return self.ds.coco


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
    else:
        raise ValueError(f"unknown dataset {config.dataset}")


def collate_fn_builder(
    image_processor: YolosImageProcessor,
    *,
    transform: Optional[str] = None,
    processor_kwargs: Optional[Dict[str, Any]] = None,
):
    augment = build_transform(transform)
    processor_kwargs = processor_kwargs or {}

    def _collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images: List[Image.Image] = []
        annotations: List[Dict[str, Any]] = []
        sizes: List[Tuple[int, int]] = []
        image_ids: List[int] = []
        for idx, example in enumerate(batch):
            img = _ensure_pil(example["image"])
            images.append(img)
            anns = _format_annotations(example)
            annotations.append({"image_id": example.get("image_id", idx), "annotations": anns})
            sizes.append((img.height, img.width))
            image_ids.append(int(example.get("image_id", idx)))

        images = augment(images)
        processed = image_processor(images=images, annotations=annotations, return_tensors="pt", **processor_kwargs)
        processed["target_sizes"] = torch.tensor(sizes, dtype=torch.long)
        processed["original_sizes"] = sizes
        processed["image_ids"] = image_ids
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
        collate_fn=collate_fn_builder(processor, transform=train_config.transform, processor_kwargs={"size": FIXED_IMAGE_SIZE}),
    )

    eval_dl = None
    if eval_config is not None:
        eval_ds = load_dataset(eval_config)
        eval_dl = DataLoader(
            eval_ds,
            batch_size=per_device_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_builder(processor, transform=eval_config.transform, processor_kwargs={"size": FIXED_IMAGE_SIZE}),
        )
    return train_dl, eval_dl


def create_dataloader(
    images_dir: str | Path,
    annotations_file: str | Path,
    processor: YolosImageProcessor,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: int | None = None,
) -> tuple[DataLoader, Dataset]:
    dataset: Dataset = TorchvisionCocoDetection(Path(images_dir), Path(annotations_file))
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_builder(processor, processor_kwargs={"size": FIXED_IMAGE_SIZE}),
        pin_memory=False,
    )

    return dataloader, dataset


__all__ = [
    "DetectionDataConfig",
    "TorchvisionCocoDetection",
    "build_transform",
    "load_dataset",
    "collate_fn_builder",
    "build_dataloaders",
    "create_dataloader",
]
