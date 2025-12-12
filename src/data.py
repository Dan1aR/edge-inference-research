"""
COCO Dataset and Preprocessing for YOLOS

This module provides dataset classes and utilities for loading COCO 2017
validation data and preprocessing it for the YOLOS model.
"""
import errno
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from transformers import YolosImageProcessor
from PIL import Image  # pillow

from .config import FIXED_IMAGE_SIZE


def _is_permission_denied(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        if getattr(exc, "errno", None) == errno.EACCES:
            return True
        # Some environments don’t populate errno reliably:
        if "permission denied" in str(exc).lower():
            return True
    return False


class CocoYolosDataset(Dataset):
    """
    COCO dataset wrapper for YOLOS model.

    Wraps torchvision's CocoDetection and preprocesses images using
    HuggingFace's YolosImageProcessor.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        processor: YolosImageProcessor,
        max_samples: int | None = None,
        permission_retry_sleep_s: int = 300,   # 5 minutes
        permission_retry_count: int = 3,       # retry 3 times (after first failure)
    ):
        self.coco_dataset = CocoDetection(
            root=str(images_dir),
            annFile=str(annotations_file),
        )
        self.processor = processor
        self.max_samples = max_samples

        self.permission_retry_sleep_s = int(permission_retry_sleep_s)
        self.permission_retry_count = int(permission_retry_count)

        # Store reference to underlying COCO object for evaluation
        self.coco = self.coco_dataset.coco

    def __len__(self) -> int:
        if self.max_samples is not None:
            return min(self.max_samples, len(self.coco_dataset))
        return len(self.coco_dataset)

    def _process_pil(self, image: Image.Image) -> torch.Tensor:
        encoding = self.processor(
            images=image,
            return_tensors="pt",
            size=FIXED_IMAGE_SIZE,
        )
        return encoding["pixel_values"].squeeze(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a preprocessed sample.

        Returns:
            Dictionary containing:
                - pixel_values: Preprocessed image tensor
                - image_id: Original COCO image ID
                - original_size: Original image dimensions (height, width)
                - skipped: True if we had to skip due to repeated permission errors
        """
        image_id = self.coco_dataset.ids[idx]

        # Prefer COCO metadata for size (does not require opening the file)
        img_info = self.coco.imgs.get(image_id, None)
        if img_info is not None and "height" in img_info and "width" in img_info:
            original_size = (int(img_info["height"]), int(img_info["width"]))
        else:
            original_size = (0, 0)

        last_exc: BaseException | None = None

        # Attempt normal load + preprocess
        for attempt in range(self.permission_retry_count + 1):  # 0..count (initial + retries)
            try:
                image, _raw_target = self.coco_dataset[idx]
                pixel_values = self._process_pil(image)
                return {
                    "pixel_values": pixel_values,
                    "image_id": image_id,
                    "original_size": original_size if original_size != (0, 0) else (image.height, image.width),
                    "skipped": False,
                }
            except BaseException as e:
                last_exc = e
                if _is_permission_denied(e) and attempt < self.permission_retry_count:
                    warnings.warn(
                        f"[COCO] Permission denied for idx={idx} (image_id={image_id}). "
                        f"Sleeping {self.permission_retry_sleep_s}s then retrying "
                        f"({attempt+1}/{self.permission_retry_count}). Error: {e!r}"
                    )
                    time.sleep(self.permission_retry_sleep_s)
                    continue
                # Non-permission errors, or retries exhausted: break and “skip”
                break

        # Skip behavior: return a dummy (black) image processed identically.
        # This prevents the dataloader/eval loop from crashing.
        warnings.warn(
            f"[COCO] Skipping idx={idx} (image_id={image_id}) after permission retries. "
            f"Returning dummy sample. Last error: {last_exc!r}"
        )
        dummy = Image.new("RGB", (1, 1), (0, 0, 0))
        pixel_values = self._process_pil(dummy)
        return {
            "pixel_values": pixel_values,
            "image_id": image_id,
            "original_size": original_size,
            "skipped": True,
        }


class CollateFn:
    """
    Collate function for YOLOS + COCO.

    Since we use a fixed image size during preprocessing, all tensors have
    identical dimensions and can be simply stacked without padding.
    """

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        image_ids = [item["image_id"] for item in batch]
        original_sizes = [item["original_size"] for item in batch]
        skipped = [bool(item.get("skipped", False)) for item in batch]

        return {
            "pixel_values": pixel_values,  # [B, 3, H, W]
            "image_ids": image_ids,
            "original_sizes": original_sizes,
            "skipped": skipped,            # [B] flags
        }


def create_dataloader(
    images_dir: str | Path,
    annotations_file: str | Path,
    processor: YolosImageProcessor,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: int | None = None,
) -> tuple[DataLoader, CocoYolosDataset]:
    dataset = CocoYolosDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        processor=processor,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CollateFn(),
        pin_memory=False,
    )

    return dataloader, dataset
