"""
COCO Dataset and Preprocessing for YOLOS

This module provides dataset classes and utilities for loading COCO 2017
validation data and preprocessing it for the YOLOS model.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from transformers import YolosImageProcessor
from pathlib import Path
from typing import Any

from .config import FIXED_IMAGE_SIZE


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
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Path to COCO images directory (e.g., val2017/)
            annotations_file: Path to COCO annotations JSON
            processor: YolosImageProcessor for preprocessing
            max_samples: Optional limit on number of samples (for quick tests)
        """
        self.coco_dataset = CocoDetection(
            root=str(images_dir),
            annFile=str(annotations_file),
        )
        self.processor = processor
        self.max_samples = max_samples
        
        # Store reference to underlying COCO object for evaluation
        self.coco = self.coco_dataset.coco
    
    def __len__(self) -> int:
        if self.max_samples is not None:
            return min(self.max_samples, len(self.coco_dataset))
        return len(self.coco_dataset)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a preprocessed sample.
        
        Returns:
            Dictionary containing:
                - pixel_values: Preprocessed image tensor
                - image_id: Original COCO image ID
                - original_size: Original image dimensions (height, width)
        """
        image, raw_target = self.coco_dataset[idx]
        
        # Get image ID from COCO
        image_id = self.coco_dataset.ids[idx]
        
        # Get original image size
        original_size = (image.height, image.width)
        
        # Preprocess image using YOLOS processor with fixed size
        # Fixed size ensures consistent tensor shapes across all images,
        # which is required because YOLOS (ViT-based) doesn't support pixel_mask
        encoding = self.processor(
            images=image,
            return_tensors="pt",
            size=FIXED_IMAGE_SIZE,
        )
        
        # Remove batch dimension (will be added by collate_fn)
        pixel_values = encoding["pixel_values"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "image_id": image_id,
            "original_size": original_size,
        }


class CollateFn:
    """
    Collate function for YOLOS + COCO.
    
    Since we use a fixed image size during preprocessing, all tensors have
    identical dimensions and can be simply stacked without padding.
    
    Note: YOLOS (ViT-based) doesn't support pixel_mask for attention masking,
    so we must use fixed-size images to ensure consistent results across
    different batch sizes.
    """
    
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries from CocoYolosDataset
        
        Returns:
            Batched dictionary ready for model input
        """
        # All images have the same fixed size, so we can simply stack them
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        image_ids = [item["image_id"] for item in batch]
        original_sizes = [item["original_size"] for item in batch]
        
        return {
            "pixel_values": pixel_values,  # [B, 3, H, W]
            "image_ids": image_ids,
            "original_sizes": original_sizes,
        }


def create_dataloader(
    images_dir: str | Path,
    annotations_file: str | Path,
    processor: YolosImageProcessor,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: int | None = None,
) -> tuple[DataLoader, CocoYolosDataset]:
    """
    Create a DataLoader for COCO validation.
    
    Args:
        images_dir: Path to COCO images directory
        annotations_file: Path to COCO annotations JSON
        processor: YolosImageProcessor instance
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        max_samples: Optional limit on number of samples
    
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    dataset = CocoYolosDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        processor=processor,
        max_samples=max_samples,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep deterministic order for evaluation
        num_workers=num_workers,
        collate_fn=CollateFn(),
        pin_memory=False,  # Disable for MPS/CPU compatibility
    )
    
    return dataloader, dataset
