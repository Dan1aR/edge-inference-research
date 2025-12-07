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
        
        # Preprocess image using YOLOS processor
        # We just need the pixel_values for inference
        encoding = self.processor(
            images=image,
            return_tensors="pt",
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
    
    Uses YolosImageProcessor.pad(...) to:
      - pad images in the batch to a common size
      - create a pixel_mask (1 = real pixel, 0 = padding)
    
    This follows the official HuggingFace object detection pipeline pattern.
    """
    
    def __init__(self, image_processor: YolosImageProcessor):
        """
        Initialize the collate function.
        
        Args:
            image_processor: YolosImageProcessor instance for padding
        """
        self.image_processor = image_processor
    
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries from CocoYolosDataset
        
        Returns:
            Batched dictionary ready for model input
        """
        # Lists of per-sample data
        pixel_values_list = [item["pixel_values"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
        original_sizes = [item["original_size"] for item in batch]
        
        # Use the processor's padding logic instead of manual F.pad
        # This pads to the max height/width in the batch, bottom/right, with zeros
        encoding = self.image_processor.pad(pixel_values_list, return_tensors="pt")
        
        out = {
            "pixel_values": encoding["pixel_values"],  # [B, 3, H_max, W_max]
            "image_ids": image_ids,
            "original_sizes": original_sizes,
        }
        
        # YOLOS doesn't require pixel_mask, but pad() provides it;
        # include it in case it's useful for debugging or future use
        if "pixel_mask" in encoding:
            out["pixel_mask"] = encoding["pixel_mask"]  # [B, H_max, W_max]
        
        return out


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
        collate_fn=CollateFn(processor),
        pin_memory=False,  # Disable for MPS/CPU compatibility
    )
    
    return dataloader, dataset
