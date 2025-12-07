"""
Configuration module for paths, constants, and enums.
"""
import os
from enum import Enum
from pathlib import Path


class PrecisionMode(Enum):
    """Supported precision modes for inference."""
    FP32 = "fp32"
    BF16_DEFAULT = "bf16_default"
    BF16_ACCUM = "bf16_accum"


# Model configuration
MODEL_NAME = "hustvl/yolos-tiny"

# Fixed image size for consistent batching
# YOLOS (ViT-based) doesn't support pixel_mask, so we use a fixed size
# to ensure consistent results across different batch sizes
FIXED_IMAGE_SIZE = {"height": 512, "width": 512}

# COCO paths - can be overridden via environment variable or CLI
DEFAULT_COCO_ROOT = os.environ.get("COCO_ROOT", "./coco")


def get_coco_paths(coco_root: str | None = None) -> dict:
    """
    Get paths for COCO 2017 validation dataset.
    
    Args:
        coco_root: Root directory containing COCO data. 
                   Defaults to COCO_ROOT env var or ./coco
    
    Returns:
        Dictionary with 'images' and 'annotations' paths
    """
    root = Path(coco_root or DEFAULT_COCO_ROOT)
    return {
        "images": root / "val2017",
        "annotations": root / "annotations" / "instances_val2017.json",
    }


# Default evaluation settings
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 42

# Results output
DEFAULT_RESULTS_DIR = Path("./results")

