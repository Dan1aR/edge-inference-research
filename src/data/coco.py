"""Compatibility wrappers for data loading.

All logic now lives in :mod:`src.data` to keep a single entry point for
training and evaluation dataloaders. This module re-exports the public API
so older imports keep working.
"""

from src.data import (  # noqa: F401
    DetectionDataConfig,
    TorchvisionCocoDetection,
    build_dataloaders,
    collate_fn_builder,
    load_dataset,
)

__all__ = [
    "DetectionDataConfig",
    "TorchvisionCocoDetection",
    "build_dataloaders",
    "collate_fn_builder",
    "load_dataset",
]
