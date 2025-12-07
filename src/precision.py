"""
Precision Mode Model Builders

This module provides functions to build YOLOS models under different precision modes:
1. FP32 - Full precision
2. BF16 Default - BF16 with hardware (higher precision) accumulators
3. BF16 Accum - BF16 with software-emulated BF16 accumulators
"""
import copy
import torch
import torch.nn as nn
from transformers import YolosForObjectDetection, YolosImageProcessor

from .config import MODEL_NAME, PrecisionMode
from .bf16_accum import replace_linear_with_bf16_accum


def load_base_model() -> tuple[YolosForObjectDetection, YolosImageProcessor]:
    """
    Load the base YOLOS model and processor from HuggingFace Hub.
    
    Always loads in FP32 on CPU to serve as the reference checkpoint.
    
    Returns:
        Tuple of (model, processor)
    """
    model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
    processor = YolosImageProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def build_fp32_model(
    base_model: YolosForObjectDetection,
    device: torch.device,
) -> YolosForObjectDetection:
    """
    Build an FP32 model for inference.
    
    Args:
        base_model: Base model loaded in FP32
        device: Target device (cuda or cpu)
    
    Returns:
        Model ready for FP32 inference
    """
    model = copy.deepcopy(base_model)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def build_bf16_default_model(
    base_model: YolosForObjectDetection,
    device: torch.device,
) -> YolosForObjectDetection:
    """
    Build a BF16 model with default (hardware) accumulators.
    
    Uses standard PyTorch BF16 operations which typically use higher
    precision accumulators internally.
    
    Args:
        base_model: Base model loaded in FP32
        device: Target device (cuda or cpu)
    
    Returns:
        Model ready for BF16 inference with default accumulators
    """
    model = copy.deepcopy(base_model)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def build_bf16_accum_model(
    base_model: YolosForObjectDetection,
    device: torch.device,
) -> YolosForObjectDetection:
    """
    Build a BF16 model with software-emulated BF16 accumulators.
    
    Replaces all nn.Linear layers with custom BF16AccumLinear layers
    that emulate true BF16 accumulation behavior.
    
    Args:
        base_model: Base model loaded in FP32
        device: Target device (cuda or cpu)
    
    Returns:
        Model ready for BF16 inference with emulated BF16 accumulators
    """
    model = copy.deepcopy(base_model)
    # Replace linear layers FIRST (while still on CPU) to avoid device mismatch
    # when BF16AccumLinear.from_linear() creates new parameters
    replace_linear_with_bf16_accum(model)
    # Then move entire model (including replaced layers) to device and convert to bf16
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def build_model(
    base_model: YolosForObjectDetection,
    precision_mode: PrecisionMode,
    device: torch.device,
) -> YolosForObjectDetection:
    """
    Build a model for the specified precision mode.
    
    Args:
        base_model: Base model loaded in FP32
        precision_mode: Target precision mode
        device: Target device
    
    Returns:
        Model ready for inference in the specified precision mode
    """
    builders = {
        PrecisionMode.FP32: build_fp32_model,
        PrecisionMode.BF16_DEFAULT: build_bf16_default_model,
        PrecisionMode.BF16_ACCUM: build_bf16_accum_model,
    }
    
    builder = builders[precision_mode]
    return builder(base_model, device)

