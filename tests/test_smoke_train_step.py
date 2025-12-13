import pathlib
import sys

import torch
from PIL import Image
import numpy as np

from transformers import YolosForObjectDetection, YolosImageProcessor

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.patching import apply_attention_patch, apply_triton_linear_patches


def _make_batch(processor: YolosImageProcessor, device):
    images = [Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)) for _ in range(2)]
    annotations = []
    for idx in range(2):
        annotations.append({"image_id": idx, "annotations": [{"bbox": [5, 5, 20, 20], "category_id": 0}]})
    processed = processor(images=images, annotations=annotations, return_tensors="pt")
    processed["pixel_values"] = processed["pixel_values"].to(device)
    processed["labels"] = [{"class_labels": l["class_labels"].to(device), "boxes": l["boxes"].to(device)} for l in processed["labels"]]
    return processed


def _run_step(model, batch):
    outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
    loss = outputs.loss
    loss.backward()
    return loss


def test_baseline_and_patched_step():
    device = torch.device("cpu")
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    batch = _make_batch(processor, device)

    loss = _run_step(model, batch)
    assert torch.isfinite(loss)
    head_param = next(model.class_labels_classifier.parameters())
    assert head_param.grad is not None

    model.zero_grad(set_to_none=True)
    apply_attention_patch(model)
    apply_triton_linear_patches(model, enabled_groups={"mlp", "qkv", "head"})
    loss2 = _run_step(model, batch)
    assert torch.isfinite(loss2)
    head_param = next(model.class_labels_classifier.parameters())
    assert head_param.grad is not None
