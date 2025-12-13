import pathlib
import sys

import torch
from torch.optim import AdamW
from transformers import YolosForObjectDetection

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.patching import apply_triton_linear_patches


def test_parameter_ids_preserved():
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    params_before = list(model.parameters())
    ids_before = {id(p) for p in params_before}
    opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

    patched = apply_triton_linear_patches(model, enabled_groups={"mlp", "qkv", "head"})
    assert patched, "expected some modules to be patched"

    params_after = list(model.parameters())
    ids_after = {id(p) for p in params_after}
    assert len(params_before) == len(params_after)
    assert ids_before == ids_after
    assert opt_param_ids == {id(p) for group in optimizer.param_groups for p in group["params"]}
