from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from src.triton_kernels.triton_bf16acc_linear_ste import TritonBF16AccLinearSTE


LINEAR_GROUPS = {"mlp", "qkv", "head"}


@dataclass
class LinearTarget:
    name: str
    module: nn.Linear
    group: str
    block_idx: Optional[int]


def _detect_group(name: str, module: nn.Module) -> Optional[str]:
    lname = name.lower()
    if "class_labels_classifier" in lname or "bbox_predictor" in lname or "box_predictor" in lname:
        return "head"
    if "attention" in lname or "attn" in lname:
        return "qkv"
    if "mlp" in lname or "intermediate" in lname or "output.dense" in lname:
        return "mlp"
    # Default: treat generic linear layers in encoder as MLP
    if "layer" in lname and isinstance(module, nn.Linear):
        return "mlp"
    return None


def _extract_block_idx(name: str) -> Optional[int]:
    match = re.search(r"\.layer\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


def collect_linear_targets(module: nn.Module) -> List[LinearTarget]:
    targets: List[LinearTarget] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            group = _detect_group(name, child)
            if group is None:
                continue
            targets.append(LinearTarget(name=name, module=child, group=group, block_idx=_extract_block_idx(name)))
    return targets


def _cast_parameter_(param: nn.Parameter, dtype: torch.dtype) -> nn.Parameter:
    with torch.no_grad():
        param.data = param.data.to(dtype)
    return param


def _wrap_with_triton_linear(module: nn.Module, target: LinearTarget) -> nn.Module:
    old = target.module
    # Instantiate wrapper
    wrapper = TritonBF16AccLinearSTE(
        old.in_features,
        old.out_features,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=torch.bfloat16,
    )
    # Drop wrapper parameters and reuse originals to preserve optimizer state
    wrapper._parameters.pop("weight")
    wrapper.register_parameter("weight", _cast_parameter_(old.weight, torch.bfloat16))
    if old.bias is not None:
        wrapper._parameters.pop("bias")
        wrapper.register_parameter("bias", _cast_parameter_(old.bias, torch.bfloat16))
    else:
        wrapper._parameters.pop("bias", None)
        wrapper.register_parameter("bias", None)
    return wrapper


def apply_triton_linear_patches(
    module: nn.Module,
    *,
    enabled_groups: Set[str],
    max_block: Optional[int] = None,
) -> List[str]:
    """Replace selected Linear modules with TritonBF16AccLinearSTE wrappers.

    Args:
        module: Root module to patch.
        enabled_groups: Which semantic groups to patch (mlp/qkv/head).
        max_block: If provided, only blocks with ``block_idx`` <= max_block will
            be patched.

    Returns:
        List of module names that were patched in this call.
    """

    patched: List[str] = []
    for target in collect_linear_targets(module):
        if target.group not in enabled_groups:
            continue
        if max_block is not None and target.block_idx is not None and target.block_idx > max_block:
            continue
        parent_path, _, child_name = target.name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        setattr(parent, child_name, _wrap_with_triton_linear(module, target))
        patched.append(target.name)
    return patched


def rebuild_optimizer(optimizer: torch.optim.Optimizer, model: nn.Module, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Rebuild an AdamW optimizer after patching while preserving hyperparams.

    This is useful when new parameters are introduced; in our case we reuse the
    same Parameter objects, so rebuilding is typically unnecessary. However, the
    helper is provided for robustness and is validated by tests.
    """

    opt_cls = type(optimizer)
    defaults = optimizer.defaults.copy()
    lr = defaults.get("lr", optimizer.param_groups[0].get("lr", 1e-4))
    params = [p for p in model.parameters() if p.requires_grad]
    new_opt = opt_cls(params, lr=lr, weight_decay=weight_decay, **{k: v for k, v in defaults.items() if k not in {"params", "lr"}})
    return new_opt
