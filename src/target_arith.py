"""Target arithmetic for YOLOS: per-component BF16 accumulation, usable in eval and training.

One construction is used everywhere (inference ablation, fine-tuning, evaluation):

* the model keeps FP32 master parameters and runs under ``torch.autocast(bfloat16)``;
  matmul operands are bf16, LayerNorm / softmax (fp32) / GELU / residuals follow autocast;
* every matmul-like op belongs to a component; when a component is enabled, its matmul
  is computed with a BF16 accumulator (Triton, forward) and an STE backward
  (:class:`~src.triton_kernels.bf16acc_core.BF16AccMatmulSTE`); otherwise the standard
  BF16 kernel with FP32 accumulation is used;
* components can be switched at any step (progressive schedules) without touching
  parameters or optimizer state.

Components: patch (patch-embedding conv), qkv, attn_out, mlp, head (linear layers),
qk (Q @ K^T), av (softmax @ V).
"""
from __future__ import annotations

import re
import types
from typing import Iterable, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.yolos.modeling_yolos import YolosSelfAttention

from .triton_kernels.bf16acc_core import bf16acc_matmul_ste

COMPONENTS = ("patch", "qkv", "attn_out", "mlp", "head", "qk", "av")
ALIASES = {
    "lin": ("qkv", "attn_out", "mlp", "head"),
    "attn": ("qk", "av"),
    "all": COMPONENTS,
    "none": (),
}
# Groups of the progressive schedule used in the paper.
SCHEDULE_GROUPS = {
    "patch": ("patch",),
    "attn": ("qk", "av"),
    "mlp": ("mlp",),
    "qkvo": ("qkv", "attn_out"),  # Q/K/V and attention-output projections ("qkv" stage of the paper)
    "head": ("head",),
}


def parse_components(spec: str | Iterable[str] | None) -> Set[str]:
    """Parse e.g. ``"lin,patch"``, ``"all-av"``, ``"all-qk-av"`` or ``"none"``."""
    if spec is None:
        return set()
    if not isinstance(spec, str):
        spec = ",".join(spec)
    out: Set[str] = set()
    for item in filter(None, (s.strip() for s in spec.split(","))):
        head, *minus = item.split("-")
        names = ALIASES.get(head, (head,))
        cur = set(names)
        for m in minus:
            cur -= set(ALIASES.get(m, (m,)))
        out |= cur
    unknown = out - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components {sorted(unknown)}")
    return out


class TargetArithmetic:
    """Holds the set of components that currently use a BF16 accumulator."""

    def __init__(self, enabled: Optional[Set[str]] = None):
        self.enabled: Set[str] = set(enabled or ())

    def set(self, components: Iterable[str]) -> None:
        self.enabled = set(components)

    def on(self, component: str) -> bool:
        return component in self.enabled

    def __repr__(self) -> str:
        return f"TargetArithmetic({','.join(c for c in COMPONENTS if c in self.enabled) or 'none'})"


class AccLinear(nn.Module):
    """nn.Linear that can switch to a BF16-accumulating matmul. Reuses the original Parameters."""

    def __init__(self, linear: nn.Linear, component: str, ctrl: TargetArithmetic):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.component = component
        self.ctrl = ctrl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ctrl.on(self.component):
            return F.linear(x, self.weight, self.bias)
        x16 = x.to(torch.bfloat16)
        w16 = self.weight.to(torch.bfloat16)
        y = bf16acc_matmul_ste(x16.reshape(-1, self.in_features), w16.t())
        if self.bias is not None:
            y = y + self.bias.to(torch.bfloat16)  # one bf16 rounding of (acc + bias)
        return y.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"{self.in_features}->{self.out_features}, component={self.component}"


class AccPatchConv(nn.Module):
    """Non-overlapping patch-embedding Conv2d (kernel == stride) as im2col + matmul."""

    def __init__(self, conv: nn.Conv2d, ctrl: TargetArithmetic):
        super().__init__()
        if conv.kernel_size != conv.stride or conv.padding != (0, 0) or conv.groups != 1:
            raise ValueError("only non-overlapping patch convolutions are supported")
        self.weight = conv.weight
        self.bias = conv.bias
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.out_channels = conv.out_channels
        self.ctrl = ctrl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ctrl.on("patch"):
            return F.conv2d(x, self.weight, self.bias, stride=self.stride)
        B, _, H, W = x.shape
        kh, kw = self.kernel_size
        h_out, w_out = H // kh, W // kw
        patches = F.unfold(x.to(torch.bfloat16), kernel_size=self.kernel_size, stride=self.stride)  # (B, C*kh*kw, L)
        cols = patches.transpose(1, 2).reshape(B * h_out * w_out, -1)
        w16 = self.weight.to(torch.bfloat16).reshape(self.out_channels, -1)
        y = bf16acc_matmul_ste(cols, w16.t())
        if self.bias is not None:
            y = y + self.bias.to(torch.bfloat16)
        return y.reshape(B, h_out * w_out, self.out_channels).transpose(1, 2).reshape(B, self.out_channels, h_out, w_out)


def _self_attention_forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None):
    """Mirror of HF eager YOLOS self-attention with switchable BF16-accum matmuls."""
    ctrl: TargetArithmetic = self._target_ctrl
    batch_size = hidden_states.shape[0]
    new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size
    key = self.key(hidden_states).view(*new_shape).transpose(1, 2)
    value = self.value(hidden_states).view(*new_shape).transpose(1, 2)
    query = self.query(hidden_states).view(*new_shape).transpose(1, 2)

    if ctrl.on("qk"):
        scores = bf16acc_matmul_ste(query.to(torch.bfloat16), key.to(torch.bfloat16).transpose(-1, -2))
    else:
        scores = torch.matmul(query, key.transpose(-1, -2))
    scores = scores * self.scaling
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    probs = F.dropout(probs, p=self.dropout_prob if self.training else 0.0, training=self.training)
    if head_mask is not None:
        probs = probs * head_mask
    if ctrl.on("av"):
        context = bf16acc_matmul_ste(probs.to(torch.bfloat16), value.to(torch.bfloat16))
    else:
        context = torch.matmul(probs, value)
    context = context.transpose(1, 2).contiguous()
    context = context.reshape(context.size()[:-2] + (self.all_head_size,))
    return context, probs


_LINEAR_RULES = [
    (re.compile(r"\.attention\.attention\.(query|key|value)$"), "qkv"),
    (re.compile(r"\.attention\.output\.dense$"), "attn_out"),
    (re.compile(r"\.intermediate\.dense$"), "mlp"),
    (re.compile(r"\.layer\.\d+\.output\.dense$"), "mlp"),
    (re.compile(r"^(class_labels_classifier|bbox_predictor)\."), "head"),
]


def linear_component(name: str) -> str:
    for pattern, comp in _LINEAR_RULES:
        if pattern.search(name):
            return comp
    raise ValueError(f"Linear layer {name!r} is not assigned to any component")


def install_target_arithmetic(model: nn.Module, enabled: Iterable[str] = ()) -> TargetArithmetic:
    """Wrap all matmul-like ops of a YOLOS model in place; returns the controller."""
    ctrl = TargetArithmetic(set(enabled))
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear):
                setattr(module, child_name, AccLinear(child, linear_component(full), ctrl))
            elif isinstance(child, nn.Conv2d):
                if not full.endswith("patch_embeddings.projection"):
                    raise ValueError(f"unexpected Conv2d {full}")
                setattr(module, child_name, AccPatchConv(child, ctrl))
    n_attn = 0
    for module in model.modules():
        if isinstance(module, YolosSelfAttention):
            module._target_ctrl = ctrl
            module.forward = types.MethodType(_self_attention_forward, module)
            n_attn += 1
    if n_attn == 0:
        raise ValueError("no YolosSelfAttention modules found")
    model._target_ctrl = ctrl
    return ctrl


class ProgressiveSchedule:
    """``"patch@0,attn@60,mlp@120,qkv@180,head@240"`` -> components enabled at a given step.

    Milestones are absolute optimizer steps; group names from :data:`SCHEDULE_GROUPS`
    or component names.
    """

    def __init__(self, spec: str):
        self.stages = []
        for item in filter(None, (s.strip() for s in spec.split(","))):
            group, step = item.split("@")
            comps = SCHEDULE_GROUPS.get(group) or tuple(parse_components(group))
            self.stages.append((int(step), set(comps)))
        self.stages.sort(key=lambda s: s[0])

    def components_at(self, step: int) -> Set[str]:
        out: Set[str] = set()
        for start, comps in self.stages:
            if step >= start:
                out |= comps
        return out

    def full_at(self) -> int:
        return self.stages[-1][0] if self.stages else 0
