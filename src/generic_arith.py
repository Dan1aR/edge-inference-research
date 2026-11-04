"""Model-agnostic target arithmetic: low-precision accumulation for every matmul-like op.

Generalises :mod:`src.target_arith` (YOLOS only) to other HF vision models (YOLOS, DeiT,
DETR, RT-DETR) without rewriting their forward code:

* a ``TorchFunctionMode`` intercepts ``F.linear``, ``F.conv2d`` and the ``matmul``/``bmm``
  calls inside attention modules; forward hooks keep track of the module that issued the op,
  so every op gets an id (``<module name>#<kind>``) and a component;
* an enabled op is computed by the Triton accumulator kernel
  (:func:`src.triton_kernels.bf16acc_core.bf16acc_matmul_ste`, STE backward) on bf16
  operands, exactly as in :mod:`src.target_arith` (tests check bit-equality on YOLOS);
  disabled ops run the original function (autocast bf16, FP32 accumulation);
* the deformable-attention aggregation of RT-DETR (sum over levels x points) is routed
  through the same kernel (component ``msda``);
* optional *centering* (exact algebraic identity, corrections in FP32):
  ``"a"``: A@B = (A - m_A)@B + m_A (1^T B), m_A = row means of A over K;
  ``"b"``: A@B = A@(B - mu_B) + (A 1) mu_B, mu_B = column means of B over K
  (for softmax@V this removes the token mean of V, the source of "swamping").

Components: patch (patch-embedding conv), conv (other convolutions), qkv, attn_out, mlp,
head, aux (other linears: deformable offsets/weights, query heads ...), qk, av, msda.
"""
from __future__ import annotations

import re
import types
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from .triton_kernels.bf16acc_core import bf16acc_matmul_ste

COMPONENTS = ("patch", "conv", "qkv", "attn_out", "mlp", "head", "aux", "qk", "av", "msda")
ALIASES = {
    "lin": ("qkv", "attn_out", "mlp", "head", "aux"),
    "attn": ("qk", "av", "msda"),
    "all": COMPONENTS,
    "none": (),
}


def parse_components(spec: str | Iterable[str] | None) -> Set[str]:
    """``"lin,patch"``, ``"all-av"``, ``"all-qk-av"``, ``"none"`` -> set of components."""
    if spec is None:
        return set()
    if not isinstance(spec, str):
        spec = ",".join(spec)
    out: Set[str] = set()
    for item in filter(None, (s.strip() for s in spec.split(","))):
        head, *minus = item.split("-")
        cur = set(ALIASES.get(head, (head,)))
        for m in minus:
            cur -= set(ALIASES.get(m, (m,)))
        out |= cur
    unknown = out - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components {sorted(unknown)}")
    return out


def parse_center(spec: str | None) -> Dict[str, str]:
    """``"av:b,patch:a,mlp:a"`` -> {component: mode}; aliases allowed (``"lin:a"``)."""
    out: Dict[str, str] = {}
    for item in filter(None, (s.strip() for s in (spec or "").split(","))):
        comp, mode = item.split(":")
        if mode not in ("a", "b"):
            raise ValueError(f"centering mode must be a|b, got {mode}")
        for c in parse_components(comp):
            out[c] = mode
    return out


# ---------------------------------------------------------------------------------------------
# Per-family naming rules: module name -> component
# ---------------------------------------------------------------------------------------------
_HEAD = r"(class_labels_classifier|bbox_predictor|classifier|distillation_classifier|cls_classifier|class_embed|bbox_embed|enc_score_head|enc_bbox_head)"
LINEAR_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(_HEAD + r"(\.|$)"), "head"),
    (re.compile(r"\.attention\.attention\.(query|key|value)$"), "qkv"),          # ViT / YOLOS / DeiT
    (re.compile(r"\.attention\.output\.dense$"), "attn_out"),
    (re.compile(r"\.intermediate\.dense$"), "mlp"),
    (re.compile(r"\.layer\.\d+\.output\.dense$"), "mlp"),
    (re.compile(r"(self_attn|encoder_attn)\.(q_proj|k_proj|v_proj)$"), "qkv"),  # DETR / RT-DETR
    (re.compile(r"(self_attn|encoder_attn)\.out_proj$"), "attn_out"),
    (re.compile(r"encoder_attn\.value_proj$"), "qkv"),                           # deformable attention
    (re.compile(r"encoder_attn\.output_proj$"), "attn_out"),
    (re.compile(r"encoder_attn\.(sampling_offsets|attention_weights)$"), "aux"),
    (re.compile(r"\.(fc1|fc2)$"), "mlp"),
    (re.compile(r"(query_pos_head|enc_output|denoising)"), "aux"),
]
PATCH_RE = re.compile(r"patch_embeddings\.projection$")
ATTENTION_CLASSES = ("YolosSelfAttention", "ViTSelfAttention", "DeiTSelfAttention", "DetrAttention",
                     "RTDetrMultiheadAttention")


def linear_component(name: str) -> str:
    for pattern, comp in LINEAR_RULES:
        if pattern.search(name):
            return comp
    raise ValueError(f"Linear layer {name!r} is not assigned to any component")


@dataclass
class OpStat:
    component: str
    kind: str
    K: int = 0
    outputs: int = 0
    macs: int = 0
    calls: int = 0


@dataclass
class Arith:
    """Controller: which components accumulate in low precision, exclusions, centering, stats."""

    enabled: Set[str] = field(default_factory=set)
    exclude: Set[str] = field(default_factory=set)       # op ids forced to the standard path
    center: Dict[str, str] = field(default_factory=dict)  # component -> "a" | "b"
    record: bool = False
    ops: Dict[str, OpStat] = field(default_factory=dict)
    unhandled: Dict[str, int] = field(default_factory=dict)  # matmuls outside attention: name -> K
    present: Set[str] = field(default_factory=set)  # components of the model (filled by install_arith)
    diag: Optional[Dict[str, dict]] = None  # if a dict: per-op error statistics vs the standard kernel

    def set(self, components: Iterable[str]) -> None:
        self.enabled = set(components)

    def on(self, component: str, op_id: str) -> bool:
        return component in self.enabled and op_id not in self.exclude

    def ops_components(self) -> Set[str]:
        """Components that exist in the instrumented model."""
        return set(self.present) if self.present else set(COMPONENTS)

    @property
    def active(self) -> bool:
        return bool(self.enabled) or self.record


def _centered_matmul(a: torch.Tensor, b: torch.Tensor, mode: Optional[str]) -> torch.Tensor:
    """bf16 operands -> bf16 result of the low-precision-accumulation matmul (+ centering)."""
    if mode is None:
        return bf16acc_matmul_ste(a, b)
    if mode == "a":
        m = a.float().mean(-1, keepdim=True)
        y = bf16acc_matmul_ste((a.float() - m).to(torch.bfloat16), b).float()
        return (y + m * b.float().sum(-2, keepdim=True)).to(torch.bfloat16)
    mu = b.float().mean(-2, keepdim=True)
    y = bf16acc_matmul_ste(a, (b.float() - mu).to(torch.bfloat16)).float()
    return (y + a.float().sum(-1, keepdim=True) * mu).to(torch.bfloat16)


_MATMULS = {torch.matmul, torch.bmm, torch.Tensor.matmul, torch.Tensor.__matmul__, torch.Tensor.bmm}


def _diag_update(ctrl: "Arith", op_id: str, y: torch.Tensor, y_ref: torch.Tensor, ch_dim: int) -> None:
    """Accumulate per-output-channel error sums: systematic (channel mean) vs total error energy."""
    d = (y.float() - y_ref.float()).movedim(ch_dim, -1)
    r = y_ref.float().movedim(ch_dim, -1)
    d2, r2 = d.reshape(-1, d.shape[-1]), r.reshape(-1, r.shape[-1])
    st = ctrl.diag.setdefault(op_id, {"n": 0, "sum": torch.zeros(d2.shape[1], dtype=torch.float64, device=d.device),
                                      "sq": 0.0, "sig": 0.0})
    st["n"] += d2.shape[0]
    st["sum"] += d2.sum(0, dtype=torch.float64)
    st["sq"] += float(d2.double().pow(2).sum())
    st["sig"] += float(r2.double().pow(2).sum())


class _AccMode(TorchFunctionMode):
    def __init__(self, ctrl: Arith, stack: list):
        super().__init__()
        self.ctrl = ctrl
        self.stack = stack

    def _stat(self, op_id, comp, kind, K, outputs):
        if self.ctrl.record:
            st = self.ctrl.ops.setdefault(op_id, OpStat(comp, kind))
            st.K = K
            st.outputs += outputs
            st.macs += outputs * K
            st.calls += 1

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.linear:
            return self._linear(func, *args, **kwargs)
        if func is F.conv2d:
            return self._conv(func, *args, **kwargs)
        if func in _MATMULS and self.stack:
            name, mod = self.stack[-1]
            if type(mod).__name__ in ATTENTION_CLASSES:
                return self._attn(func, name, mod, args, kwargs)
            a, b = args[0], args[1]
            if self.ctrl.record and a.dim() >= 2 and a.shape[-1] > 1:
                self.ctrl.unhandled[f"{name}#matmul"] = int(a.shape[-1])
        return func(*args, **kwargs)

    def _linear(self, func, x, weight, bias=None):
        name, mod = self.stack[-1]
        comp = mod._acc_component
        op_id = f"{name}#{comp}"
        K = weight.shape[1]
        self._stat(op_id, comp, "linear", K, x.numel() // K * weight.shape[0])
        if not self.ctrl.on(comp, op_id):
            return func(x, weight, bias)
        y = _centered_matmul(x.to(torch.bfloat16).reshape(-1, K), weight.to(torch.bfloat16).t(), self.ctrl.center.get(comp))
        if bias is not None:
            y = y + bias.to(torch.bfloat16)  # one bf16 rounding of (acc + bias)
        y = y.reshape(*x.shape[:-1], weight.shape[0])
        if self.ctrl.diag is not None:
            _diag_update(self.ctrl, op_id, y, func(x, weight, bias), -1)
        return y

    def _conv(self, func, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        name, mod = self.stack[-1]
        comp = mod._acc_component
        op_id = f"{name}#{comp}"
        O, Cg, kh, kw = weight.shape
        K = Cg * kh * kw
        pair = lambda v: tuple(v) if isinstance(v, (tuple, list)) else (v, v)  # noqa: E731
        stride, padding, dilation = pair(stride), pair(padding), pair(dilation)
        if isinstance(padding[0], str):
            raise ValueError(f"string padding not supported ({name})")
        B, C, H, W = x.shape
        h_out = (H + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        w_out = (W + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        L = h_out * w_out
        self._stat(op_id, comp, "conv", K, B * L * O)
        if not self.ctrl.on(comp, op_id):
            return func(x, weight, bias, stride, padding, dilation, groups)
        cols = F.unfold(x.to(torch.bfloat16), (kh, kw), dilation=dilation, padding=padding, stride=stride)  # (B, C*kh*kw, L)
        mode = self.ctrl.center.get(comp)
        if groups == 1:
            a = cols.transpose(1, 2).reshape(B * L, K)
            y = _centered_matmul(a, weight.to(torch.bfloat16).reshape(O, K).t(), mode)  # (B*L, O)
            if bias is not None:
                y = y + bias.to(torch.bfloat16)
            y = y.reshape(B, L, O).transpose(1, 2).reshape(B, O, h_out, w_out)
            if self.ctrl.diag is not None:
                _diag_update(self.ctrl, op_id, y, func(x, weight, bias, stride, padding, dilation, groups), 1)
            return y
        G = groups
        a = cols.reshape(B, G, K, L).permute(1, 0, 3, 2).reshape(G, B * L, K)
        w = weight.to(torch.bfloat16).reshape(G, O // G, K).transpose(1, 2)
        y = _centered_matmul(a, w, mode)  # (G, B*L, O/G)
        y = y.reshape(G, B, L, O // G).permute(1, 0, 3, 2).reshape(B, O, L)
        if bias is not None:
            y = y + bias.to(torch.bfloat16)[None, :, None]
        return y.reshape(B, O, h_out, w_out)

    def _attn(self, func, name, mod, args, kwargs):
        idx = mod._acc_calls
        mod._acc_calls += 1
        comp = ("qk", "av")[idx] if idx < 2 else "attn_extra"
        if comp == "attn_extra":
            raise RuntimeError(f"more than 2 matmuls in attention module {name}")
        a, b = args[0], args[1]
        op_id = f"{name}#{comp}"
        K = a.shape[-1]
        self._stat(op_id, comp, "attn", K, a.numel() // K * b.shape[-1])
        if not self.ctrl.on(comp, op_id):
            return func(*args, **kwargs)
        y = _centered_matmul(a.to(torch.bfloat16), b.to(torch.bfloat16), self.ctrl.center.get(comp))
        if self.ctrl.diag is not None:
            _diag_update(self.ctrl, op_id, y, func(*args, **kwargs), -1)
        return y


def _msda_forward(self, value, value_spatial_shapes, value_spatial_shapes_list, level_start_index,
                  sampling_locations, attention_weights, im2col_step):
    """Copy of HF MultiScaleDeformableAttention.forward (pytorch path) with a switchable final sum."""
    ctrl: Arith = self._acc_ctrl
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in value_spatial_shapes_list], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes_list):
        value_l_ = value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros",
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(batch_size * num_heads, 1, num_queries,
                                                                   num_levels * num_points)
    sampled = torch.stack(sampling_value_list, dim=-2).flatten(-2)  # (BH, D, Q, LP)
    op_id = f"{self._acc_name}#msda"
    LP = num_levels * num_points
    if ctrl.record:
        st = ctrl.ops.setdefault(op_id, OpStat("msda", "msda"))
        st.K = LP
        st.outputs += batch_size * num_heads * hidden_dim * num_queries
        st.macs += batch_size * num_heads * hidden_dim * num_queries * LP
        st.calls += 1
    if ctrl.on("msda", op_id):
        a = sampled.permute(0, 2, 1, 3).reshape(-1, hidden_dim, LP).to(torch.bfloat16)  # (BH*Q, D, LP)
        w = attention_weights.permute(0, 2, 3, 1).reshape(-1, LP, 1).to(torch.bfloat16)  # (BH*Q, LP, 1)
        out = _centered_matmul(a, w, ctrl.center.get("msda"))  # (BH*Q, D, 1)
        output = out.reshape(batch_size * num_heads, num_queries, hidden_dim).transpose(1, 2)
        output = output.reshape(batch_size, num_heads * hidden_dim, num_queries)
    else:
        output = (sampled * attention_weights).sum(-1).view(batch_size, num_heads * hidden_dim, num_queries)
    return output.transpose(1, 2).contiguous()


def install_arith(model: nn.Module, enabled: Iterable[str] = ()) -> Arith:
    """Instrument a model in place (no parameter changes); returns the controller."""
    ctrl = Arith(set(enabled))
    stack: list = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            mod._acc_component = linear_component(name)
            ctrl.present.add(mod._acc_component)
        elif isinstance(mod, nn.Conv2d):
            mod._acc_component = "patch" if PATCH_RE.search(name) else "conv"
            ctrl.present.add(mod._acc_component)
        elif isinstance(mod, (nn.Conv1d, nn.Conv3d, nn.ConvTranspose2d)):
            raise ValueError(f"unsupported conv module {name}")
        if type(mod).__name__ in ATTENTION_CLASSES:
            mod._acc_calls = 0
            ctrl.present.update(("qk", "av"))
        if type(mod).__name__ == "MultiScaleDeformableAttention":
            mod._acc_ctrl, mod._acc_name = ctrl, name
            ctrl.present.add("msda")
            mod.forward = types.MethodType(_msda_forward, mod)

        def pre(m, _inp, _name=name):
            if hasattr(m, "_acc_calls"):
                m._acc_calls = 0
            stack.append((_name, m))

        def post(m, _inp, _out):
            if stack:
                stack.pop()

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post, always_call=True)  # keeps the stack balanced on exceptions

    mode = _AccMode(ctrl, stack)
    orig_forward = model.forward

    def forward(*args, **kwargs):
        if not ctrl.active:
            return orig_forward(*args, **kwargs)
        with mode:
            return orig_forward(*args, **kwargs)

    model.forward = forward
    model._target_ctrl = ctrl
    return ctrl


SCHEDULE_GROUPS = {
    "patch": ("patch",),
    "attn": ("qk", "av"),
    "mlp": ("mlp",),
    "qkvo": ("qkv", "attn_out"),
    "head": ("head",),
}


class ProgressiveSchedule:
    """``"patch@0,attn@60,..."`` -> components enabled at a step (generic component names)."""

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
