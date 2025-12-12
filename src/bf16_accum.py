"""
BF16 Accumulator Emulation Utilities

This module provides utilities to emulate true BF16 accumulation semantics in software.
The key insight is that standard PyTorch operations may use higher-precision accumulators
internally, so we need to explicitly perform additions with BF16 rounding after each step.
"""
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.yolos.modeling_yolos import (
    YolosPatchEmbeddings,
    YolosSelfAttention,
)

from .triton_kernels.triton_bf16acc_linear_ste import TritonBF16AccLinearSTE
from .triton_kernels.triton_bf16acc_bmm import triton_bf16acc_matmul


def to_bf16(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor values are representable in bfloat16."""
    return x.to(torch.bfloat16)


def bf16_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply with bf16 rounding on inputs and outputs.

    Args:
        a: First tensor
        b: Second tensor (must be broadcastable with a)

    Returns:
        Product in bfloat16
    """
    return (to_bf16(a) * to_bf16(b)).to(torch.bfloat16)


def bf16_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise add with bf16 rounding on inputs and outputs.

    Args:
        a: First tensor
        b: Second tensor (must be broadcastable with a)

    Returns:
        Sum in bfloat16
    """
    return (to_bf16(a) + to_bf16(b)).to(torch.bfloat16)


def bf16_accum_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Emulate a matmul where inputs and accumulations happen in BF16.

    Args:
        x: Input tensor of shape (..., K)
        w: Weight tensor of shape (K, N)

    Returns:
        Output tensor of shape (..., N) in bfloat16
    """
    assert x.dtype == torch.bfloat16, f"Expected bf16 input, got {x.dtype}"
    assert w.dtype == torch.bfloat16, f"Expected bf16 weight, got {w.dtype}"

    orig_shape = x.shape[:-1]
    K = x.shape[-1]
    N = w.shape[1]

    x2d = x.reshape(-1, K)
    w2d = w
    B = x2d.shape[0]

    out = torch.zeros(B, N, dtype=torch.bfloat16, device=x.device)

    for k in range(K):
        x_k = x2d[:, k].unsqueeze(1)
        w_k = w2d[k, :].unsqueeze(0)
        prod = bf16_mul(x_k, w_k)
        out = bf16_add(out, prod)

    out = out.reshape(*orig_shape, N)
    return out


def bf16_accum_bmm(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = False) -> torch.Tensor:
    """
    Emulated BF16 batched matmul.

    Args:
        a: Tensor of shape (..., M, K)
        b: Tensor of shape (..., K, N) if transpose_b is False, otherwise (..., N, K)
        transpose_b: Whether to transpose the last two dims of b before multiplication

    Returns:
        Tensor of shape (..., M, N) in bfloat16
    """
    assert a.dtype == torch.bfloat16, f"Expected bf16 input, got {a.dtype}"
    assert b.dtype == torch.bfloat16, f"Expected bf16 input, got {b.dtype}"

    if transpose_b:
        b = b.transpose(-1, -2)

    batch_dims = a.shape[:-2]
    M, K = a.shape[-2:]
    K2, N = b.shape[-2:]
    assert K == K2, "Inner dimensions must match for matmul"

    a_flat = a.reshape(-1, M, K)
    b_flat = b.reshape(-1, K, N)

    out_flat = torch.empty(a_flat.shape[0], M, N, dtype=torch.bfloat16, device=a.device)

    for i in range(a_flat.shape[0]):
        out_flat[i] = bf16_accum_matmul(a_flat[i], b_flat[i])

    return out_flat.reshape(*batch_dims, M, N)


class BF16AccumConv2d(nn.Module):
    """Conv2d layer that accumulates in BF16 via im2col + matmul."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        k_h, k_w = self.kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, k_h, k_w, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d) -> "BF16AccumConv2d":
        new = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            bias=conv.bias is not None,
        )
        new.weight.data.copy_(conv.weight.data.to(torch.bfloat16))
        if conv.bias is not None:
            new.bias.data.copy_(conv.bias.data.to(torch.bfloat16))
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.bfloat16)
        B, C_in, H, W = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride

        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        W_flat = self.weight.view(self.out_channels, -1)
        K = W_flat.shape[1]
        assert patches.shape[1] == K, "Unfolded patch dimension mismatch"

        patches_T = patches.transpose(1, 2)
        B_, L, K_ = patches_T.shape
        assert K_ == K

        x2d = patches_T.reshape(B_ * L, K)
        W_t = W_flat.transpose(0, 1)

        if x2d.is_cuda:
            out2d = triton_bf16acc_matmul(x2d, W_t)
        else:
            out2d = bf16_accum_matmul(x2d, W_t)

        out = out2d.reshape(B_, L, self.out_channels).transpose(1, 2)

        H_out = (H - k_h) // s_h + 1
        W_out = (W - k_w) // s_w + 1
        out = out.view(B_, self.out_channels, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out


class BF16AccumLinear(nn.Module):
    """
    Linear layer that uses BF16-accumulating matmul.

    This module emulates the behavior of hardware that uses BF16 for both
    operands and accumulators in matrix multiply operations.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "BF16AccumLinear":
        """
        Create a BF16AccumLinear from an existing nn.Linear module.

        Args:
            linear: Source linear layer

        Returns:
            New BF16AccumLinear with weights copied from source
        """
        new = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        new.weight.data.copy_(linear.weight.data.to(torch.bfloat16))
        if linear.bias is not None:
            new.bias.data.copy_(linear.bias.data.to(torch.bfloat16))
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using BF16-accumulating matmul.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features) in bfloat16
        """
        x = x.to(torch.bfloat16)
        # Weight shape is (out_features, in_features)
        # We need (in_features, out_features) for our matmul
        y = bf16_accum_matmul(x, self.weight.transpose(0, 1))
        if self.bias is not None:
            y = bf16_add(y, self.bias)
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def replace_linear_with_bf16_accum(module: nn.Module) -> None:
    """
    Recursively replace all nn.Linear layers with BF16AccumLinear.

    This function modifies the module in-place, replacing standard Linear
    layers with custom ones that use BF16-accumulating matmul.

    Args:
        module: Root module to patch
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            device_type = child.weight.device.type if child.weight is not None else None
            if device_type == "cuda":
                new_linear = TritonBF16AccLinearSTE.from_linear(child)
            else:
                new_linear = BF16AccumLinear.from_linear(child)
            setattr(module, name, new_linear)
        else:
            replace_linear_with_bf16_accum(child)


def replace_conv_with_bf16_accum(module: nn.Module) -> None:
    """
    Recursively replace YOLOS patch embedding conv with BF16AccumConv2d.
    """

    for name, child in list(module.named_children()):
        if isinstance(child, YolosPatchEmbeddings) and isinstance(child.projection, nn.Conv2d):
            child.projection = BF16AccumConv2d.from_conv2d(child.projection)
        else:
            replace_conv_with_bf16_accum(child)


def yolos_self_attention_forward_bf16(self, hidden_states, head_mask=None, output_attentions=False):
    """
    BF16-accumulating forward pass for YolosSelfAttention.

    Matches the current transformers API which uses manual reshape/transpose
    instead of transpose_for_scores method.
    """
    hidden_states = hidden_states.to(torch.bfloat16)
    batch_size = hidden_states.shape[0]

    # Reshape and transpose for multi-head attention
    # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, seq_len, head_size)
    new_shape = (batch_size, -1, self.num_attention_heads, self.attention_head_size)

    key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
    value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
    query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

    # BF16-accumulating attention scores: Q @ K^T
    if query_layer.is_cuda:
        attention_scores = triton_bf16acc_matmul(query_layer, key_layer.transpose(-1, -2))
    else:
        attention_scores = bf16_accum_bmm(query_layer, key_layer, transpose_b=True)
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # Softmax in fp32 for numerical stability, then back to bf16
    attention_probs = F.softmax(attention_scores.to(torch.float32), dim=-1).to(torch.bfloat16)

    # Apply dropout during training
    if self.training and self.dropout_prob > 0:
        attention_probs = F.dropout(attention_probs, p=self.dropout_prob, training=True)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask.to(torch.bfloat16)

    # BF16-accumulating context: attn_probs @ V
    if attention_probs.is_cuda:
        context_layer = triton_bf16acc_matmul(attention_probs, value_layer)
    else:
        context_layer = bf16_accum_bmm(attention_probs, value_layer)

    # Reshape back: (batch_size, num_heads, seq_len, head_size) -> (batch_size, seq_len, hidden_size)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs if output_attentions else None)
    return outputs


def patch_yolos_self_attention_bf16(module: nn.Module) -> None:
    """
    Monkey-patch YOLOS self-attention modules to use bf16-accum matmuls.
    """

    for submodule in module.modules():
        if isinstance(submodule, YolosSelfAttention):
            submodule.forward = types.MethodType(yolos_self_attention_forward_bf16, submodule)

