"""
BF16 Accumulator Emulation Utilities

This module provides utilities to emulate true BF16 accumulation semantics in software.
The key insight is that standard PyTorch operations may use higher-precision accumulators
internally, so we need to explicitly perform additions with BF16 rounding after each step.
"""
import torch
import torch.nn as nn


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
    Emulate matmul where both inputs and the accumulator are bf16.
    
    Uses an outer-product formulation with explicit BF16 rounding after each
    accumulation step, ensuring true BF16 accumulation behavior.
    
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
    
    # Flatten batch dimensions
    x2d = x.reshape(-1, K)  # (B, K)
    w2d = w                  # (K, N)
    B = x2d.shape[0]
    
    # Initialize output accumulator in bf16
    out = torch.zeros(B, N, dtype=torch.bfloat16, device=x.device)
    
    # Loop over reduction dimension K
    # Each step: outer product + bf16 accumulation
    for k in range(K):
        x_k = x2d[:, k].unsqueeze(1)  # (B, 1)
        w_k = w2d[k, :].unsqueeze(0)  # (1, N)
        prod = bf16_mul(x_k, w_k)      # (B, N) in bf16
        out = bf16_add(out, prod)      # bf16 accumulation
    
    # Restore original batch shape
    out = out.reshape(*orig_shape, N)
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
            setattr(module, name, BF16AccumLinear.from_linear(child))
        else:
            replace_linear_with_bf16_accum(child)

