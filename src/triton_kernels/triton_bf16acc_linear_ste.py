import math
from typing import Optional

import torch
import torch.nn as nn

import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Autotune configs
# NOTE: Because we explicitly accumulate in bf16 (no tl.dot), very large tiles
# can explode register pressure. Keep these modest; extend carefully.
# -----------------------------------------------------------------------------
BF16ACC_LINEAR_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
]


@triton.autotune(configs=BF16ACC_LINEAR_CONFIGS, key=["M", "N", "K"])
@triton.jit
def linear_fwd_bf16acc_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Grouped ordering (same as matmul tutorial style)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_in_group = GROUP_M * grid_n
    group_id = pid // pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % pid_in_group) % group_size_m
    pid_n = (pid % pid_in_group) // group_size_m

    # "Raw" offsets used for masks / stores (no modulo)
    offs_m_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Modulo offsets used for safe in-bounds pointers even when masked
    offs_m = offs_m_raw % M
    offs_n = offs_n_raw % N

    # bf16 accumulator tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)

    # Loop over K tiles
    k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in range(0, k_tiles):
        k_base = kt * BLOCK_K

        # Unroll within the tile
        for kk in tl.static_range(0, BLOCK_K):
            k = k_base + kk
            k_in = k < K  # 0-d tensor scalar (broadcasts fine in masks)

            # Load x[:, k] as a vector (BLOCK_M,)
            xk = tl.load(
                x_ptr + offs_m * stride_xm + k * stride_xk,
                mask=(offs_m_raw < M) & k_in,
                other=0,
            ).to(tl.bfloat16)

            # Load w[:, k] (i.e., W[n, k]) as a vector (BLOCK_N,)
            wk = tl.load(
                w_ptr + offs_n * stride_wn + k * stride_wk,
                mask=(offs_n_raw < N) & k_in,
                other=0,
            ).to(tl.bfloat16)

            # bf16 MAC update
            acc += xk[:, None] * wk[None, :]

    if HAS_BIAS:
        bias = tl.load(
            b_ptr + offs_n,
            mask=offs_n_raw < N,
            other=0,
        ).to(tl.bfloat16)
        acc += bias[None, :]

    # Store
    y_ptrs = y_ptr + offs_m_raw[:, None] * stride_ym + offs_n_raw[None, :] * stride_yn
    y_mask = (offs_m_raw[:, None] < M) & (offs_n_raw[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)



def triton_linear_forward_bf16acc(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: (..., K) bf16
    w: (N, K)  bf16 (torch.nn.Linear weight layout)
    b: (N,) bf16 or None
    returns: (..., N) bf16
    """
    if not (x.is_cuda and w.is_cuda):
        raise RuntimeError("This Triton kernel targets CUDA/HIP GPUs.")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError(f"bf16-only forward: got x={x.dtype}, w={w.dtype}")
    if b is not None and b.dtype != torch.bfloat16:
        raise TypeError(f"bf16-only forward: got b={b.dtype}")

    # Flatten x -> 2D
    orig_shape = x.shape
    K = orig_shape[-1]
    x2d = x.reshape(-1, K)
    M = x2d.shape[0]
    N = w.shape[0]

    # Output bf16
    y2d = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    # Launch
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    linear_fwd_bf16acc_kernel[grid](
        x2d, w, b if b is not None else y2d, y2d,
        M, N, K,
        x2d.stride(0), x2d.stride(1),
        w.stride(0), w.stride(1),
        y2d.stride(0), y2d.stride(1),
        HAS_BIAS=(b is not None),
    )

    return y2d.reshape(*orig_shape[:-1], N)


class TritonBF16AccLinearSTEFunction(torch.autograd.Function):
    """
    Forward: Triton bf16-only + bf16 accumulation
    Backward: standard Linear gradients via torch ops (STE-style w.r.t. forward)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        # Enforce bf16-only forward
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError(f"bf16-only forward: got x={x.dtype}, w={weight.dtype}")
        if bias is not None and bias.dtype != torch.bfloat16:
            raise TypeError(f"bf16-only forward: got b={bias.dtype}")

        # Make contiguous for predictable strides
        x_c = x.contiguous()
        w_c = weight.contiguous()
        b_c = bias.contiguous() if bias is not None else None

        y = triton_linear_forward_bf16acc(x_c, w_c, b_c)

        ctx.save_for_backward(x_c, w_c, b_c if b_c is not None else torch.tensor([], device=x.device))
        ctx.has_bias = (b_c is not None)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Standard linear backward:
          dX = dY @ W
          dW = dY^T @ X
          db = sum(dY)
        """
        x, w, b_or_empty = ctx.saved_tensors
        has_bias = ctx.has_bias

        grad_out = grad_out.contiguous()

        # Flatten to 2D
        K = x.shape[-1]
        x2d = x.reshape(-1, K)                 # (M, K)
        N = w.shape[0]
        dy2d = grad_out.reshape(-1, N)         # (M, N)

        dx = dw = db = None

        # dX
        if ctx.needs_input_grad[0]:
            dx2d = dy2d.matmul(w)             # (M, K)
            dx = dx2d.reshape_as(x)

        # dW
        if ctx.needs_input_grad[1]:
            dw = dy2d.transpose(0, 1).matmul(x2d)  # (N, K)

        # db
        if has_bias and ctx.needs_input_grad[2]:
            db = dy2d.sum(dim=0)

        return dx, dw, db


class TritonBF16AccLinearSTE(nn.Module):
    """
    Drop-in nn.Linear replacement:
      - forward: Triton bf16 accumulate
      - backward: torch matmul gradients (standard linear backward)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if dtype != torch.bfloat16:
            raise TypeError("This module is bf16-only by design (dtype must be torch.bfloat16).")

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonBF16AccLinearSTEFunction.apply(x, self.weight, self.bias)

    @staticmethod
    def from_linear(m: nn.Linear, *, cast_to_bf16: bool = True) -> "TritonBF16AccLinearSTE":
        """
        Create TritonBF16AccLinearSTE from an existing nn.Linear.
        If cast_to_bf16=True, weights/bias are cast to bf16.
        """
        if cast_to_bf16:
            w = m.weight.detach().to(torch.bfloat16)
            b = m.bias.detach().to(torch.bfloat16) if m.bias is not None else None
        else:
            w = m.weight.detach()
            b = m.bias.detach() if m.bias is not None else None

        out = TritonBF16AccLinearSTE(
            m.in_features,
            m.out_features,
            bias=(m.bias is not None),
            device=m.weight.device,
            dtype=torch.bfloat16,
        )
        out.weight.data.copy_(w)
        if b is not None:
            out.bias.data.copy_(b)
        return out


def replace_linears_bf16acc_ste(module: nn.Module, *, cast_to_bf16: bool = True) -> nn.Module:
    """
    Recursively replace nn.Linear with TritonBF16AccLinearSTE (weights/bias copied).
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, TritonBF16AccLinearSTE.from_linear(child, cast_to_bf16=cast_to_bf16))
        else:
            replace_linears_bf16acc_ste(child, cast_to_bf16=cast_to_bf16)
    return module


# -----------------------------------------------------------------------------
# Minimal smoke test (optional)
# -----------------------------------------------------------------------------
def _smoke_test():
    assert torch.cuda.is_available(), "CUDA/HIP required"
    torch.manual_seed(0)

    B, K, N = 512, 256, 256
    device = torch.device("cuda")
    dtype = torch.bfloat16

    x = torch.randn((B, K), device=device, dtype=dtype, requires_grad=True)
    ref = nn.Linear(K, N, bias=True, device=device, dtype=dtype)
    tri = TritonBF16AccLinearSTE.from_linear(ref, cast_to_bf16=True)

    # Forward compare (will differ due to bf16 accumulation)
    with torch.no_grad():
        y_ref = ref(x)
        y_tri = tri(x)
        print("[forward] max |ref-triton| =", (y_ref - y_tri).abs().max().item())

    # Backward compare (should be close to ref backward since we use standard formulas)
    ref.zero_grad(set_to_none=True)
    tri.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    y0 = ref(x); (y0 * y0).mean().backward()
    gx_ref = x.grad.detach().clone()
    gw_ref = ref.weight.grad.detach().clone()
    gb_ref = ref.bias.grad.detach().clone()

    if x.grad is not None:
        x.grad = None
    y1 = tri(x); (y1 * y1).mean().backward()
    gx_tri = x.grad.detach().clone()
    gw_tri = tri.weight.grad.detach().clone()
    gb_tri = tri.bias.grad.detach().clone()

    print("[grad x] max |ref-triton| =", (gx_ref - gx_tri).abs().max().item())
    print("[grad w] max |ref-triton| =", (gw_ref - gw_tri).abs().max().item())
    print("[grad b] max |ref-triton| =", (gb_ref - gb_tri).abs().max().item())


if __name__ == "__main__":
    _smoke_test()
