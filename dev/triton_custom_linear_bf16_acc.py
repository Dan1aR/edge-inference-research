import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

import triton
import triton.language as tl


# -----------------------------
# Autotune configs (keep modest for demo; expand for production)
# -----------------------------
MATMUL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=2,
        num_stages=5,
    ),
]


def _dtype_to_out_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise TypeError(f"Demo supports fp16/bf16, got {dtype}")


@triton.autotune(configs=MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Generic C = A @ B
      A: (M, K)
      B: (K, N)
      C: (M, N)
    Strides are in elements (PyTorch-style).
    """
    pid = tl.program_id(axis=0)

    # Grouped ordering to improve L2 reuse (same idea as Triton matmul tutorial)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_in_group = GROUP_M * grid_n
    group_id = pid // pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % pid_in_group) % group_size_m
    pid_n = (pid % pid_in_group) // group_size_m

    # Offsets
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Optional compiler hints (often help vectorization / codegen)
    # (Hints are documented in Triton language API.)
    tl.multiple_of(offs_k, 8)
    tl.max_contiguous(offs_n, 8)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if OUT_DTYPE == "fp16":
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
    elif OUT_DTYPE == "bf16":
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.static_assert(False, "Unsupported OUT_DTYPE")


@triton.autotune(configs=MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def linear_fwd_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,   # W stored as (N, K): w[n, k] = w_ptr + n*stride_wn + k*stride_wk
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Fused Linear forward:
      Y = X @ W^T + b
    where:
      X: (M, K)
      W: (N, K)  (PyTorch Linear layout)
      Y: (M, N)
      b: (N,)
    """
    pid = tl.program_id(axis=0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_in_group = GROUP_M * grid_n
    group_id = pid // pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % pid_in_group) % group_size_m
    pid_n = (pid % pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # X tile pointers: (BLOCK_M, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    # W^T tile pointers: need W^T[k, n] = W[n, k]
    # build pointers of shape (BLOCK_K, BLOCK_N)
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # Store
    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)

    if OUT_DTYPE == "fp16":
        tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)
    elif OUT_DTYPE == "bf16":
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=y_mask)
    else:
        tl.static_assert(False, "Unsupported OUT_DTYPE")


@triton.jit
def bias_grad_kernel(
    dy_ptr, db_ptr,
    M, N,
    stride_dym, stride_dyn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    db[n] = sum_m dy[m, n]
    dy: (M, N)
    db: (N,)
    """
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduce over M in chunks
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        dy_ptrs = dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(dy, axis=0)

    if OUT_DTYPE == "fp16":
        tl.store(db_ptr + offs_n, acc.to(tl.float16), mask=offs_n < N)
    elif OUT_DTYPE == "bf16":
        tl.store(db_ptr + offs_n, acc.to(tl.bfloat16), mask=offs_n < N)
    else:
        tl.static_assert(False, "Unsupported OUT_DTYPE")


def triton_linear_forward(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: (..., K)
    w: (N, K)  torch.nn.Linear weight layout
    b: (N,) or None
    returns: (..., N)
    """
    assert x.is_cuda and w.is_cuda, "This demo targets CUDA/HIP GPUs"
    assert x.dtype == w.dtype, "Keep x and w same dtype in this demo"
    out_dtype_str = _dtype_to_out_dtype_str(x.dtype)

    orig_shape = x.shape
    K = orig_shape[-1]
    x2d = x.reshape(-1, K)
    M = x2d.shape[0]
    N = w.shape[0]

    y2d = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    linear_fwd_kernel[grid](
        x2d, w, b if b is not None else y2d, y2d,
        M, N, K,
        x2d.stride(0), x2d.stride(1),
        w.stride(0), w.stride(1),
        y2d.stride(0), y2d.stride(1),
        HAS_BIAS=(b is not None),
        OUT_DTYPE=out_dtype_str,
    )

    return y2d.reshape(*orig_shape[:-1], N)


def triton_matmul_strided(
    a: torch.Tensor,
    b: torch.Tensor,
    M: int, N: int, K: int,
    stride_am: int, stride_ak: int,
    stride_bk: int, stride_bn: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out_dtype_str = _dtype_to_out_dtype_str(out_dtype)
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        c.stride(0), c.stride(1),
        OUT_DTYPE=out_dtype_str,
    )
    return c


def triton_bias_grad(dy2d: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    out_dtype_str = _dtype_to_out_dtype_str(out_dtype)
    M, N = dy2d.shape
    db = torch.empty((N,), device=dy2d.device, dtype=out_dtype)

    grid = (triton.cdiv(N, 1024),)
    bias_grad_kernel[grid](
        dy2d, db,
        M, N,
        dy2d.stride(0), dy2d.stride(1),
        OUT_DTYPE=out_dtype_str,
        BLOCK_N=1024,
        BLOCK_M=256,
        num_warps=8,
    )
    return db


class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        # For best performance, feed contiguous (demo choice)
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        y = triton_linear_forward(x, weight, bias)
        ctx.save_for_backward(x, weight, bias if bias is not None else torch.tensor([], device=x.device))
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w, b_or_empty = ctx.saved_tensors
        has_bias = ctx.has_bias
        b = b_or_empty if has_bias else None

        grad_out = grad_out.contiguous()

        # Flatten to 2D
        K = x.shape[-1]
        x2d = x.reshape(-1, K)                  # (M, K)
        M = x2d.shape[0]
        N = w.shape[0]
        dy2d = grad_out.reshape(-1, N)          # (M, N)

        dx = dw = db = None

        # dX = dY @ W   where dY: (M, N), W: (N, K) -> dX: (M, K)
        if ctx.needs_input_grad[0]:
            dx2d = triton_matmul_strided(
                a=dy2d, b=w,
                M=M, N=K, K=N,
                stride_am=dy2d.stride(0), stride_ak=dy2d.stride(1),
                stride_bk=w.stride(0), stride_bn=w.stride(1),
                out_dtype=x.dtype,
            )
            dx = dx2d.reshape_as(x)

        # dW = dY^T @ X   where dY^T: (N, M), X: (M, K) -> dW: (N, K)
        if ctx.needs_input_grad[1]:
            dw = triton_matmul_strided(
                a=dy2d, b=x2d,
                M=N, N=K, K=M,
                stride_am=dy2d.stride(1), stride_ak=dy2d.stride(0),   # treat dy2d as transposed
                stride_bk=x2d.stride(0), stride_bn=x2d.stride(1),
                out_dtype=w.dtype,
            )

        # db = sum_m dY[m, :]
        if has_bias and ctx.needs_input_grad[2]:
            db = triton_bias_grad(dy2d, out_dtype=b.dtype)

        return dx, dw, db


class TritonLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float16):
        super().__init__()
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
        return TritonLinearFunction.apply(x, self.weight, self.bias)

    @staticmethod
    def from_linear(m: nn.Linear) -> "TritonLinear":
        out = TritonLinear(
            m.in_features,
            m.out_features,
            bias=(m.bias is not None),
            device=m.weight.device,
            dtype=m.weight.dtype,
        )
        out.weight.data.copy_(m.weight.data)
        if m.bias is not None:
            out.bias.data.copy_(m.bias.data)
        return out


def replace_linears(module: nn.Module) -> nn.Module:
    """
    Recursively replace nn.Linear with TritonLinear (keeping weights/bias).
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, TritonLinear.from_linear(child))
        else:
            replace_linears(child)
    return module


# -----------------------------
# Correctness + benchmark
# -----------------------------
def _bench_ms(fn, iters=200, warmup=50) -> float:
    # Warmup (also triggers JIT + autotune)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def demo():
    assert torch.cuda.is_available(), "Need a CUDA/HIP GPU for this demo"

    torch.manual_seed(0)

    # Choose a representative Linear
    B = 4096
    K = 1024
    N = 1024
    dtype = torch.bfloat16
    device = torch.device("cuda")

    x = torch.randn((B, K), device=device, dtype=dtype, requires_grad=True)

    ref = nn.Linear(K, N, bias=True, device=device, dtype=dtype)
    tri = TritonLinear.from_linear(ref)

    # ---- Forward correctness
    with torch.no_grad():
        y_ref = ref(x)
        y_tri = tri(x)
        max_abs = (y_ref - y_tri).abs().max().item()
        print(f"[forward] max |ref-triton| = {max_abs:.4e}")

    # ---- Backward correctness (compare grads)
    def run_ref():
        ref.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        y = ref(x)
        loss = (y * y).mean()
        loss.backward()

    def run_tri():
        tri.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        y = tri(x)
        loss = (y * y).mean()
        loss.backward()

    run_ref()
    gx_ref = x.grad.detach().clone()
    gw_ref = ref.weight.grad.detach().clone()
    gb_ref = ref.bias.grad.detach().clone()

    run_tri()
    gx_tri = x.grad.detach().clone()
    gw_tri = tri.weight.grad.detach().clone()
    gb_tri = tri.bias.grad.detach().clone()

    print(f"[grad x] max |ref-triton| = {(gx_ref - gx_tri).abs().max().item():.4e}")
    print(f"[grad w] max |ref-triton| = {(gw_ref - gw_tri).abs().max().item():.4e}")
    print(f"[grad b] max |ref-triton| = {(gb_ref - gb_tri).abs().max().item():.4e}")

    # ---- Benchmark (forward only)
    ref_fwd_ms = _bench_ms(lambda: ref(x))
    tri_fwd_ms = _bench_ms(lambda: tri(x))

    # ---- Benchmark (forward+backward)
    ref_bwd_ms = _bench_ms(run_ref, iters=100, warmup=20)
    tri_bwd_ms = _bench_ms(run_tri, iters=100, warmup=20)

    flops_fwd = 2 * B * N * K  # GEMM flops
    tflops_ref_fwd = flops_fwd / (ref_fwd_ms * 1e-3) / 1e12
    tflops_tri_fwd = flops_fwd / (tri_fwd_ms * 1e-3) / 1e12

    print(f"[bench fwd] torch:  {ref_fwd_ms:.3f} ms  ({tflops_ref_fwd:.2f} TFLOP/s)")
    print(f"[bench fwd] triton: {tri_fwd_ms:.3f} ms  ({tflops_tri_fwd:.2f} TFLOP/s)")
    print(f"[bench bwd] torch:  {ref_bwd_ms:.3f} ms (fwd+loss+bwd)")
    print(f"[bench bwd] triton: {tri_bwd_ms:.3f} ms (fwd+loss+bwd)")


if __name__ == "__main__":
    demo()
