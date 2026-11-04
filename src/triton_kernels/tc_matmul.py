"""Tensor-core matmuls with a hardware low-precision accumulator (real arithmetic, not emulation).

``tc_matmul(a, b, acc="fp16")`` converts the operands to fp16 and runs ``tl.dot`` with an fp16
accumulator (HMMA with f16 C/D registers): products, the in-MMA reduction and the running sum
over K tiles are all done by the tensor cores, in their own order and with their own rounding.
``acc="fp32"`` is the same kernel with fp32 accumulators, which isolates the accumulator effect
(same fp16 operands). ``in_dtype="fp8"`` uses e4m3 operands (fp32 accumulation only).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _tc_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
               stride_ab, stride_am, stride_ak, stride_bb, stride_bk, stride_bn, stride_cb, stride_cm, stride_cn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_FP16: tl.constexpr,
               PROMOTE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    a_ptr += pid_b * stride_ab
    b_ptr += pid_b * stride_bb
    c_ptr += pid_b * stride_cb
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    if ACC_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + kk[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(kk[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        if ACC_FP16:
            acc = tl.dot(a, b, acc, out_dtype=tl.float16)
        elif PROMOTE:
            # DeepSeek-V3-style promotion: each K block starts from a fresh tensor-core accumulator,
            # the block result is added to the running sum by an fp32 add on the CUDA cores
            # (identity asm keeps the compiler from fusing the add back into the MMA accumulator)
            part = tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [tl.dot(a, b)], dtype=tl.float32,
                                             is_pure=True, pack=1)
            acc += part
        else:
            acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


E4M3_MAX = 448.0


def quantize_fp8(x: torch.Tensor, dim: int):
    """e4m3 quantization with one scale per row (dim=-1) or per column (dim=-2); returns (q, scale)."""
    amax = x.float().abs().amax(dim=dim, keepdim=True).clamp_min(1e-12)
    scale = amax / E4M3_MAX
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def fp8_matmul(a: torch.Tensor, b: torch.Tensor, mode: str = "tc", promote_k: int = 128) -> torch.Tensor:
    """FP8 (e4m3, per-row/per-column scales) matmul, result in bf16.

    mode="tc": tensor-core accumulation (limited alignment precision of the hardware);
    "promote": tensor-core partial sums over ``promote_k`` then fp32 adds (DeepSeek-V3 style);
    "exact": the same quantized operands multiplied in fp32 by cuBLAS (reference accumulation).
    Scales are applied after the reduction, so all three differ only in how the sum is accumulated.
    """
    qa, sa = quantize_fp8(a, -1)
    qb, sb = quantize_fp8(b, -2)
    if mode == "exact":
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            y = qa.float() @ qb.float()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    else:
        y = tc_matmul(qa, qb, acc="fp32", in_dtype="fp8", out_dtype=torch.float32,
                      block=(128, 128, promote_k if mode == "promote" else 32), promote=mode == "promote")
    return (y * sa * sb).to(torch.bfloat16)


def tc_matmul(a: torch.Tensor, b: torch.Tensor, acc: str = "fp16", in_dtype: str = "fp16",
              out_dtype: torch.dtype = torch.bfloat16, block=(128, 128, 32), promote: bool = False) -> torch.Tensor:
    """(..., M, K) @ (..., K, N) on tensor cores with an fp16 or fp32 accumulator (no scaling)."""
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[in_dtype]
    if acc == "fp16" and in_dtype != "fp16":
        raise ValueError("fp16 accumulation requires fp16 operands")
    M, K = a.shape[-2:]
    N = b.shape[-1]
    batch = a.shape[:-2]
    nb = math.prod(batch) if batch else 1
    a3 = a.reshape(nb, M, K).to(dt)
    b3 = b.reshape(nb, K, N).to(dt)
    if in_dtype == "fp8":  # e4m3 dot wants K-major B on sm_89
        b3 = b3.transpose(1, 2).contiguous().transpose(1, 2)
    c3 = torch.empty((nb, M, N), device=a.device, dtype=out_dtype)
    bm, bn, bk = block
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn), nb)
    _tc_kernel[grid](a3, b3, c3, M, N, K, a3.stride(0), a3.stride(1), a3.stride(2), b3.stride(0), b3.stride(1),
                     b3.stride(2), c3.stride(0), c3.stride(1), c3.stride(2),
                     BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, ACC_FP16=acc == "fp16", PROMOTE=promote,
                     num_warps=8, num_stages=3 if bk <= 64 else 2)
    return c3.reshape(*batch, M, N)
