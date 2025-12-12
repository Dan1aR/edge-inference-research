import math
from typing import Optional

import torch
import torch.nn as nn

import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# BF16-acc matmul / bmm (C = A @ B) with true bf16 accumulation (slow, research-use)
# Supports:
#   A: (M, K),     B: (K, N)       -> C: (M, N)
#   A: (B, M, K),  B: (B, K, N)    -> C: (B, M, N)
# -----------------------------------------------------------------------------

BF16ACC_MATMUL_CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 16, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 16, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=BF16ACC_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def bf16acc_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,   # A: (B,M,K) or (M,K) with B=1
    stride_bb, stride_bk, stride_bn,   # B: (B,K,N) or (K,N) with B=1
    stride_cb, stride_cm, stride_cn,   # C: (B,M,N) or (M,N) with B=1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    C[b, m, n] = sum_k A[b, m, k] * B[b, k, n]
    Accumulation is performed in bf16 (true bf16 accumulator).
    """

    pid = tl.program_id(axis=0)     # tile id over (m,n)
    pid_b = tl.program_id(axis=1)   # batch id

    # Base pointers for this batch
    a_ptr_b = a_ptr + pid_b * stride_ab
    b_ptr_b = b_ptr + pid_b * stride_bb
    c_ptr_b = c_ptr + pid_b * stride_cb

    # Grouped ordering for L2 reuse (same scheme as Triton matmul tutorial)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_in_group = GROUP_M * grid_n
    group_id = pid // pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % pid_in_group) % group_size_m
    pid_n = (pid % pid_in_group) // group_size_m

    # Raw offsets (used for masks/stores)
    offs_m_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Modulo offsets (used for safe in-bounds pointer arithmetic even when masked)
    offs_m = offs_m_raw % M
    offs_n = offs_n_raw % N

    # bf16 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)

    # Loop over K tiles
    k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in range(0, k_tiles):
        k_base = kt * BLOCK_K

        # Unroll within tile
        for kk in tl.static_range(0, BLOCK_K):
            k = k_base + kk
            k_in = k < K

            # Load A[:, k] as vector (BLOCK_M,)
            a_vec = tl.load(
                a_ptr_b + offs_m * stride_am + k * stride_ak,
                mask=(offs_m_raw < M) & k_in,
                other=0,
            ).to(tl.bfloat16)

            # Load B[k, :] as vector (BLOCK_N,)
            b_vec = tl.load(
                b_ptr_b + k * stride_bk + offs_n * stride_bn,
                mask=(offs_n_raw < N) & k_in,
                other=0,
            ).to(tl.bfloat16)

            # bf16 MAC update
            acc += a_vec[:, None] * b_vec[None, :]

    # Store C
    c_ptrs = c_ptr_b + offs_m_raw[:, None] * stride_cm + offs_n_raw[None, :] * stride_cn
    c_mask = (offs_m_raw[:, None] < M) & (offs_n_raw[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_bf16acc_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    BF16-only matmul/bmm with true bf16 accumulation.

    Supports:
      - (M, K) @ (K, N) -> (M, N)
      - (..., M, K) @ (..., K, N) -> (..., M, N)   where "..." batch dims must match exactly

    Notes:
      - Very slow vs tl.dot / tensor cores (research / numerical study use).
      - Accumulation is performed in bf16 (per-MAC rounding behavior).
      - Non-contiguous inputs are materialized to contiguous (e.g., transpose(-1, -2)).
    """
    if not (a.is_cuda and b.is_cuda):
        raise RuntimeError("triton_bf16acc_matmul: CUDA/HIP device required.")
    if a.device != b.device:
        raise RuntimeError(f"triton_bf16acc_matmul: device mismatch a={a.device}, b={b.device}")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"triton_bf16acc_matmul: bf16-only, got a={a.dtype}, b={b.dtype}")
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError(f"triton_bf16acc_matmul: expected dim>=2, got a.dim={a.dim()}, b.dim={b.dim()}")
    if a.dim() != b.dim():
        raise ValueError(f"triton_bf16acc_matmul: expected same rank, got a.dim={a.dim()}, b.dim={b.dim()}")

    # Matmul dims
    M, K = a.shape[-2], a.shape[-1]
    K2, N = b.shape[-2], b.shape[-1]
    if K != K2:
        raise ValueError(f"Shape mismatch: a(...,{M},{K}) but b(...,{K2},{N})")

    # Batch dims must match exactly (no broadcasting in this wrapper)
    batch_shape = a.shape[:-2]
    if batch_shape != b.shape[:-2]:
        raise ValueError(f"Batch shape mismatch: a batch={batch_shape} vs b batch={b.shape[:-2]}")

    # 2D fast-path (still launches 2D grid with B=1)
    if len(batch_shape) == 0:
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
            1,
        )
        bf16acc_matmul_kernel[grid](
            a, b, c,
            1, M, N, K,
            0, a.stride(0), a.stride(1),
            0, b.stride(0), b.stride(1),
            0, c.stride(0), c.stride(1),
        )
        return c

    # N-D batch case (3D, 4D, ...)
    Bsz = math.prod(batch_shape)

    # Make contiguous because attention often passes transposed views
    a3 = a.contiguous().reshape(Bsz, M, K)
    b3 = b.contiguous().reshape(Bsz, K, N)

    c3 = torch.empty((Bsz, M, N), device=a.device, dtype=torch.bfloat16)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        Bsz,
    )

    bf16acc_matmul_kernel[grid](
        a3, b3, c3,
        Bsz, M, N, K,
        a3.stride(0), a3.stride(1), a3.stride(2),
        b3.stride(0), b3.stride(1), b3.stride(2),
        c3.stride(0), c3.stride(1), c3.stride(2),
    )

    return c3.reshape(*batch_shape, M, N)
