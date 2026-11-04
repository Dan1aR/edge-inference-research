"""Unified Triton kernel for matmul with a low-precision accumulator.

C[b, m, n] = sum_k A[b, m, k] * B[b, k, n] with bf16 operands, where the running sum is
kept in an accumulator format ``fmt``:

* ``"bf16"`` (e8m7, default), ``"e8m10"``, ``"e8m13"``, ``"e8m16"`` - fp32 exponent range,
  the given number of explicit mantissa bits (``e8m13`` ~ the ~14-bit accumulation reported
  for FP8 tensor cores);
* ``"fp16"`` - IEEE half (e5m10, subnormals, overflow to inf);
* ``"fp32"`` - plain fp32 running sum (sequential order, unlike cuBLAS).

The arithmetic of one multiply-accumulate step is fixed explicitly in the kernel
(instead of being left to the compiler):

* ``round_product=False`` ("fma" semantics): the product of two bf16 numbers is
  exact in fp32, the sum ``s + a*b`` is formed in fp32 and rounded to ``fmt``:
  ``s <- fmt(fp32(s + a*b))``.
* ``round_product=True``: the product is rounded to bf16 first:
  ``s <- fmt(fp32(s + bf16(a*b)))``.
* ``rounding="rne"`` (round to nearest even) or ``"sr"`` (stochastic rounding of the sum,
  e8 formats only; random bits from Philox, seeded per call).
* ``kahan=True``: Kahan-compensated summation, the compensation term is kept in ``fmt``
  too: ``y = fmt(p - c); t = fmt(s + y); c = fmt(fmt(t - s) - y); s = t``.

Reduction order over K:

* ``chunk=0``: strictly sequential, k = 0 .. K-1 (one running sum per output).
* ``chunk=C``: K is split into consecutive chunks of length C; each chunk is
  accumulated sequentially into a fresh partial sum, partial sums are merged
  sequentially either in the accumulator format (``chunk_merge="acc"``; ``"bf16"`` is a
  legacy alias) or in fp32 with a single final rounding (``chunk_merge="fp32"``).

The result is returned in bf16 (one final RNE rounding, a no-op for ``fmt="bf16"``), as the
output of a bf16 matmul under autocast.

The pure-PyTorch reference with identical semantics is
:func:`src.bf16_accum.bf16_accum_matmul_ref`.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

FORMATS = {"bf16": 7, "e8m10": 10, "e8m13": 13, "e8m16": 16, "fp32": 23, "fp16": 10}
# real tensor-core arithmetic (not emulated): name -> (operand dtype, accumulator dtype)
TC_FORMATS = {"tc_fp16acc": ("fp16", "fp16"), "tc_fp16": ("fp16", "fp32"), "tc_bf16": ("bf16", "fp32"),
              "fp8_tc": ("fp8", "tc"), "fp8_promote": ("fp8", "promote"), "fp8_exact": ("fp8", "exact")}
CHUNKS = (8, 16, 32, 64, 128, 256)


@dataclass
class AccumConfig:
    round_product: bool = False
    chunk: int = 0
    chunk_merge: str = "bf16"  # "acc" (= legacy "bf16") | "fp32"
    fmt: str = "bf16"
    rounding: str = "rne"  # "rne" | "sr"
    kahan: bool = False

    def validate(self) -> "AccumConfig":
        if self.fmt in TC_FORMATS:
            if self.chunk or self.round_product or self.kahan or self.rounding != "rne":
                raise ValueError("tensor-core formats use the hardware order/rounding only")
            return self
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown accumulator format {self.fmt}")
        if self.chunk_merge not in ("bf16", "acc", "fp32"):
            raise ValueError(f"chunk_merge must be acc|bf16|fp32, got {self.chunk_merge}")
        if self.chunk and self.chunk not in CHUNKS:
            raise ValueError(f"unsupported chunk {self.chunk}")
        if self.rounding not in ("rne", "sr"):
            raise ValueError(f"unknown rounding {self.rounding}")
        if self.rounding == "sr" and self.fmt in ("fp16", "fp32"):
            raise ValueError("stochastic rounding is implemented for e8 formats only")
        if self.kahan and (self.chunk or self.rounding == "sr"):
            raise ValueError("kahan is only supported with sequential RNE accumulation")
        return self

    def describe(self) -> str:
        """Legacy strings for the default bf16/RNE configurations (used in stored results)."""
        sem = "round_product" if self.round_product else "fma"
        prefix = "" if self.fmt == "bf16" else f"{self.fmt},"
        if self.rounding == "sr":
            prefix += "sr,"
        if self.kahan:
            prefix += "kahan,"
        if self.chunk == 0:
            return f"{prefix}{sem},sequential"
        merge = "bf16" if self.chunk_merge in ("bf16", "acc") and self.fmt == "bf16" else self.chunk_merge
        return f"{prefix}{sem},chunk{self.chunk},merge_{merge}"


ACCUM_CONFIG = AccumConfig()
_SR_CALLS = itertools.count()


def set_accum_config(round_product: bool = False, chunk: int = 0, chunk_merge: str = "bf16", fmt: str = "bf16",
                     rounding: str = "rne", kahan: bool = False) -> AccumConfig:
    cfg = AccumConfig(round_product, chunk, chunk_merge, fmt, rounding, kahan).validate()
    for k, v in vars(cfg).items():
        setattr(ACCUM_CONFIG, k, v)
    return ACCUM_CONFIG


@triton.jit
def _round_to_bf16_rne(x):
    # fp32 -> bf16 -> fp32 with round-to-nearest-even, done on the bit pattern so that the
    # compiler cannot fold the fptrunc/fpext pair away (finite inputs only).
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _round_fmt(x, FMT_KIND: tl.constexpr, MAN: tl.constexpr, SR: tl.constexpr, rbits):
    """Round fp32 ``x`` to the accumulator format, result kept in fp32.

    FMT_KIND: 0 = e8 format with MAN explicit mantissa bits (MAN < 23), 1 = fp32, 2 = IEEE fp16.
    """
    if FMT_KIND == 1:
        return x
    elif FMT_KIND == 2:
        h = tl.inline_asm_elementwise("cvt.rn.f16.f32 $0, $1;", "=h,r", [x], dtype=tl.float16, is_pure=True, pack=1)
        return h.to(tl.float32)
    else:
        SHIFT: tl.constexpr = 23 - MAN
        LOW: tl.constexpr = (1 << SHIFT) - 1
        HIGH: tl.constexpr = 0xFFFFFFFF ^ LOW
        bits = x.to(tl.uint32, bitcast=True)
        if SR:
            bits = (bits + (rbits & LOW)) & HIGH
        else:
            bits = (bits + (LOW >> 1) + ((bits >> SHIFT) & 1)) & HIGH
        return bits.to(tl.float32, bitcast=True)


@triton.jit
def _accmatmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K, seed, M_BUCKET,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ROUND_PRODUCT: tl.constexpr,
    CHUNK: tl.constexpr,  # 0 sequential, else chunk length (multiple of BLOCK_K)
    MERGE_FP32: tl.constexpr,
    FMT_KIND: tl.constexpr, MAN: tl.constexpr,
    SR: tl.constexpr, KAHAN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    a_ptr += pid_b * stride_ab
    b_ptr += pid_b * stride_bb
    c_ptr += pid_b * stride_cb

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_in_group = GROUP_M * grid_n
    group_id = pid // pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % pid_in_group) % group_size_m
    pid_n = (pid % pid_in_group) // group_size_m

    offs_m_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = offs_m_raw % M
    offs_n = offs_n_raw % N
    mask_m = offs_m_raw < M
    mask_n = offs_n_raw < N

    # all running values are kept in fp32 registers, already rounded to the accumulator format
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    comp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rbits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.uint32)
    if SR:
        # unique stream per output element and batch; one Philox draw per k step
        rng_off = ((pid_b * M + offs_m_raw[:, None]) * N + offs_n_raw[None, :]).to(tl.uint32)

    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        if CHUNK > 0:
            if (kt > 0) & ((kt * BLOCK_K) % CHUNK == 0):
                if MERGE_FP32:
                    acc += part
                else:
                    acc = _round_fmt(acc + part, FMT_KIND, MAN, False, rbits)
                part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in tl.static_range(0, BLOCK_K):
            k = kt * BLOCK_K + kk
            k_in = k < K
            a_vec = tl.load(a_ptr + offs_m * stride_am + k * stride_ak, mask=mask_m & k_in, other=0.0)
            b_vec = tl.load(b_ptr + k * stride_bk + offs_n * stride_bn, mask=mask_n & k_in, other=0.0)
            # bf16 x bf16 is exact in fp32
            prod = a_vec.to(tl.bfloat16).to(tl.float32)[:, None] * b_vec.to(tl.bfloat16).to(tl.float32)[None, :]
            if ROUND_PRODUCT:
                prod = _round_to_bf16_rne(prod)
            if SR:
                rbits = tl.randint(seed + k, rng_off)
            if KAHAN:
                y = _round_fmt(prod - comp, FMT_KIND, MAN, False, rbits)
                t = _round_fmt(part + y, FMT_KIND, MAN, False, rbits)
                c_new = _round_fmt(_round_fmt(t - part, FMT_KIND, MAN, False, rbits) - y, FMT_KIND, MAN, False, rbits)
                # padded steps beyond K must not apply the pending compensation
                comp = tl.where(k_in, c_new, comp)
                part = tl.where(k_in, t, part)
            else:
                part = _round_fmt(part + prod, FMT_KIND, MAN, SR, rbits)

    if CHUNK > 0:
        if MERGE_FP32:
            acc += part
        else:
            acc = _round_fmt(acc + part, FMT_KIND, MAN, False, rbits)
    else:
        acc = part

    c_ptrs = c_ptr + offs_m_raw[:, None] * stride_cm + offs_n_raw[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


_TUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
]

# Sequential mode: BLOCK_K only controls unrolling, results do not depend on it.
_accmatmul_tuned = triton.autotune(
    configs=_TUNE_CONFIGS, key=["M_BUCKET", "N", "K", "ROUND_PRODUCT", "FMT_KIND", "MAN", "SR", "KAHAN"]
)(_accmatmul_kernel)


def bf16acc_matmul(a: torch.Tensor, b: torch.Tensor, cfg: AccumConfig | None = None,
                   block_k: int | None = None) -> torch.Tensor:
    """(..., M, K) @ (..., K, N) -> (..., M, N) in bf16 with a low-precision accumulator.

    Batch dimensions of ``a`` and ``b`` must match exactly. Arbitrary strides are
    supported (no copies for transposed 2D views). ``block_k`` forces a specific
    unroll length in sequential mode (used by tests to show BLOCK_K independence).
    """
    cfg = cfg or ACCUM_CONFIG
    cfg.validate()
    if not (a.is_cuda and b.is_cuda):
        raise RuntimeError("bf16acc_matmul requires CUDA tensors")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"bf16 inputs required, got {a.dtype}, {b.dtype}")
    M, K = a.shape[-2:]
    K2, N = b.shape[-2:]
    if K != K2:
        raise ValueError(f"shape mismatch {tuple(a.shape)} @ {tuple(b.shape)}")
    batch_shape = a.shape[:-2]
    if batch_shape != b.shape[:-2]:
        raise ValueError(f"batch mismatch {tuple(a.shape)} @ {tuple(b.shape)}")

    if cfg.fmt in TC_FORMATS:
        from .tc_matmul import fp8_matmul, tc_matmul

        in_dt, acc_dt = TC_FORMATS[cfg.fmt]
        if in_dt == "fp8":
            return fp8_matmul(a, b, mode=acc_dt)
        return tc_matmul(a, b, acc=acc_dt, in_dtype=in_dt)
    nb = math.prod(batch_shape) if batch_shape else 1
    a3 = a.reshape(nb, M, K)
    b3 = b.reshape(nb, K, N)
    c3 = torch.empty((nb, M, N), device=a.device, dtype=torch.bfloat16)
    for t in (a3, b3, c3):  # the kernel uses int32 offsets
        if max((d - 1) * st for d, st in zip(t.shape, t.stride())) >= 2**31:
            raise ValueError("tensor too large for int32 offsets; reduce the batch size")
    sr = cfg.rounding == "sr"
    seed = (next(_SR_CALLS) * 1_000_003) % (2**31 - 2**20) if sr else 0
    fmt_kind = {"fp32": 1, "fp16": 2}.get(cfg.fmt, 0)
    args = (
        a3, b3, c3, M, N, K, seed, triton.next_power_of_2(M * nb),  # bucketed autotune key: no re-tuning per shape
        a3.stride(0), a3.stride(1), a3.stride(2),
        b3.stride(0), b3.stride(1), b3.stride(2),
        c3.stride(0), c3.stride(1), c3.stride(2),
    )
    flags = dict(ROUND_PRODUCT=cfg.round_product, FMT_KIND=fmt_kind, MAN=FORMATS[cfg.fmt], SR=sr, KAHAN=cfg.kahan)
    if cfg.chunk == 0 and block_k is None:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), nb)
        _accmatmul_tuned[grid](*args, CHUNK=0, MERGE_FP32=False, **flags)
    else:
        bk = min(cfg.chunk, 16) if cfg.chunk else block_k
        grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), nb)
        _accmatmul_kernel[grid](
            *args, BLOCK_M=64, BLOCK_N=64, BLOCK_K=bk, GROUP_M=8, CHUNK=cfg.chunk,
            MERGE_FP32=cfg.chunk_merge == "fp32", num_warps=4, num_stages=2, **flags,
        )
    return c3.reshape(*batch_shape, M, N)


class BF16AccMatmulSTE(torch.autograd.Function):
    """Forward: low-precision-accumulating matmul. Backward: gradients of the exact matmul.

    For C = A @ B the backward returns dA = dC @ B^T and dB = A^T @ dC computed with
    standard (FP32-accumulating) kernels on the same bf16 operands, i.e. the Jacobian
    of the accumulator rounding is replaced by identity (straight-through estimator).
    Equivalent formulation: C = A@B + stop_gradient(BF16acc(A,B) - A@B).
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return bf16acc_matmul(a, b)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_out = grad_out.to(a.dtype)
        grad_a = torch.matmul(grad_out, b.transpose(-1, -2)) if ctx.needs_input_grad[0] else None
        grad_b = torch.matmul(a.transpose(-1, -2), grad_out) if ctx.needs_input_grad[1] else None
        return grad_a, grad_b


def bf16acc_matmul_ste(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return BF16AccMatmulSTE.apply(a, b)
