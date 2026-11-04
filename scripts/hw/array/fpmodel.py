"""E7: bit-exact numpy model of the accumulator variants of scripts/hw/array/rtl/acc_array.v.

One MAC step (a, b bf16; product exact):  s <- RNE_fmt(s + a*b), a single correct rounding with IEEE
subnormals, overflow to +-Inf and IEEE signed-zero rules.  Values are carried as float64 (exact for all
formats used here); a rounding of an exact sum x+y is done from the float64 TwoSum pair (sum, err), so
it is correct even when the float64 sum itself is inexact.

Variants (names used everywhere in scripts/hw/array/ and results/hw/array/):
  fp32   VAR 0   e8m23 accumulator, product rounded to fp32 first (exact unless outside the fp32 range)
  bf16   VAR 1   e8m7, fused (single rounding of s + exact product)
  fp16   VAR 2   e5m10 (IEEE half range), fused
  e8m13  VAR 3   e8m13, fused
  kahan  VAR 4   bf16 s and bf16 compensation c:  y=RN(ab-c), t=RN(s+y), z=RN(t-s), c=RN(z-y), s=t
  chunk  VAR 5   bf16 chain over C = R consecutive k (fresh +0 per chunk), acc32 <- RN32(acc32 + chunk)
  *_sep  VAR 11-13  product first rounded to (8,16) (= exact for normal-range products), then one rounding

Also provided: the *software* semantics of src/bf16_accum.py::bf16_accum_matmul_ref (every step is
computed in fp32 and then rounded to the accumulator format, i.e. a double rounding), to count where the
single-rounding hardware differs from the training/eval emulation.
"""
from __future__ import annotations

import numpy as np

FMT = {  # name: (exponent bits, fraction bits)
    "bf16": (8, 7), "fp16": (5, 10), "e8m13": (8, 13), "fp32": (8, 23), "p16": (8, 15),
}
VAR_ID = {"fp32": 0, "bf16": 1, "fp16": 2, "e8m13": 3, "kahan": 4, "chunk": 5,
          "bf16_sep": 11, "fp16_sep": 12, "e8m13_sep": 13}
ACC_FMT = {"fp32": "fp32", "bf16": "bf16", "fp16": "fp16", "e8m13": "e8m13", "kahan": "bf16",
           "chunk": "bf16", "bf16_sep": "bf16", "fp16_sep": "fp16", "e8m13_sep": "e8m13"}


# ------------------------------------------------------------------------------------------------
# IEEE bit patterns <-> float64
# ------------------------------------------------------------------------------------------------
def decode(bits, fmt: str) -> np.ndarray:
    E, M = FMT[fmt]
    b = np.asarray(bits, dtype=np.int64)
    sign = (b >> (E + M)) & 1
    e = (b >> M) & ((1 << E) - 1)
    f = b & ((1 << M) - 1)
    bias = (1 << (E - 1)) - 1
    mag = np.where(e == 0, np.ldexp(f.astype(np.float64), 1 - bias - M),
                   np.ldexp((f + (1 << M)).astype(np.float64), np.maximum(e, 1) - bias - M))
    mag = np.where(e == (1 << E) - 1, np.where(f == 0, np.inf, np.nan), mag)
    return np.where(sign == 1, -mag, mag)


def encode(x, fmt: str) -> np.ndarray:
    """float64 values that are exactly representable in fmt -> bit patterns (NaN -> canonical qNaN)."""
    E, M = FMT[fmt]
    x = np.asarray(x, dtype=np.float64)
    bias = (1 << (E - 1)) - 1
    sign = np.signbit(x).astype(np.int64)
    a = np.abs(x)
    m, ex = np.frexp(a)                     # a = m * 2**ex, 0.5 <= m < 1
    e = ex - 1 + bias                        # biased exponent if normal
    normal = e >= 1
    with np.errstate(invalid="ignore", over="ignore"):
        fr_n = np.ldexp(m, M + 1) - (1 << M)
        fr_s = np.ldexp(a, bias - 1 + M)
    frac = np.where(normal, fr_n, fr_s)
    frac = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
    frac = np.where(a == 0, 0, frac)
    e = np.where(normal & (a != 0), e, 0)
    out = (sign << (E + M)) | (e << M) | frac
    out = np.where(np.isinf(x), (sign << (E + M)) | (((1 << E) - 1) << M), out)
    out = np.where(np.isnan(x), (((1 << E) - 1) << M) | (1 << (M - 1)), out)
    return out.astype(np.int64)


def bf16_from_f32(x) -> np.ndarray:
    """fp32 values -> bf16 bit patterns, RNE (operand quantisation as torch .to(bfloat16))."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


# ------------------------------------------------------------------------------------------------
# correct rounding
# ------------------------------------------------------------------------------------------------
def two_sum(a, b):
    with np.errstate(invalid="ignore"):
        s = a + b
        bb = s - a
        err = (a - (s - bb)) + (b - bb)
    err = np.where(np.isfinite(s), err, 0.0)
    return s, err


def round_fmt(x, fmt: str, err=None) -> np.ndarray:
    """RNE of the exact value x+err (|err| <= ulp64(x)/2) to fmt; IEEE subnormals, +-Inf on overflow."""
    E, M = FMT[fmt]
    x = np.asarray(x, dtype=np.float64)
    if err is None:
        err = np.zeros_like(x)
    bias = (1 << (E - 1)) - 1
    emin = 1 - bias
    _, ex = np.frexp(x)
    qe = np.maximum(ex - 1, emin) - M        # exponent of the quantum (ulp) at x
    with np.errstate(invalid="ignore", over="ignore"):
        y = np.ldexp(x, -qe)
        r = np.rint(y)
        fl = np.floor(y)
        tie = (y - fl == 0.5) & (err != 0)
        r = np.where(tie, fl + (err > 0), r)
        res = np.ldexp(r, qe)
    res = np.where(res == 0, np.copysign(0.0, x), res)
    maxf = np.ldexp(float((1 << (M + 1)) - 1), bias - M)
    res = np.where(np.abs(res) > maxf, np.copysign(np.inf, x), res)
    return np.where(np.isfinite(x), res, x)


def rnd_sum(a, b, fmt: str):
    s, e = two_sum(a, b)
    return round_fmt(s, fmt, e)


# ------------------------------------------------------------------------------------------------
# MAC steps (hardware semantics) -- s, c are float64 values in the accumulator format
# ------------------------------------------------------------------------------------------------
def mac_step(var: str, s, p, c=None):
    """One step for exact product p (float64).  Returns s' (and c' for kahan)."""
    if var == "fp32":
        return rnd_sum(s, round_fmt(p, "fp32"), "fp32")
    if var in ("bf16", "fp16", "e8m13", "chunk"):
        return rnd_sum(s, p, ACC_FMT[var])
    if var.endswith("_sep"):
        return rnd_sum(s, round_fmt(p, "p16"), ACC_FMT[var])
    if var == "kahan":
        y = rnd_sum(p, -c, "bf16")
        t = rnd_sum(s, y, "bf16")
        z = rnd_sum(t, -s, "bf16")
        return t, rnd_sum(z, -y, "bf16")
    raise KeyError(var)


def mac_step_sw(var: str, s, p, c=None):
    """Software emulation semantics (src/bf16_accum.py: fp32 op, then round to the acc format)."""
    f32 = lambda v: np.asarray(v, dtype=np.float32).astype(np.float64)  # noqa: E731
    pf = f32(p)
    if var == "fp32":
        return f32(f32(s) + pf)
    fmt = ACC_FMT[var]
    if var == "kahan":
        y = round_fmt(f32(pf - c), fmt)
        t = round_fmt(f32(s + y), fmt)
        z = round_fmt(f32(t - s), fmt)
        return t, round_fmt(f32(z - y), fmt)
    return round_fmt(f32(s + pf), fmt)


# ------------------------------------------------------------------------------------------------
# weight-stationary array model (R rows = one K-tile, sequential over K-tiles)
# ------------------------------------------------------------------------------------------------
def ws_matmul(var: str, A, W, R: int, record=None, sw: bool = False):
    """A: (M,K) float64 bf16 values, W: (K,N).  Returns the array result (float64) as the hardware
    computes it with K split into tiles of R rows (zero padding of the last tile).
    For 'chunk' the result is the fp32 edge accumulator.  record(kt, r, s_in, c_in) is called with the
    psum entering row r of K-tile kt (shape (M,N)) when given."""
    M, K = A.shape
    N = W.shape[1]
    step = mac_step_sw if sw else mac_step
    s = np.zeros((M, N))
    c = np.zeros((M, N))
    acc32 = np.zeros((M, N))
    nkt = -(-K // R)
    for kt in range(nkt):
        if var == "chunk":
            s = np.zeros((M, N))
        for r in range(R):
            k = kt * R + r
            if record is not None:
                record(kt, r, s, c)
            if k < K:
                p = A[:, k:k + 1] * W[k:k + 1, :]
            else:
                p = np.zeros((M, N))            # zero padding (a = +0, w = +0)
            if var == "kahan":
                s, c = step(var, s, p, c)
            else:
                s = step(var, s, p)
        if var == "chunk":
            acc32 = rnd_sum(acc32, s, "fp32") if not sw else np.asarray(
                np.float32(acc32) + np.float32(s), dtype=np.float64)
    if record is not None:
        record(nkt, 0, s, c)
    out = acc32 if var == "chunk" else s
    return (out, c) if var == "kahan" else out


def seq_matmul(var: str, A, W, chunk: int = 0, sw: bool = False):
    """Plain sequential reference (no padding); chunk>0 = chunked with fp32 merge."""
    M, K = A.shape
    step = mac_step_sw if sw else mac_step
    N = W.shape[1]
    tot = np.zeros((M, N))
    stepk = chunk or K
    for k0 in range(0, K, stepk):
        s = np.zeros((M, N))
        c = np.zeros((M, N))
        for k in range(k0, min(K, k0 + stepk)):
            p = A[:, k:k + 1] * W[k:k + 1, :]
            if var == "kahan":
                s, c = step(var, s, p, c)
            else:
                s = step("bf16" if var == "chunk" else var, s, p)
        if chunk:
            tot = rnd_sum(tot, s, "fp32") if not sw else (np.float32(tot) + np.float32(s)).astype(np.float64)
        else:
            tot = s
    return tot


# ------------------------------------------------------------------------------------------------
# HardFloat recoded format (fNToRecFN / recFNToFN of HardFloat Release 1), integer bit patterns
# ------------------------------------------------------------------------------------------------
def _clog2(n):
    return int(np.ceil(np.log2(n))) if n > 1 else 0


def fn_to_rec(bits, E: int, S: int) -> np.ndarray:
    """IEEE (E, S-1 fraction bits) -> recoded (E+S+1 bits), bit-exact port of fNToRecFN.v."""
    b = np.asarray(bits, dtype=np.int64)
    F = S - 1
    sign = (b >> (E + F)) & 1
    expIn = (b >> F) & ((1 << E) - 1)
    fract = b & ((1 << F) - 1)
    zE = expIn == 0
    zF = fract == 0
    # leading zeros of the F-bit fraction (countLeadingZeros, width normDistWidth)
    ndw = _clog2(S)
    lz = np.full(b.shape, F, dtype=np.int64)
    for i in range(F - 1, -1, -1):
        hit = ((fract >> i) & 1).astype(bool) & (lz == F)
        lz = np.where(hit, F - 1 - i, lz)
    normDist = lz & ((1 << ndw) - 1)
    subFract = ((fract << normDist) << 1) & ((1 << F) - 1)
    mask_e1 = (1 << (E + 1)) - 1
    adj = (np.where(zE, normDist ^ mask_e1, expIn) + ((1 << (E - 1)) | np.where(zE, 2, 1))) & mask_e1
    isZero = zE & zF
    isSpecial = (adj >> (E - 1)) == 3
    top3 = np.where(isSpecial, 0b110 | (~zF).astype(np.int64), np.where(isZero, 0, adj >> (E - 2)))
    exp = (top3 << (E - 2)) | (adj & ((1 << (E - 2)) - 1))
    return (sign << (E + S)) | (exp << F) | np.where(zE, subFract, fract)


def rec_to_fn(rec, E: int, S: int) -> np.ndarray:
    """recoded -> IEEE, bit-exact port of recFNToFN.v."""
    r = np.asarray(rec, dtype=np.int64)
    F = S - 1
    sign = (r >> (E + S)) & 1
    exp = (r >> F) & ((1 << (E + 1)) - 1)
    fract = r & ((1 << F) - 1)
    isSpecial = (exp >> (E - 1)) == 3
    isNaN = isSpecial & (((exp >> (E - 2)) & 1) == 1)
    isInf = isSpecial & (((exp >> (E - 2)) & 1) == 0)
    isZero = (exp >> (E - 2)) == 0
    sig = (np.where(isZero, 0, 1) << F) | fract
    minNorm = (1 << (E - 1)) + 2
    isSub = exp < minNorm
    ndw = _clog2(S)
    dist = (minNorm - 1 - exp) & ((1 << ndw) - 1)
    expOut = np.where(isSub, 0, (exp - minNorm + 1) & ((1 << E) - 1))
    expOut = expOut | np.where(isNaN | isInf, (1 << E) - 1, 0)
    fractOut = np.where(isSub, ((sig >> 1) >> dist) & ((1 << F) - 1), np.where(isInf, 0, fract))
    return (sign << (E + F)) | (expOut << F) | fractOut
