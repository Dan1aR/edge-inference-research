"""E1-hw: what the real tensor cores of this GPU do with partial sums, and what fp16 accumulation buys.

Part A (precision probe, in the spirit of Fasi et al., PeerJ CS 2021, "Numerical behavior of NVIDIA
tensor cores"): one output = big + small, where big = a0*b0 = 2^E and small = a1*b1 = 2^e; we find the
largest exponent gap E - e for which the small term survives (a) inside one MMA K-block and (b) across
K-blocks (running sum in the accumulator registers), and whether lost bits are truncated or rounded.
Part B (speed): tensor-core GEMM with fp16 vs fp32 accumulators on the YOLOS / RT-DETR GEMM shapes.

Usage: python scripts/e1_hw_probe.py results/e1/hw_probe.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.triton_kernels.tc_matmul import tc_matmul  # noqa: E402

DEV = "cuda"
RANGE = {"fp16": (-14, 15), "bf16": (-60, 60), "fp8": (-6, 8)}  # normal exponent ranges used for factors


def _factors(exp: int, in_dtype: str):
    """Split 2^exp into two exactly representable factors."""
    lo, hi = RANGE[in_dtype]
    x = max(lo, min(hi, exp // 2))
    y = exp - x
    if not (lo <= y <= hi):
        return None
    return 2.0 ** x, 2.0 ** y


def probe(in_dtype: str, acc: str, gap: int, same_block: bool, frac: float = 1.0, bk: int = 32):
    """Return the value of (big + frac*small) - big as computed by the tensor cores, divided by small."""
    E = 6 if in_dtype == "fp8" else 10
    e = E - gap
    fb, fs = _factors(E, in_dtype), _factors(e, in_dtype)
    if fb is None or fs is None:
        return None
    K = 2 * bk
    a = torch.zeros(16, K, dtype=torch.float32, device=DEV)
    b = torch.zeros(K, 16, dtype=torch.float32, device=DEV)
    k_small = 1 if same_block else bk  # second element of the first block, or first of the second block
    a[:, 0], b[0, :] = fb
    a[:, k_small], b[k_small, :] = fs[0] * frac, fs[1]
    out = tc_matmul(a, b, acc=acc, in_dtype=in_dtype, out_dtype=torch.float32, block=(16, 16, bk)).double()
    return ((out[0, 0] - 2.0 ** E) / 2.0 ** e).item()


def precision_table():
    res = {}
    for in_dtype, acc in (("fp16", "fp16"), ("fp16", "fp32"), ("bf16", "fp32"), ("fp8", "fp32")):
        row = {}
        for same in (True, False):
            kept = [g for g in range(1, 40) if (v := probe(in_dtype, acc, g, same)) is not None and abs(v - 1.0) < 1e-9]
            # 1.5 * 2^e at the last kept gap + 1: truncation drops it to 1, RNE rounds to 2 (ties-to-even) or 1
            g = (max(kept) + 1) if kept else None
            half = probe(in_dtype, acc, g, same, frac=1.5) if g else None
            row["same_block" if same else "next_block"] = {
                "max_gap_kept": max(kept) if kept else None, "value_at_gap+1_with_1.5x_small": half}
        res[f"{in_dtype}_in/{acc}_acc"] = row
        print(in_dtype, acc, row, flush=True)
    return res


def bench(f, n=20):
    f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def speed_table():
    shapes = {  # (M, K, N): YOLOS-tiny batch 8 (1125 tokens), YOLOS-small, RT-DETR-like conv as GEMM
        "tiny_qkv": (8 * 1125, 192, 576), "tiny_mlp_up": (8 * 1125, 192, 768), "small_mlp_up": (8 * 1125, 384, 1536),
        "small_mlp_down": (8 * 1125, 1536, 384), "conv3x3_256": (8 * 1600, 2304, 256), "big_4k": (4096, 4096, 4096),
    }
    res = {}
    for name, (M, K, N) in shapes.items():
        a = torch.randn(M, K, device=DEV)
        b = torch.randn(K, N, device=DEV) * 0.05
        row = {}
        for acc in ("fp32", "fp16"):
            row[f"triton_fp16in_{acc}acc_ms"] = bench(lambda: tc_matmul(a, b, acc=acc, in_dtype="fp16", out_dtype=torch.float16))
        ah, bh = a.half(), b.half()
        row["cublas_fp16_ms"] = bench(lambda: ah @ bh)
        row["cublas_bf16_ms"] = bench(lambda: a.bfloat16() @ b.bfloat16())
        row["speedup_fp16acc_vs_fp32acc"] = row["triton_fp16in_fp32acc_ms"] / row["triton_fp16in_fp16acc_ms"]
        ref = a.double() @ b.double()
        for acc in ("fp32", "fp16"):
            y = tc_matmul(a, b, acc=acc, in_dtype="fp16", out_dtype=torch.float32).double()
            row[f"rel_err_{acc}acc"] = ((y - ref).norm() / ref.norm()).item()
        res[name] = row
        print(name, {k: round(v, 4) for k, v in row.items()}, flush=True)
    return res


def main():
    out = {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
           "torch": torch.__version__, "precision": precision_table(), "speed": speed_table()}
    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
