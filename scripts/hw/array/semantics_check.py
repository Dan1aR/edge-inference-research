#!/usr/bin/env python
"""E7: hardware semantics (single rounding of s + a*b, verified RTL/gate level) vs the software emulation
used for training/eval (src/bf16_accum.py::bf16_accum_matmul_ref and the Triton kernel: s + a*b is formed
in fp32 and then rounded to the accumulator format, i.e. a double rounding) on the full dumped YOLOS-tiny
matmuls.  Reports the fraction of outputs whose final bf16 result differs and the max relative error of
each against an exact float64 reference.  (Sequential order over K, chunk = 16 for 'chunk'.)

  nice -n 19 .venv/bin/python scripts/hw/array/semantics_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fpmodel as fm  # noqa: E402
from common import RESULTS  # noqa: E402


def to_bf16(x):
    return fm.round_fmt(x, "bf16")


def main():
    res = {}
    for f in sorted((RESULTS / "operands").glob("*img0.npz")):
        d = np.load(f)
        A = fm.decode(d["A"].astype(np.int64), "bf16")
        B = fm.decode(d["B"].astype(np.int64), "bf16")
        if A.shape[0] * B.shape[1] > 300_000:          # subsample output columns for the big ones
            cols = np.linspace(0, B.shape[1] - 1, max(1, 300_000 // A.shape[0])).astype(int)
            B = B[:, cols]
        exact = A @ B                                   # float64 (exact enough as reference)
        r = {}
        for var in ["fp32", "bf16", "fp16", "e8m13", "kahan", "chunk"]:
            ch = 16 if var == "chunk" else 0
            hw = to_bf16(fm.seq_matmul(var, A, B, chunk=ch))
            sw = to_bf16(fm.seq_matmul(var, A, B, chunk=ch, sw=True))
            den = np.maximum(np.abs(exact), 1e-30)
            fin = np.isfinite(hw) & np.isfinite(sw)
            r[var] = dict(outputs=int(hw.size), differ=int((hw != sw).sum()),
                          differ_frac=float((hw != sw).mean()),
                          max_ulp_diff_bf16=float(np.nanmax(np.abs(hw - sw)[fin] / np.maximum(
                              np.abs(np.ldexp(1.0, np.frexp(np.abs(sw[fin]))[1] - 8)), 1e-300)) if fin.any() else 0.0),
                          median_rel_err_hw=float(np.median(np.abs(hw - exact)[fin] / den[fin])),
                          median_rel_err_sw=float(np.median(np.abs(sw - exact)[fin] / den[fin])),
                          nonfinite_hw=int((~np.isfinite(hw)).sum()))
            print(f.stem, var, r[var], flush=True)
        res[f.stem] = r
    (RESULTS / "semantics_check.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
