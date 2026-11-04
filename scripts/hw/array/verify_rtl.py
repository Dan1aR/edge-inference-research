#!/usr/bin/env python
"""E7 / A: RTL simulation (Icarus Verilog) of the R x C weight-stationary array for every accumulator
variant, compared bit-exactly with the numpy model (fpmodel.py) of the same semantics.

Stimulus: K = nkt*R (several K-tiles: exercises the psum feed-back of the sequential variants and the
fp32 edge accumulator of 'chunk'), weights shifted in with wload, inputs skewed as in a real array.
Operand sets: 'random' (wide exponent range, exact/near cancellations, zeros, tiny and huge values) and
real YOLOS-tiny tiles from results/hw/array/operands/*.npz (scripts/hw/array/dump_operands.py).

  nice -n 19 .venv/bin/python scripts/hw/array/verify_rtl.py [--R 16 --C 16] [--vars fp32,bf16,...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fpmodel as fm  # noqa: E402
from common import (BUILD, ENV, HF_FILES, RESULTS, RTL, VAR_ID, iw_bits, prepare_hf)  # noqa: E402


def tile_states(var, A, W, R):
    """Per-K-tile states of the WS array model: s/c entering tile kt and the fp32 edge accumulator."""
    M, K = A.shape
    N = W.shape[1]
    nkt = -(-K // R)
    s = np.zeros((M, N)); c = np.zeros((M, N)); acc = np.zeros((M, N))
    S, Cs, ACC = [s], [c], [acc]
    for kt in range(nkt):
        if var == "chunk":
            s = np.zeros((M, N))
        for r in range(R):
            k = kt * R + r
            p = A[:, k:k + 1] * W[k:k + 1, :] if k < K else np.zeros((M, N))
            if var == "kahan":
                s, c = fm.mac_step(var, s, p, c)
            else:
                s = fm.mac_step(var, s, p)
        if var == "chunk":
            acc = fm.rnd_sum(acc, s, "fp32")
        S.append(s); Cs.append(c); ACC.append(acc)
    return S, Cs, ACC


def make_stimulus(var, Ab, Wb, R, C):
    """Ab (M,K) bf16 bits, Wb (K,C) bf16 bits -> (lines, checks).  checks: (cycle, col, m, kt)."""
    M, K = Ab.shape
    A = fm.decode(Ab, "bf16"); W = fm.decode(Wb, "bf16")
    S, Cs, ACC = tile_states(var, A, W, R)
    fmt = fm.ACC_FMT[var]
    IW = iw_bits(var)
    nkt = len(S) - 1
    lines = []
    expect = []  # (cycle, col, expected bits)

    def pack(wl, a, w, ps, acc):
        v = wl
        for vec, width in ((a, 16), (w, 16), (ps, IW), (acc, 32)):
            word = 0
            for i in reversed(range(len(vec))):
                word = (word << width) | int(vec[i])
            v = (v << (len(vec) * width)) | word
        return v

    zero_r, zero_c = [0] * R, [0] * C
    for kt in range(nkt):
        for j in range(R):   # weight shift-in: row R-1 receives the first value
            k = kt * R + (R - 1 - j)
            w = [int(Wb[k, c]) if k < K else 0 for c in range(C)]
            lines.append(pack(1, zero_r, w, zero_c, zero_c))
        if var == "kahan":
            ps_tile = (fm.encode(S[kt], "bf16") << 16) | fm.encode(Cs[kt], "bf16")
        elif var == "chunk":
            ps_tile = np.zeros((M, C), dtype=np.int64)
        else:
            ps_tile = fm.encode(S[kt], fmt)
        acc_tile = fm.encode(ACC[kt], "fp32")
        if var == "kahan":
            exp_tile = (fm.encode(S[kt + 1], "bf16") << 16) | fm.encode(Cs[kt + 1], "bf16")
        elif var == "chunk":
            exp_tile = fm.encode(ACC[kt + 1], "fp32")
        else:
            exp_tile = fm.encode(S[kt + 1], fmt)
        t0 = len(lines)
        lat = R if var == "chunk" else R - 1
        for t in range(M + R + C + 1):
            a = [int(Ab[t - r, kt * R + r]) if 0 <= t - r < M and kt * R + r < K else 0 for r in range(R)]
            ps = [int(ps_tile[t - c, c]) if 0 <= t - c < M else 0 for c in range(C)]
            acc = [int(acc_tile[t - R - c, c]) if 0 <= t - R - c < M else 0 for c in range(C)]
            lines.append(pack(0, a, zero_c, ps, acc))
        for m in range(M):
            for c in range(C):
                expect.append((t0 + m + lat + c, c, int(exp_tile[m, c])))
    return lines, expect


def write_tb(build: Path, tag: str, var: str, R: int, C: int, T: int) -> Path:
    IW = iw_bits(var)
    OW = 32 if var == "chunk" else IW
    W = 1 + R * 16 + C * 16 + C * IW + C * 32
    tb = build / f"tb_{tag}.v"
    tb.write_text(f"""`timescale 1ns/1ps
module tb_{tag};
  reg clk = 0; reg wload;
  reg [{R*16-1}:0] a_in; reg [{C*16-1}:0] w_in; reg [{C*IW-1}:0] ps_in; reg [{C*32-1}:0] acc_in;
  wire [{C*OW-1}:0] ps_out;
  reg [{W-1}:0] mem [0:{T-1}];
  integer t, fd;
  ws_array #({R}, {C}, {VAR_ID[var]}) dut (clk, wload, a_in, w_in, ps_in, acc_in, ps_out);
  initial begin
    $readmemh("stim_{tag}.hex", mem);
    fd = $fopen("out_{tag}.hex", "w");
    for (t = 0; t < {T}; t = t + 1) begin
      {{wload, a_in, w_in, ps_in, acc_in}} = mem[t];
      #1 clk = 1; #1 $fwrite(fd, "%h\\n", ps_out); #1 clk = 0;
    end
    $fclose(fd); $finish;
  end
endmodule
""")
    return tb


def run_case(build: Path, tag: str, var: str, Ab, Wb, R: int, C: int) -> dict:
    t_start = time.time()
    lines, expect = make_stimulus(var, Ab, Wb, R, C)
    (build / f"stim_{tag}.hex").write_text("\n".join(f"{v:x}" for v in lines) + "\n")
    write_tb(build, tag, var, R, C, len(lines))
    srcs = [str(build / f) for f in HF_FILES] + [str(build / RTL.name), str(build / f"tb_{tag}.v")]
    exe = build / f"sim_{tag}.vvp"
    subprocess.run([str(ENV / "iverilog"), "-g2005", "-I", str(build), "-o", str(exe), "-s", f"tb_{tag}"] + srcs,
                   check=True, capture_output=True, text=True)
    subprocess.run(["nice", "-n", "19", str(ENV / "vvp"), "-n", str(exe)], cwd=build, check=True,
                   capture_output=True, text=True)
    out = (build / f"out_{tag}.hex").read_text().split()
    IW = iw_bits(var)
    OW = 32 if var == "chunk" else IW
    fmt = "fp32" if var == "chunk" else fm.ACC_FMT[var]
    bad = []
    nan_both = 0
    for cyc, c, e in expect:
        word = out[cyc]
        if "x" in word.lower() or "z" in word.lower():
            v = None
        else:
            v = (int(word, 16) >> (c * OW)) & ((1 << OW) - 1)
        if v == e:
            continue
        if v is not None:   # NaN encodings may differ (canonical qNaN): compare as NaN
            parts = [(v >> 16, e >> 16), (v & 0xFFFF, e & 0xFFFF)] if var == "kahan" else [(v, e)]
            f = "bf16" if var == "kahan" else fmt
            if all(x == y or (np.isnan(fm.decode(x, f)) and np.isnan(fm.decode(y, f))) for x, y in parts):
                nan_both += 1
                continue
        bad.append(dict(cycle=cyc, col=c, expected=hex(e), got=word if v is None else hex(v)))
    for f in build.glob(f"*_{tag}.*"):
        f.unlink()
    return dict(checked=len(expect), mismatches=len(bad), nan_matches=nan_both, first_bad=bad[:5],
                cycles=len(lines), seconds=round(time.time() - t_start, 1))


def random_operands(rng, M, K, C, var):
    def bf(shape, lo, hi):
        v = rng.standard_normal(shape) * np.exp2(rng.integers(lo, hi, shape))
        return fm.bf16_from_f32(v.astype(np.float32))
    A = bf((M, K), -12, 12)
    W = bf((K, C), -12, 12)
    # zeros, tiny/huge values (fp16 overflow / fp32 underflow of the product), exact cancellation
    A[rng.random((M, K)) < 0.05] = 0
    A[0, :] = fm.bf16_from_f32(np.full(K, 2.0 ** -70, np.float32))
    A[1, :] = fm.bf16_from_f32(np.full(K, 3.0e3, np.float32))
    if K % 2 == 0:   # exact pairwise cancellation in rows 2, 3: a_{2j+1} = a_{2j}, w_{2j+1} = -w_{2j}
        W[1::2, :] = W[0::2, :] ^ 0x8000
        A[2:4, 1::2] = A[2:4, 0::2]
        A[3, 1::2] ^= 1           # near cancellation
    return A.astype(np.int64), W.astype(np.int64)


def real_operands(path: Path, rng, M, K, C, R):
    d = np.load(path)
    A, W = d["A"], d["B"]
    m0 = rng.integers(0, A.shape[0] - M + 1)
    k0 = (rng.integers(0, max(1, (A.shape[1] - K) // R + 1))) * R
    n0 = rng.integers(0, W.shape[1] - C + 1)
    return A[m0:m0 + M, k0:k0 + K].astype(np.int64), W[k0:k0 + K, n0:n0 + C].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--M", type=int, default=48)
    ap.add_argument("--ktiles", type=int, default=3)
    ap.add_argument("--vars", default="fp32,bf16,fp16,e8m13,kahan,chunk,bf16_sep,fp16_sep,e8m13_sep")
    ap.add_argument("--operands", default=str(RESULTS / "operands"))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--no-patch", action="store_true", help="unpatched HardFloat addRecFN (negative test)")
    ap.add_argument("--out", default=str(RESULTS / "verify_rtl.json"))
    args = ap.parse_args()
    build = prepare_hf(BUILD / ("rtlsim_nopatch" if args.no_patch else "rtlsim"), patch=not args.no_patch)
    rng = np.random.default_rng(7)
    K = args.ktiles * args.R
    cases = []
    for var in args.vars.split(","):
        cases.append((f"{var}_random", var, *random_operands(rng, args.M, K, args.C, var)))
        for p in sorted(Path(args.operands).glob("*img0.npz")):
            cases.append((f"{var}_{p.stem}", var, *real_operands(p, rng, args.M, K, args.C, args.R)))
    res = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {name: ex.submit(run_case, build, name, var, Ab, Wb, args.R, args.C) for name, var, Ab, Wb in cases}
        for name, f in futs.items():
            res[name] = f.result()
            print(name, res[name], flush=True)
    out = dict(array=f"{args.R}x{args.C} weight-stationary", M=args.M, K=K, rtl=str(RTL.relative_to(RTL.parents[4])),
               simulator=subprocess.run([str(ENV / "iverilog"), "-V"], capture_output=True, text=True).stdout.split("\n")[0],
               cases=res, total_checked=sum(r["checked"] for r in res.values()),
               total_mismatches=sum(r["mismatches"] for r in res.values()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("total checked", out["total_checked"], "mismatches", out["total_mismatches"])


if __name__ == "__main__":
    main()
