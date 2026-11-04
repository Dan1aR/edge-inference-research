#!/usr/bin/env python
"""E7 / A+C: gate-level simulation of the synthesised (Nangate45) PEs and edge units on real YOLOS-tiny
operand streams; bit-exact check against the numpy model; per-net toggle counts for power (sta_power.py).

Weight-stationary R x C array (default 16 x 16): for one block of C output columns, *all* K-tiles of the
GEMM are streamed in order (sequential psum semantics), every PE row is evaluated on its exact input
stream: a_in = A[m, k] for all M rows of A (tokens), stationary weight B[k, n], psum from the gate-level
output of the row above (row 0: previous K-tile result from the accumulator SRAM, or +0 for 'chunk').
The bottom-edge netlist (recFNToFN / fp32 edge adder of 'chunk') is simulated on the column outputs and
compared with the model.  Output-stationary PEs: a block of PEs, each accumulating several outputs
back to back (clr at k = 0), acc register resolved by the fixed-point iteration of gatesim.simulate.

  nice -n 19 .venv/bin/python scripts/hw/array/power_sim.py --var bf16 [--df ws|os] [--files 'qkv*img0']
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fpmodel as fm  # noqa: E402
from common import ACC_REC, BUILD, LIBS, RESULTS  # noqa: E402
from gatesim import Activity, GateNetlist, pack_bits  # noqa: E402
from verify_rtl import tile_states  # noqa: E402

SYN = BUILD / "synth"
LIB = LIBS["nangate45"]


def netlist(tag: str) -> GateNetlist:
    return GateNetlist(SYN / f"{tag}.nangate45.area.net.json", LIB, tag)


def rec_acc(var: str, s, c=None):
    """model values (float64) -> recoded psum bus of the variant."""
    AE, AS = ACC_REC[var]
    fmt = fm.ACC_FMT[var]
    if var == "kahan":
        return (fm.fn_to_rec(fm.encode(s, "bf16"), 8, 8) << 17) | fm.fn_to_rec(fm.encode(c, "bf16"), 8, 8)
    return fm.fn_to_rec(fm.encode(s, fmt), AE, AS)


def ieee_of_rec(var: str, rec):
    AE, AS = ACC_REC[var]
    if var == "kahan":
        return (fm.rec_to_fn(rec >> 17, 8, 8) << 16) | fm.rec_to_fn(rec & ((1 << 17) - 1), 8, 8)
    return fm.rec_to_fn(rec, AE, AS)


def ieee_of_model(var: str, s, c=None):
    if var == "kahan":
        return (fm.encode(s, "bf16") << 16) | fm.encode(c, "bf16")
    return fm.encode(s, fm.ACC_FMT[var])


def same_bits(var, got, exp):
    """bitwise equality, NaN == NaN (canonical NaN encodings may differ)."""
    ok = got == exp
    if not ok.all():
        fmt = "bf16" if var == "kahan" else fm.ACC_FMT[var]
        if var == "kahan":
            parts = [(got >> 16, exp >> 16), (got & 0xFFFF, exp & 0xFFFF)]
            ok |= np.all([(x == y) | (np.isnan(fm.decode(x, fmt)) & np.isnan(fm.decode(y, fmt))) for x, y in parts], 0)
        else:
            ok |= np.isnan(fm.decode(got, fmt)) & np.isnan(fm.decode(exp, fmt))
    return ok


def self_check_converters(var: str, rng) -> dict:
    """Python ports of fNToRecFN / recFNToFN vs the synthesised edge netlists."""
    res = {}
    g = netlist("edge_west")
    x = np.arange(1 << 16, dtype=np.int64)
    V, _ = g.simulate({"a": x}, {}, x.size, np.zeros(-(-x.size // 64), np.uint64))
    got = g.get_nets(V, g.ports["r"][1], x.size)
    res["edge_west_exhaustive_mismatch"] = int((got != fm.fn_to_rec(x, 8, 8)).sum())
    if var != "chunk":
        AE, AS = ACC_REC[var]
        IW = 32 if var == "kahan" else AE + AS
        n = 1 << 16
        ps = rng.integers(0, 1 << IW, n, dtype=np.int64) if IW > 16 else np.arange(n, dtype=np.int64)
        gt = netlist(f"edge_top_{var.replace('_sep', '')}")
        V, _ = gt.simulate({"w": np.zeros(n, np.int64), "ps": ps}, {}, n, np.zeros(-(-n // 64), np.uint64))
        got = gt.get_nets(V, gt.ports["rps"][1], n)
        if var == "kahan":
            exp = (fm.fn_to_rec(ps >> 16, 8, 8) << 17) | fm.fn_to_rec(ps & 0xFFFF, 8, 8)
        else:
            exp = fm.fn_to_rec(ps, AE, AS)
        res["edge_top_mismatch"] = int((got != exp).sum())
        res["edge_top_vectors"] = n
    return res


# ------------------------------------------------------------------------------------------------
def ws_run(var, Ab, Bb, R, C, n0, unit_delay_every, log):
    M, K = Ab.shape
    Bt = Bb[:, n0:n0 + C]
    A = fm.decode(Ab, "bf16"); B = fm.decode(Bt, "bf16")
    S, Cs, ACC = tile_states(var, A, B, R)
    nkt = len(S) - 1
    g = netlist(f"pe_ws_{var}")
    base = var.replace("_sep", "")            # *_sep variants share the edge units of the fused ones
    gb = netlist(f"edge_bot_{base}")
    gt = netlist(f"edge_top_{base}")
    gw = netlist("edge_west")
    act, act_b, act_t, act_w = Activity(g), Activity(gb), Activity(gt), Activity(gw)
    nbits = C * M
    nw = -(-nbits // 64)
    seg = np.zeros(nbits, np.uint8); seg[::M] = 1
    segm = pack_bits(seg, nw)
    full = pack_bits(np.ones(nbits, np.uint8), nw)
    valid = full & ~segm
    a_rec = fm.fn_to_rec(Ab, 8, 8)                 # (M, K)
    w_rec = fm.fn_to_rec(Bt, 8, 8)                 # (K, C)
    D_s, _, _ = g.flop_nets("s_out")
    mism = dict(pe_chain=0, edge=0, checked=0)
    segw = pack_bits(np.eye(1, M, 0, dtype=np.uint8)[0], -(-M // 64))
    fullw = pack_bits(np.ones(M, np.uint8), -(-M // 64))
    iters = []
    for kt in range(nkt):
        if var == "chunk":
            s_cur = np.zeros(nbits, np.int64)
        else:
            s_cur = rec_acc(var, S[kt], Cs[kt]).T.reshape(-1)          # column-major: c*M + m
            ps_ieee = ieee_of_model(var, S[kt], Cs[kt]).T.reshape(-1)
            w_last = np.repeat(Bt[min(kt * R, K - 1)], M)
            Vt, _ = gt.simulate({"w": w_last, "ps": ps_ieee}, {}, nbits, segm)
            assert np.array_equal(gt.get_nets(Vt, gt.ports["rps"][1], nbits), s_cur), "top converter"
            act_t.add(Vt, valid, full, nbits - C, False, segm)
        for r in range(R):
            k = kt * R + r
            a = np.tile(a_rec[:, k], C) if k < K else np.zeros(nbits, np.int64)
            Vw, _ = gw.simulate({"a": Ab[:, k] if k < K else np.zeros(M, np.int64)}, {}, M, segw)
            act_w.add(Vw, fullw & ~segw, fullw, M - 1, False, segw)
            w = np.repeat(w_rec[k], M) if k < K else np.zeros(nbits, np.int64)
            wn = np.repeat(w_rec[k - 1], M) if 0 < k <= K else np.zeros(nbits, np.int64)
            ports = {"a_in": a, "s_in": s_cur, "w_in": wn, "wload": np.zeros(nbits, np.int64),
                     "clk": np.zeros(nbits, np.int64)}
            V, it = g.simulate(ports, {"w": w}, nbits, segm)
            iters.append(it)
            ud = unit_delay_every > 0 and kt % unit_delay_every == 0
            act.add(V, valid, full, nbits - C, ud, segm)
            s_cur = g.get_nets(V, D_s, nbits)
        # bottom edge
        if var == "chunk":
            acc_in = fm.encode(ACC[kt], "fp32").T.reshape(-1)
            Vb, _ = gb.simulate({"ps": s_cur, "acc_in": acc_in, "clk": np.zeros(nbits, np.int64)}, {}, nbits, segm)
            Db, _, _ = gb.flop_nets("acc_out")
            got = gb.get_nets(Vb, Db, nbits)
            exp = fm.encode(ACC[kt + 1], "fp32").T.reshape(-1)
            part = fm.encode(fm.seq_matmul("bf16", A[:, kt * R:(kt + 1) * R], B[kt * R:(kt + 1) * R]), "bf16").T.reshape(-1)
            mism["pe_chain"] += int((~same_bits("bf16", fm.rec_to_fn(s_cur, 8, 8), part)).sum())
            mism["edge"] += int((~same_bits("fp32", got, exp)).sum())
        else:
            Vb, _ = gb.simulate({"rps": s_cur}, {}, nbits, segm)
            got = gb.get_nets(Vb, gb.ports["ps"][1], nbits)
            exp = ieee_of_model(var, S[kt + 1], Cs[kt + 1]).T.reshape(-1)
            mism["pe_chain"] += int((~same_bits(var, ieee_of_rec(var, s_cur), exp)).sum())
            mism["edge"] += int((~same_bits(var, got, exp)).sum())
        act_b.add(Vb, valid, full, nbits - C, False, segm)
        mism["checked"] += nbits
        if kt % 8 == 0 or kt == nkt - 1:
            log(f"  kt {kt+1}/{nkt}  mismatches {mism}")
    return act, act_b, act_t, act_w, mism, dict(K=K, M=M, ktiles=nkt, n0=int(n0), C=C, R=R, flop_iterations_max=int(max(iters)),
                                   macs_simulated=int(M * C * K), pe_cycles=int(nkt * R * (nbits - C)))


def os_run(var, Ab, Bb, n_pe, n_out, rng, unit_delay, log):
    """n_pe PEs, each computes n_out outputs back to back (K steps each, clr at k=0)."""
    M, K = Ab.shape
    N = Bb.shape[1]
    g = netlist(f"pe_os_{var}")
    act = Activity(g)
    rows = rng.choice(M, size=(n_pe, n_out), replace=False if M >= n_pe * n_out else True)
    cols = rng.choice(N, size=n_pe, replace=True)
    L = n_out * K
    nbits = n_pe * L
    nw = -(-nbits // 64)
    A = fm.decode(Ab, "bf16"); B = fm.decode(Bb, "bf16")
    a_rec = fm.fn_to_rec(Ab, 8, 8); b_rec = fm.fn_to_rec(Bb, 8, 8)
    a_s = np.zeros(nbits, np.int64); b_s = np.zeros(nbits, np.int64); clr = np.zeros(nbits, np.int64)
    q0 = np.zeros(nbits, np.int64); expD = np.zeros(nbits, np.int64)
    for p in range(n_pe):
        n = cols[p]
        for u in range(n_out):
            m = rows[p, u]
            o = p * L + u * K
            a_s[o:o + K] = a_rec[m]
            b_s[o:o + K] = b_rec[:, n]
            clr[o] = 1
            s = np.zeros(1); c = np.zeros(1)
            states = []
            for k in range(K):
                pk = np.array([A[m, k] * B[k, n]])
                if var == "kahan":
                    s, c = fm.mac_step(var, s, pk, c)
                else:
                    s = fm.mac_step(var, s, pk)
                states.append((s[0], c[0]))
            st = np.array(states)
            recs = rec_acc(var, st[:, 0], st[:, 1])
            expD[o:o + K] = ieee_of_model(var, st[:, 0], st[:, 1])
            q0[o + 1:o + K] = recs[:-1]
            q0[o] = recs[-1] if u == 0 else q0[o]      # state before clr is irrelevant (clr)
    seg = np.zeros(nbits, np.uint8); seg[::L] = 1
    segm = pack_bits(seg, nw)
    full = pack_bits(np.ones(nbits, np.uint8), nw)
    valid = full & ~segm
    ports = {"a_in": a_s, "b_in": b_s, "clr": clr, "clk": np.zeros(nbits, np.int64)}
    V, it = g.simulate(ports, {"acc": q0, "a_out": np.roll(a_s, 1), "b_out": np.roll(b_s, 1)}, nbits, segm,
                       maxit=60)
    Dacc, _, _ = g.flop_nets("acc")
    got = ieee_of_rec(var, g.get_nets(V, Dacc, nbits))
    mism = int((~same_bits(var, got, expD)).sum())
    act.add(V, valid, full, nbits - n_pe, unit_delay, segm)
    log(f"  OS {n_pe} PEs x {n_out} outputs x K={K}: mismatches {mism}, flop iterations {it}")
    return act, dict(checked=nbits, mismatches=mism, flop_iterations=it, K=K, pe_cycles=int(nbits - n_pe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True)
    ap.add_argument("--df", default="ws", choices=["ws", "os"])
    ap.add_argument("--files", default="*img0")
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--unit-delay-every", type=int, default=4, help="glitch sim on every n-th K-tile (0: off)")
    ap.add_argument("--os-pes", type=int, default=64)
    ap.add_argument("--os-bits", type=int, default=1 << 17, help="OS: PE-cycles per run")
    ap.add_argument("--operands", default=str(RESULTS / "operands"))
    ap.add_argument("--out", default=str(RESULTS / "activity"))
    args = ap.parse_args()
    rng = np.random.default_rng(11)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    def log(s):
        print(s, flush=True)

    checks = self_check_converters(args.var, rng) if args.df == "ws" else {}
    log(f"{args.var}: converter self-check {checks}")
    for f in sorted(Path(args.operands).glob(f"{args.files}.npz")):
        d = np.load(f)
        Ab, Bb = d["A"].astype(np.int64), d["B"].astype(np.int64)
        t0 = time.time()
        tag = f"{args.df}_{args.var}_{f.stem}"
        log(f"{tag}: A {Ab.shape} B {Bb.shape}")
        if args.df == "ws":
            n0 = (Bb.shape[1] // 2 // args.C) * args.C
            act, act_b, act_t, act_w, mism, info = ws_run(args.var, Ab, Bb, args.R, args.C, n0,
                                                          args.unit_delay_every, log)
            rec = dict(variant=args.var, dataflow="ws", file=f.name, info=info, mismatches=mism, converters=checks,
                       pe=act.to_dict(), edge_bottom=act_b.to_dict(), edge_top=act_t.to_dict(),
                       edge_west=act_w.to_dict(), seconds=round(time.time() - t0, 1))
        else:
            n_out = max(1, args.os_bits // (args.os_pes * Ab.shape[1]))
            act, info = os_run(args.var, Ab, Bb, args.os_pes, n_out, rng, args.unit_delay_every > 0, log)
            rec = dict(variant=args.var, dataflow="os", file=f.name, info=info, mismatches=info["mismatches"],
                       pe=act.to_dict(), seconds=round(time.time() - t0, 1))
        (outdir / f"{tag}.json").write_text(json.dumps(rec))
        log(f"{tag}: done in {rec['seconds']} s, mismatches {rec['mismatches']}")


if __name__ == "__main__":
    main()
