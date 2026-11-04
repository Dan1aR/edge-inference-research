#!/usr/bin/env python
"""E6 / step 4: logic synthesis of BF16-input MAC units with FP32 vs BF16 accumulation.

Flow (all open source, CPU only, no system install):
  RTL      : scripts/hw/rtl/mac_units.v on top of Berkeley HardFloat Release 1
  synthesis: Yosys 0.70 (pip `yowasp-yosys`, WebAssembly build)
             (1) generic:  `synth -flatten` -> 2-input gate netlist (yosys internal cell library)
             (2) Nangate45: `dfflibmap` + `abc -liberty` with NangateOpenCellLibrary_typical.lib
                 (a) default ABC area-oriented script, (b) timing-driven `abc -D 2000` (2 ns target)
  check    : the four MAC datapaths (standard IEEE in/out wrappers t_*) are synthesised to generic
             gates, written as JSON and simulated bit-parallel in Python on random vectors against a
             numpy/torch reference (gate-level functional self-check of the synthesised netlist).
Not done: placement/routing, wire load, timing sign-off, power simulation (no switching activity).

Usage:
  scripts/hw/fetch_hw_deps.sh            # once: HardFloat + Nangate45 liberty + yowasp-yosys
  nice -n 19 .venv/bin/python scripts/hw/run_synth.py [--deps /tmp/hwdeps] [--jobs 2]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
YOSYS = str(Path(sys.executable).parent / "yowasp-yosys")
HF_FILES = ["HardFloat_primitives.v", "HardFloat_rawFN.v", "isSigNaNRecFN.v", "addRecFN.v", "mulRecFN.v",
            "mulAddRecFN.v", "fNToRecFN.v", "recFNToFN.v", "HardFloat_specialize.v", "mac_units.v"]
# liberty files (fetched by fetch_hw_deps.sh): name -> (file, NAND2 x1 area um^2, dont_use patterns)
LIBS = {
    "nangate45": ("nangate45_typ.lib", 0.798, []),
    "sky130hd": ("sky130hd_tt.lib", 3.7536,
                 ["sky130_fd_sc_hd__probe_p_*", "sky130_fd_sc_hd__probec_p_*", "sky130_fd_sc_hd__lpflow_*",
                  "sky130_fd_sc_hd__clkdlybuf*", "sky130_fd_sc_hd__dlygate*", "sky130_fd_sc_hd__dlymetal*"]),
}

DESIGNS = {
    # name: description
    "add_fp32": "FP32 adder (addRecFN 8/24)",
    "add_bf16": "BF16 adder (addRecFN 8/8)",
    "mul_bf16": "BF16 x BF16 -> BF16 multiplier, rounded (mulRecFN 8/8)",
    "mul_bf16_exact_fp32": "BF16 x BF16 -> FP32 exact multiplier (8x8 sig, mulRecFNToFullRaw + round to fp32)",
    "mul_fp32": "FP32 x FP32 -> FP32 multiplier (mulRecFN 8/24), reference",
    "fma_fp32_bf16in": "fused a*b+c, a,b BF16, c/result FP32 (mulAddRecFN 8/24, a,b widened)",
    "fma_bf16": "fused a*b+c fully BF16 (mulAddRecFN 8/8)",
    "pe_fp32acc_sep": "PE: exact bf16 mult + FP32 adder, 33b acc reg, 2x17b operand regs",
    "pe_bf16acc_sep": "PE: bf16 mult (rounded) + BF16 adder, 17b acc reg, 2x17b operand regs",
    "pe_fp32acc_fma": "PE: fused bf16-in/FP32-acc FMA, 33b acc reg, 2x17b operand regs",
    "pe_bf16acc_fma": "PE: fused BF16 FMA, 17b acc reg, 2x17b operand regs",
}
TESTS = ["t_add_bf16", "t_mul_bf16", "t_add_fp32", "t_mul_bf16_exact",
         "t_fp32acc_sep", "t_bf16acc_sep", "t_fp32acc_fma", "t_bf16acc_fma"]


# HardFloat Release 1 (Verilog) addRecFN is not correct for sigWidth = 8 (bf16), found by the
# gate-level self-check below (~60% of random bf16 additions wrong, e.g. -51.25 + 0.0187 -> -51.0):
#   localparam alignDistWidth = clog2(sigWidth);   // = 3 for sigWidth = 8 -> max align shift 7
# In the far path the smaller significand ({sig, 5'b0}, sigWidth+5 bits) must be shiftable by at
# least sigWidth+2 positions to collapse into the sticky bit; with a 3-bit shift (max 7 < 10) operands
# whose exponents differ by >= 8 are mis-aligned.  For the IEEE formats HardFloat is tested with
# (sigWidth 11/24/53/113) clog2(sigWidth) == clog2(sigWidth + 3), so the fix below changes nothing
# for them (verified: fp32 netlists/areas identical, all fp32 self-checks pass) and only widens the
# alignment shifter of the bf16 adder to 4 bits.  Applied to the build copy only, never to HardFloat.
HF_PATCHES = [
    ("addRecFN.v",
     "    localparam alignDistWidth = clog2(sigWidth);\n",
     "    localparam alignDistWidth = clog2(sigWidth + 3);  // E6 patch, see scripts/hw/run_synth.py\n"),
]


def prepare_build(deps: Path, build: Path, patch_hardfloat: bool = True) -> None:
    src = deps / "HardFloat-1" / "source"
    for f in src.glob("*.v*"):
        shutil.copy(f, build / f.name)
    if patch_hardfloat:
        for fn, old, new in HF_PATCHES:
            p = build / fn
            t = p.read_text()
            assert t.count(old) == 1, f"unexpected {fn} (patch target not found)"
            p.write_text(t.replace(old, new))
    for f in (src / "RISCV").glob("*"):  # RISC-V NaN conventions (canonical NaN)
        shutil.copy(f, build / f.name)
    shutil.copy(HERE / "rtl" / "mac_units.v", build / "mac_units.v")
    for f, _, _ in LIBS.values():
        shutil.copy(deps / f, build / f)


def yosys(build: Path, script: str, tag: str) -> None:
    ys = build / f"{tag}.ys"
    ys.write_text(script)
    # yowasp-yosys only sees the current directory -> run inside build dir with relative paths
    r = subprocess.run(["nice", "-n", "19", YOSYS, "-q", "-l", f"{tag}.log", "-s", ys.name], cwd=build,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = (build / f"{tag}.log").read_text()[-2000:]
        raise RuntimeError(f"yosys failed for {tag}:\n{tail}\n{r.stderr[-2000:]}")


def read_cmd() -> str:
    return "read_verilog -I. " + " ".join(HF_FILES) + "\n"


def synth_design(build: Path, top: str) -> dict:
    s = read_cmd()
    s += f"synth -flatten -top {top}\nopt_clean -purge\n"
    s += f"tee -q -o {top}.gen.json stat -json\n"
    s += "design -save gen\n"
    for lname, (lib, _, dont) in LIBS.items():
        du = "".join(f" -dont_use {d}" for d in dont)
        s += "design -load gen\n"
        s += f"dfflibmap -liberty {lib}{du}\nabc -liberty {lib}{du}\nopt_clean -purge\n"
        s += f"tee -q -o {top}.{lname}.json stat -liberty {lib} -json\n"
    yosys(build, s, top)

    def load(fn):
        j = json.loads((build / fn).read_text())
        return list(j["modules"].values())[0]

    gen = load(f"{top}.gen.json")
    gcells = {k: v for k, v in gen["num_cells_by_type"].items() if k != "$scopeinfo"}
    g_ff = sum(v for k, v in gcells.items() if "DFF" in k)
    out = dict(
        description=DESIGNS[top],
        generic=dict(cells=sum(gcells.values()) - g_ff, flipflops=g_ff, by_type=gcells),
    )
    for tag, (_, nand2, _) in LIBS.items():
        m = load(f"{top}.{tag}.json")
        cells = {k: v for k, v in m["num_cells_by_type"].items() if k != "$scopeinfo"}
        ff = sum(v for k, v in cells.items() if k.startswith(("DFF", "SDFF")) or "__df" in k)
        area = float(m["area"])
        seq = float(m.get("sequential_area", 0.0))
        out[tag] = dict(
            area_um2=area, sequential_area_um2=seq, combinational_area_um2=area - seq,
            cells=sum(cells.values()), flipflops=ff, nand2_eq=area / nand2, by_type=cells,
        )
    return out


# --------------------------------------------------------------------------------------------
# gate-level functional self-check
# --------------------------------------------------------------------------------------------
def netlist_eval(netlist: dict, inputs: dict[str, np.ndarray], nvec: int) -> dict[str, np.ndarray]:
    mod = list(netlist["modules"].values())[0]
    mask = (1 << nvec) - 1
    val: dict = {}
    ports = mod["ports"]
    for name, p in ports.items():
        if p["direction"] == "input":
            vec = inputs[name].astype(np.uint64)
            for i, bit in enumerate(p["bits"]):
                packed = 0
                bits = ((vec >> np.uint64(i)) & np.uint64(1)).astype(np.uint8)
                packed = int.from_bytes(np.packbits(bits[::-1]).tobytes(), "big") >> ((-nvec) % 8)
                val[bit] = packed

    def get(b):
        if b == "0" or b == "x":
            return 0
        if b == "1":
            return mask
        return val[b]

    cells = [c for c in mod["cells"].values() if c["type"] != "$scopeinfo"]
    ops = {
        "$_AND_": lambda a, b: a & b, "$_OR_": lambda a, b: a | b, "$_XOR_": lambda a, b: a ^ b,
        "$_NAND_": lambda a, b: ~(a & b) & mask, "$_NOR_": lambda a, b: ~(a | b) & mask,
        "$_XNOR_": lambda a, b: ~(a ^ b) & mask, "$_ANDNOT_": lambda a, b: a & ~b & mask,
        "$_ORNOT_": lambda a, b: (a | ~b) & mask,
    }
    pending = cells
    while pending:
        nxt = []
        for c in pending:
            cn = c["connections"]
            ins = [k for k in cn if k != "Y"]
            if any(b not in val and b not in ("0", "1", "x") for k in ins for b in cn[k]):
                nxt.append(c)
                continue
            t = c["type"]
            if t in ops:
                y = ops[t](get(cn["A"][0]), get(cn["B"][0]))
            elif t == "$_NOT_":
                y = ~get(cn["A"][0]) & mask
            elif t == "$_BUF_":
                y = get(cn["A"][0])
            elif t == "$_MUX_":
                s = get(cn["S"][0])
                y = (get(cn["B"][0]) & s) | (get(cn["A"][0]) & ~s & mask)
            else:
                raise ValueError(f"unsupported cell {t}")
            val[cn["Y"][0]] = y
        if len(nxt) == len(pending):
            raise RuntimeError("combinational loop / unresolved nets")
        pending = nxt
    outs = {}
    for name, p in ports.items():
        if p["direction"] == "output":
            res = np.zeros(nvec, dtype=np.uint64)
            for i, bit in enumerate(p["bits"]):
                v = get(bit)
                bits = np.array([(v >> j) & 1 for j in range(nvec)], dtype=np.uint64)
                res |= bits << np.uint64(i)
            outs[name] = res
    return outs


def _bf16_bits(x32: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bits with RNE (inputs are exactly representable when used for operands)."""
    u = x32.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((u >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)
    return ((u + r) >> np.uint64(16)).astype(np.uint64) & np.uint64(0xFFFF)


def _bf16_val(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _round_f64_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Correct RNE rounding of float64 to bf16 (normal range; inf on overflow)."""
    out = np.zeros(x.shape, dtype=np.uint64)
    m, e = np.frexp(x)  # x = m * 2**e, 0.5 <= |m| < 1
    q = np.round(m * 256.0)  # 8 significant bits, banker's rounding (exact in f64)
    y = np.ldexp(q / 256.0, e).astype(np.float32)  # exact: 8-bit significand
    out = _bf16_bits(y)
    return out


def make_vectors(rng, n: int):
    def rnd_bf16(k, lo, hi):
        v = rng.standard_normal(k).astype(np.float32) * np.exp2(rng.integers(lo, hi, k)).astype(np.float32)
        return _bf16_bits(v)

    a = rnd_bf16(n, -12, 12)
    b = rnd_bf16(n, -12, 12)
    prod = _bf16_val(a).astype(np.float64) * _bf16_val(b).astype(np.float64)
    scale = np.exp2(rng.integers(-30, 30, n)).astype(np.float64)
    c64 = rng.standard_normal(n) * np.abs(prod) * scale
    # special cases: exact cancellation, zeros, near-equal magnitudes
    k = n // 8
    c64[:k] = -prod[:k]
    c64[k:2 * k] = -prod[k:2 * k] * (1 + rng.standard_normal(k) * 2.0 ** -9)
    c64[2 * k:2 * k + 8] = 0.0
    a[2 * k + 8:2 * k + 12] = 0
    return a, b, c64


C32_TESTS = {"t_fp32acc_sep", "t_fp32acc_fma", "t_add_fp32", "t_mul_bf16_exact"}
Y32_TESTS = C32_TESTS


def reference(test: str, a, b, c64):
    """Bit-exact IEEE reference (RNE).  Returns (c bits, expected y bits, y width)."""
    fa, fb = _bf16_val(a).astype(np.float64), _bf16_val(b).astype(np.float64)
    if test in C32_TESTS:
        c = c64.astype(np.float32)
        cbits = c.view(np.uint32).astype(np.uint64)
        fc = c.astype(np.float64)
        if test == "t_fp32acc_sep":      # product exact in fp32 (range-limited vectors), then fp32 add
            y = ((fa * fb).astype(np.float32) + c).astype(np.float32)
        elif test == "t_fp32acc_fma":    # single rounding; f64 sum + f64->f32 is innocuous (53 >= 2*24+2)
            y = (fa * fb + fc).astype(np.float32)
        elif test == "t_add_fp32":       # x = {a,b} reinterpreted as an arbitrary fp32
            x = ((a.astype(np.uint32) << np.uint32(16)) | b.astype(np.uint32)).view(np.float32)
            y = (x + c).astype(np.float32)
        else:                            # t_mul_bf16_exact
            y = (fa * fb).astype(np.float32)
        return cbits, y.view(np.uint32).astype(np.uint64), 32
    cbits = _bf16_bits(c64.astype(np.float32))
    fc = _bf16_val(cbits).astype(np.float64)
    if test == "t_bf16acc_sep":          # product rounded to bf16, then bf16 add (two roundings)
        p = _bf16_val(_round_f64_to_bf16_bits(fa * fb)).astype(np.float64)
        y = _round_f64_to_bf16_bits(p + fc)  # f64 sum exact for the bounded exponent range used
    elif test == "t_bf16acc_fma":
        y = _round_f64_to_bf16_bits(fa * fb + fc)
    elif test == "t_add_bf16":
        y = _round_f64_to_bf16_bits(fa + fc)
    elif test == "t_mul_bf16":
        y = _round_f64_to_bf16_bits(fa * fb)
    else:
        raise KeyError(test)
    return cbits, y, 16


def functional_check(build: Path, test: str, nvec: int, seed: int) -> dict:
    s = read_cmd() + f"synth -flatten -top {test}\nopt_clean -purge\nwrite_json {test}.net.json\n"
    yosys(build, s, test)
    net = json.loads((build / f"{test}.net.json").read_text())
    rng = np.random.default_rng(seed)
    a, b, c64 = make_vectors(rng, nvec)
    cbits, yref, width = reference(test, a, b, c64)
    out = netlist_eval(net, dict(a=a, b=b, c=cbits), nvec)["y"]
    if width == 32:
        isnan_ref = np.isnan(yref.astype(np.uint32).view(np.float32))
        isnan_out = np.isnan(out.astype(np.uint32).view(np.float32))
    else:
        isnan_ref = np.isnan(_bf16_val(yref))
        isnan_out = np.isnan(_bf16_val(out))
    ok = (out == yref) | (isnan_ref & isnan_out)
    bad = np.nonzero(~ok)[0][:5].tolist()
    return dict(vectors=nvec, mismatches=int((~ok).sum()), first_bad=[
        dict(a=hex(int(a[i])), b=hex(int(b[i])), c=hex(int(cbits[i])), ref=hex(int(yref[i])), got=hex(int(out[i])))
        for i in bad])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps", default=os.environ.get("HW_DEPS", "/tmp/hwdeps"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--nvec", type=int, default=4000)
    ap.add_argument("--keep-build", action="store_true")
    ap.add_argument("--no-patch", action="store_true", help="do not patch HardFloat (reproduces the bf16 bug)")
    ap.add_argument("--out", default=str(REPO / "results/hw/synthesis.json"))
    args = ap.parse_args()
    build = Path(tempfile.mkdtemp(prefix="hwsynth_", dir="/tmp"))
    try:
        prepare_build(Path(args.deps), build, patch_hardfloat=not args.no_patch)
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {d: ex.submit(synth_design, build, d) for d in DESIGNS}
            tfuts = {t: ex.submit(functional_check, build, t, args.nvec, 1234 + i) for i, t in enumerate(TESTS)}
            designs = {d: f.result() for d, f in futs.items()}
            checks = {t: f.result() for t, f in tfuts.items()}
        ver = subprocess.run([YOSYS, "-V"], capture_output=True, text=True).stdout.strip()
    finally:
        if not args.keep_build:
            shutil.rmtree(build, ignore_errors=True)

    def ratio(x, y, key):
        return designs[x][key]["area_um2"] / designs[y][key]["area_um2"]

    ratios = {}
    for key in LIBS:
        ratios[key] = {
            "adder fp32/bf16": ratio("add_fp32", "add_bf16", key),
            "mult fp32/bf16(rounded)": ratio("mul_fp32", "mul_bf16", key),
            "mult bf16-exact->fp32 / bf16(rounded)": ratio("mul_bf16_exact_fp32", "mul_bf16", key),
            "fma bf16in-fp32acc / bf16": ratio("fma_fp32_bf16in", "fma_bf16", key),
            "PE sep fp32acc/bf16acc": ratio("pe_fp32acc_sep", "pe_bf16acc_sep", key),
            "PE fma fp32acc/bf16acc": ratio("pe_fp32acc_fma", "pe_bf16acc_fma", key),
        }
    res = dict(
        tool=ver,
        libraries=dict(nangate45="NangateOpenCellLibrary typical (FreePDK45, 45 nm), from OpenROAD-flow-scripts",
                       sky130hd="sky130_fd_sc_hd tt_025C_1v80 (SkyWater 130 nm), from OpenROAD-flow-scripts"),
        hardfloat_patch=([dict(file=f, old=o.strip(), new=n.strip()) for f, o, n in HF_PATCHES]
                         if not args.no_patch else "none"),
        rtl="scripts/hw/rtl/mac_units.v on Berkeley HardFloat Release 1 (RISCV specialization)",
        notes=("logic synthesis only: no placement/routing, no wire load, timing not verified; "
               "default yosys/ABC liberty mapping (area-oriented, no delay target); "
               "rounding mode fixed RNE (constant), full IEEE subnormal/Inf/NaN support"),
        designs=designs, ratios=ratios, functional_check=checks,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"{'design':22s} {'gen.gates':>9s} {'FF':>4s} {'N45 um2':>9s} {'(seq)':>7s} {'S130 um2':>9s} {'(seq)':>7s}")
    for d, v in designs.items():
        a, t = v["nangate45"], v["sky130hd"]
        print(f"{d:22s} {v['generic']['cells']:9d} {a['flipflops']:4d} {a['area_um2']:9.1f} "
              f"{a['sequential_area_um2']:7.1f} {t['area_um2']:9.1f} {t['sequential_area_um2']:7.1f}")
    for k, r in ratios.items():
        print(k, {n: round(x, 3) for n, x in r.items()})
    for t, c in checks.items():
        print(t, c["vectors"], "vectors, mismatches:", c["mismatches"], c["first_bad"][:2])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
