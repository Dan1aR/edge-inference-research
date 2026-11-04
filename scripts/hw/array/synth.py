#!/usr/bin/env python
"""E7 / B: logic synthesis of the PEs, edge units and full R x R weight-stationary arrays of every
accumulator variant (Yosys 0.70 / yowasp-yosys + ABC, Nangate45 typical and sky130hd tt liberty).

Per variant v:
  pe_ws_<v>   : weight-stationary PE (flattened)                      -> mapped netlists (.v, .json)
  pe_os_<v>   : output-stationary PE (acc register in the PE)          (not for chunk / *_sep)
  edge_top_<v>: per-column top-edge converters (weight, psum-in)        edge_west: per-row a converter
  edge_bot_<v>: per-column bottom-edge unit (psum-out converter; chunk: fp32 adder + 32b register)
  arr<R>_<v>  : ws_array #(R,R,VAR) synthesised hierarchically (PE / edge_chunk kept as modules)
ABC mapping: default (area-oriented) script; timing is closed later by OpenROAD (pnr.py).
Mapped Nangate45 netlists go to hwtools/build/synth/ (used by sta_power.py, pnr.py, gatesim.py).

  nice -n 19 .venv/bin/python scripts/hw/array/synth.py [--jobs 3] [--arrays 16,32]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ACC_REC, BUILD, DONT_USE, LIBS, RESULTS, SEP_VARIANTS, VAR_ID, VARIANTS, YOSYS,  # noqa: E402
                    iw_bits, prepare_hf, read_cmd, sw_bits, yosys)

NAND2 = {"nangate45": 0.798, "sky130hd": 3.7536}
DELAY_TARGET_PS = 1000


def edge_wrappers(var: str) -> str:
    AE, AS = ACC_REC[var]
    IW, SW = iw_bits(var), sw_bits(var)
    if var == "chunk":
        top = f"""module edge_top_{var} (input [15:0] w, output [16:0] rw);
    fNToRecFN#(8, 8) cw (w, rw);
endmodule
module edge_bot_{var} (input clk, input [16:0] ps, input [31:0] acc_in, output reg [31:0] acc_out);
    wire [32:0] ra, sum; wire [31:0] o; wire [4:0] f;   // = edge_chunk of acc_array.v, flattened
    fNToRecFN#(8, 24) ci (acc_in, ra);
    addRecFN#(8, 24) ad (`CTL, 1'b0, ra, {{ps, 16'b0}}, 3'b000, sum, f);
    recFNToFN#(8, 24) co (sum, o);
    always @(posedge clk) acc_out <= o;
endmodule
"""
    elif var == "kahan":
        top = f"""module edge_top_{var} (input [15:0] w, input [31:0] ps, output [16:0] rw, output [33:0] rps);
    fNToRecFN#(8, 8) cw (w, rw);
    fNToRecFN#(8, 8) cs (ps[31:16], rps[33:17]);
    fNToRecFN#(8, 8) cc (ps[15:0], rps[16:0]);
endmodule
module edge_bot_{var} (input [33:0] rps, output [31:0] ps);
    recFNToFN#(8, 8) os (rps[33:17], ps[31:16]);
    recFNToFN#(8, 8) oc (rps[16:0], ps[15:0]);
endmodule
"""
    else:
        top = f"""module edge_top_{var} (input [15:0] w, input [{IW-1}:0] ps, output [16:0] rw, output [{SW-1}:0] rps);
    fNToRecFN#(8, 8) cw (w, rw);
    fNToRecFN#({AE}, {AS}) cs (ps, rps);
endmodule
module edge_bot_{var} (input [{SW-1}:0] rps, output [{IW-1}:0] ps);
    recFNToFN#({AE}, {AS}) os (rps, ps);
endmodule
"""
    return top


def array_wrapper(var: str, R: int) -> str:
    IW = iw_bits(var)
    OW = 32 if var == "chunk" else IW
    return (f"module arr{R}_{var} (input clk, input wload, input [{R*16-1}:0] a_in, input [{R*16-1}:0] w_in,\n"
            f"    input [{R*IW-1}:0] ps_in, input [{R*32-1}:0] acc_in, output [{R*OW-1}:0] ps_out);\n"
            f"    ws_array #({R}, {R}, {VAR_ID[var]}) u (clk, wload, a_in, w_in, ps_in, acc_in, ps_out);\nendmodule\n")


def map_script(top: str, tag: str, lib_name: str, flavour: str, write: bool) -> str:
    lib = LIBS[lib_name].name
    du = "".join(f" -dont_use {d}" for d in DONT_USE[lib_name])
    abc = f"abc -liberty {lib}{du}" + (f" -D {DELAY_TARGET_PS}" if flavour == "delay" else "")
    s = "design -load gen\n"
    s += f"dfflibmap -liberty {lib}{du}\n{abc}\nopt_clean -purge\n"
    if lib_name == "nangate45":
        s += "hilomap -singleton -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z\nsetundef -zero\n"
        s += "insbuf -buf BUF_X1 A Z\nopt_clean -purge\nrename -enumerate\n"
    s += f"tee -q -o {tag}.{lib_name}.{flavour}.json stat -liberty {lib} -json -top {top}\n"
    if write:
        s += f"write_verilog -noattr -noexpr -nohex -nodec {tag}.{lib_name}.{flavour}.v\n"
        s += f"write_json {tag}.{lib_name}.{flavour}.net.json\n"
    return s


def synth_block(build: Path, tag: str, hier_cmd: str, flatten: bool = True, flavours=("area",),
                libs=("nangate45", "sky130hd")) -> dict:
    s = read_cmd() + "read_verilog -I. edges.v\n" + hier_cmd
    s += f"synth {'-flatten ' if flatten else ''}-top {tag}\nopt_clean -purge\n"
    s += f"tee -q -o {tag}.gen.json stat -json\ndesign -save gen\n"
    for ln in libs:
        for fl in flavours:
            s += map_script(tag, tag, ln, fl, write=(ln == "nangate45"))
    yosys(build, s, tag)
    out = {}
    gen = json.loads((build / f"{tag}.gen.json").read_text())
    gm = gen["modules"]
    top = gm.get(f"\\{tag}", gm.get(tag, list(gm.values())[0]))
    out["generic_cells"] = int(top.get("num_cells", 0))
    for ln in libs:
        for fl in flavours:
            j = json.loads((build / f"{tag}.{ln}.{fl}.json").read_text())
            mods = j["modules"]
            m = mods.get(f"\\{tag}", mods.get(tag))
            if m is None:
                m = list(mods.values())[0]
            # hierarchical designs: yosys 'stat' top entry already sums the hierarchy when 'design' given
            d = j.get("design", m)
            cells = {k: v for k, v in d.get("num_cells_by_type", {}).items() if not k.startswith("$")}
            ff = sum(v for k, v in cells.items() if k.startswith(("DFF", "SDFF")) or "__df" in k)
            area = float(d["area"])
            seq = float(d.get("sequential_area", 0.0))
            out[f"{ln}_{fl}"] = dict(area_um2=area, sequential_area_um2=seq, combinational_area_um2=area - seq,
                                    cells=int(sum(cells.values())), flipflops=int(ff), nand2_eq=area / NAND2[ln])
    return out


def pe_cmd(kind: str, var: str, tag: str) -> str:
    return (f"hierarchy -top pe_{kind} -chparam VAR {VAR_ID[var]} -chparam SW {sw_bits(var)}\n"
            f"rename -top {tag}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--vars", default=",".join(VARIANTS + SEP_VARIANTS))
    ap.add_argument("--arrays", default="16,32")
    ap.add_argument("--out", default=str(RESULTS / "synth.json"))
    ap.add_argument("--only", default="", help="comma list of block names to (re)run; merged into --out")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    build = prepare_hf(BUILD / "synth")
    vars_ = args.vars.split(",")
    (build / "edges.v").write_text("`include \"HardFloat_consts.vi\"\n`define CTL `flControl_tininessAfterRounding\n"
                                   "module edge_west (input [15:0] a, output [16:0] r);\n"
                                   "    fNToRecFN#(8, 8) u (a, r);\nendmodule\n"
                                   + "".join(edge_wrappers(v) for v in vars_ if v in VARIANTS)
                                   + "".join(array_wrapper(v, int(R)) for v in vars_ if v in VARIANTS
                                             for R in args.arrays.split(",") if R))
    jobs = {}

    class _Skip:
        def __init__(self, ex):
            self.ex = ex

        def submit(self, fn, b, tag, *a):
            return self.ex.submit(fn, b, tag, *a) if (not only or tag in only) else None

    with ThreadPoolExecutor(max_workers=args.jobs) as ex0:
        ex = _Skip(ex0)
        jobs["edge_west"] = ex.submit(synth_block, build, "edge_west", "hierarchy -top edge_west\n")
        for v in vars_:
            jobs[f"pe_ws_{v}"] = ex.submit(synth_block, build, f"pe_ws_{v}", pe_cmd("ws", v, f"pe_ws_{v}"))
            if v in VARIANTS and v != "chunk":
                jobs[f"pe_os_{v}"] = ex.submit(synth_block, build, f"pe_os_{v}", pe_cmd("os", v, f"pe_os_{v}"))
            if v in VARIANTS:
                for e in ("top", "bot"):
                    jobs[f"edge_{e}_{v}"] = ex.submit(synth_block, build, f"edge_{e}_{v}",
                                                      f"hierarchy -top edge_{e}_{v}\n")
                for R in [int(x) for x in args.arrays.split(",") if x]:
                    tag = f"arr{R}_{v}"
                    cmd = f"hierarchy -top {tag}\n"
                    jobs[tag] = ex.submit(synth_block, build, tag, cmd, True, ("area",), ("nangate45",))
        res = json.loads(Path(args.out).read_text())["blocks"] if only and Path(args.out).exists() else {}
        for k, f in jobs.items():
            if f is None:
                continue
            res[k] = f.result()
            r = res[k].get("nangate45_area", {})
            print(f"{k:22s} n45 area {r.get('area_um2', 0):12.1f} um2  seq {r.get('sequential_area_um2', 0):10.1f}"
                  f"  cells {r.get('cells', 0):8d} FF {r.get('flipflops', 0):6d}", flush=True)
    ver = subprocess.run([YOSYS, "-V"], capture_output=True, text=True).stdout.strip()
    # composed array totals from blocks (cross-check of the hierarchical array synthesis)
    comp = {}
    for v in vars_:
        if v not in VARIANTS:
            continue
        for R in [int(x) for x in args.arrays.split(",") if x]:
            a = lambda k: res[k]["nangate45_area"]["area_um2"]  # noqa: E731
            comp[f"arr{R}_{v}"] = R * R * a(f"pe_ws_{v}") + R * a("edge_west") + R * (a(f"edge_top_{v}") + a(f"edge_bot_{v}"))
    out = dict(tool=ver, delay_target_ps=DELAY_TARGET_PS,
               notes="logic synthesis only (no wires); 'area' = default ABC area script, 'delay' = abc -D target",
               blocks=res, composed_array_area_um2=comp)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
