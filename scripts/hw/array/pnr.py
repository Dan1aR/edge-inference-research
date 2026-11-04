#!/usr/bin/env python
"""E7 / B: place & route of the PEs (and edge units) with OpenROAD (litex-hub conda build) on Nangate45,
following the OpenROAD-flow-scripts nangate45 platform settings (tracks, tapcells, PDN, RC, site).

Flow per block: floorplan (utilisation U) -> IO pins -> tapcells -> PDN -> global placement ->
repair_design -> detailed placement -> CTS -> repair_timing (setup) -> global route -> parasitics from
global route (or detailed route + OpenRCX extraction with --droute) -> STA (propagated clock) ->
power with the per-pin switching activity of power_sim.py (zero-delay and unit-delay) on real operands.

The clock target is given per run; reported: area (cell area, core area), WNS, achieved clock period
(= target - WNS when negative), and internal/switching/leakage power per matmul activity set.

  nice -n 19 .venv/bin/python scripts/hw/array/pnr.py --blocks pe_ws_fp32,pe_ws_bf16 --period 2.0
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, ENV, LIBS, PLATFORM, RESULTS  # noqa: E402

SYN = BUILD / "synth"


def activity_tcl(act_json: Path, section: str, mode: str, out: Path) -> dict:
    """Per-pin set_power_activity commands from a power_sim.py activity file."""
    d = json.loads(act_json.read_text())[section]
    cyc = d["cycles"] if mode == "zero" else d["cycles_unit_delay"]
    lines = []
    tot = 0.0
    for n in d["nets"]:
        tog = n["tog0"] if mode == "zero" else n["togu"]
        a = tog / cyc
        duty = min(max(n["ones"] / d["cycles"], 0.0), 1.0)
        tot += a
        if n["pin"]:
            lines.append(f"set_power_activity -pins {{{n['pin']}}} -activity {a:.6g} -duty {duty:.4f}")
        elif n["port"] and n["port"] != "clk":
            lines.append(f"set_power_activity -input_ports {{{n['port']}}} -activity {a:.6g} -duty {duty:.4f}")
    out.write_text("\n".join(lines) + "\n")
    return dict(cycles=cyc, toggles_per_cycle=tot)


def flow_tcl(block: str, period: float, util: float, droute: bool, power_sets: list[tuple[str, Path]]) -> str:
    small = block.startswith("edge_")      # edge units: too small for the ORFS PDN straps -> no PDN
    lib = LIBS["nangate45"]
    P = PLATFORM
    s = f"""
set_thread_count 1
read_lef {P}/lef/NangateOpenCellLibrary.tech.lef
read_lef {P}/lef/NangateOpenCellLibrary.macro.mod.lef
read_liberty {lib}
read_verilog {SYN}/{block}.nangate45.area.v
link_design {block}
if {{[llength [get_ports -quiet clk]]}} {{
  create_clock -name clk -period {period} [get_ports clk]
  set ins [delete_from_list [all_inputs] [get_ports clk]]
}} else {{
  create_clock -name clk -period {period}
  set ins [all_inputs]
}}
set_input_delay {0.1:.3f} -clock clk $ins
set_output_delay 0.0 -clock clk [all_outputs]
set_driving_cell -lib_cell DFF_X1 -pin Q $ins
set_load 3.0 [all_outputs]
initialize_floorplan -site FreePDK45_38x28_10R_NP_162NW_34O -utilization {util * 100:.0f} -aspect_ratio 1 -core_space 2
source {P}/make_tracks.tcl
place_pins -hor_layers metal3 -ver_layers metal2
tapcell -distance 120 -tapcell_master TAPCELL_X1 -endcap_master TAPCELL_X1
{"" if small else f"source {P}/grid_strategy-M1-M4-M7.tcl" + chr(10) + "pdngen"}
source {P}/setRC.tcl
set_dont_use {{TAPCELL_X1 FILLCELL_* AOI211_X1 OAI211_X1}}
global_placement -density {min(0.9, util + 0.1):.2f}
estimate_parasitics -placement
repair_design
repair_tie_fanout -separation 2 LOGIC0_X1/Z
repair_tie_fanout -separation 2 LOGIC1_X1/Z
detailed_placement
if {{[llength [all_registers]]}} {{
  clock_tree_synthesis -root_buf CLKBUF_X3 -buf_list {{CLKBUF_X1 CLKBUF_X2 CLKBUF_X3}} -sink_clustering_enable
  set_propagated_clock [all_clocks]
  estimate_parasitics -placement
  repair_clock_nets
}}
detailed_placement
estimate_parasitics -placement
repair_timing -setup -setup_margin 0.0
detailed_placement
check_placement
set_global_routing_layer_adjustment metal2-metal3 0.5
set_global_routing_layer_adjustment metal4-metal10 0.25
set_routing_layers -signal metal2-metal10 -clock metal4-metal10
global_route -allow_congestion
estimate_parasitics -global_routing
repair_timing -setup -setup_margin 0.0
global_route -allow_congestion
estimate_parasitics -global_routing
"""
    if droute:
        s += f"""
detailed_route -droute_end_iter 20 -verbose 0
extract_parasitics -ext_model_file {P}/rcx_patterns.rules
write_spef {BUILD}/pnr/{block}.spef
read_spef {BUILD}/pnr/{block}.spef
"""
    s += """
puts "RESULT wns [sta::worst_slack_cmd max]"
report_worst_slack -max -digits 4
report_tns -digits 4
report_design_area
report_check_types -max_slew -max_capacitance -violators
report_checks -path_delay max -digits 4 -format full_clock_expanded
"""
    for name, tcl in power_sets:
        s += f'puts "POWERSET {name}"\nsource {tcl}\nreport_power -digits 6\n'
    s += f"write_verilog {BUILD}/pnr/{block}.pnr.v\nwrite_def {BUILD}/pnr/{block}.def\n"
    return s


def parse(log: str) -> dict:
    r = {}
    m = re.search(r"worst slack (-?[\d.]+)", log)
    r["wns_ns"] = float(m.group(1)) if m else None
    m = re.search(r"tns (-?[\d.]+)", log)
    r["tns_ns"] = float(m.group(1)) if m else None
    m = re.search(r"Design area ([\d.]+) u\^2 ([\d.]+)% utilization", log)
    if m:
        r["cell_area_um2"] = float(m.group(1)); r["utilization_pct"] = float(m.group(2))
    r["power"] = {}
    for blk in log.split("POWERSET ")[1:]:
        name = blk.split("\n", 1)[0].strip()
        p = {}
        for grp in ("Sequential", "Combinational", "Clock", "Total"):
            m = re.search(rf"^{grp}\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", blk, re.M)
            if m:
                p[grp.lower()] = dict(internal_W=float(m.group(1)), switching_W=float(m.group(2)),
                                      leakage_W=float(m.group(3)), total_W=float(m.group(4)))
        r["power"][name] = p
    return r


def run_block(block: str, period: float, util: float, droute: bool, acts: list[Path], tag: str) -> dict:
    wd = BUILD / "pnr"
    wd.mkdir(parents=True, exist_ok=True)
    sets = []
    act_info = {}
    for a in acts:
        sec = "pe" if block.startswith("pe_") else ("edge_bottom" if "edge_bot" in block else
                                                     "edge_top" if "edge_top" in block else "edge_west")
        for mode in ("zero", "unit"):
            name = f"{a.stem}:{mode}"
            d = json.loads(a.read_text())[sec]
            if d["cycles_unit_delay" if mode == "unit" else "cycles"] == 0:
                continue
            tcl = wd / f"act_{block}_{a.stem}_{mode}.tcl"
            act_info[name] = activity_tcl(a, sec, mode, tcl)
            sets.append((name, tcl))
    script = wd / f"{block}{tag}.tcl"
    script.write_text(flow_tcl(block, period, util, droute, sets))
    t0 = time.time()
    r = subprocess.run(["nice", "-n", "19", str(ENV / "openroad"), "-no_init", "-exit", str(script)],
                       capture_output=True, text=True, cwd=wd)
    log = r.stdout + r.stderr
    (wd / f"{block}{tag}.log").write_text(log)
    res = parse(log)
    res.update(block=block, period_target_ns=period, utilization_target=util, detailed_route=droute,
               returncode=r.returncode, seconds=round(time.time() - t0, 1), activity=act_info)
    if res.get("wns_ns") is not None:
        res["achieved_period_ns"] = period - min(0.0, res["wns_ns"])
        res["fmax_MHz"] = 1e3 / res["achieved_period_ns"]
    for f in wd.glob(f"act_{block}_*.tcl"):
        f.unlink()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--period", type=float, default=2.0)
    ap.add_argument("--util", type=float, default=0.6)
    ap.add_argument("--droute", action="store_true")
    ap.add_argument("--acts", default=str(RESULTS / "activity"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or RESULTS / f"pnr{args.tag}.json")
    prev = json.loads(out.read_text()) if out.exists() else {}
    blocks = args.blocks.split(",")

    def acts_for(b):
        if b.startswith("pe_ws_"):
            return sorted(f for f in Path(args.acts).glob(f"ws_{b[6:]}_*.json") if ("_sep_" in f.name) == ("_sep" in b))
        if b.startswith("pe_os_"):
            return sorted(Path(args.acts).glob(f"os_{b[6:]}_*.json"))
        if b.startswith("edge_bot_") or b.startswith("edge_top_"):
            return sorted(f for f in Path(args.acts).glob(f"ws_{b[9:]}_*.json") if "_sep_" not in f.name)
        if b == "edge_west":
            return sorted(Path(args.acts).glob("ws_fp32_*.json"))
        return []

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {b: ex.submit(run_block, b, args.period, args.util, args.droute, acts_for(b), args.tag) for b in blocks}
        for b, f in futs.items():
            prev[b] = f.result()
            r = prev[b]
            print(b, {k: r.get(k) for k in ("wns_ns", "achieved_period_ns", "cell_area_um2", "seconds", "returncode")},
                  flush=True)
            out.write_text(json.dumps(prev, indent=2))


if __name__ == "__main__":
    main()
