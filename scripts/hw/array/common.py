"""E7 shared helpers: tool paths, HardFloat build copy (with the E6 bf16 patch), yosys runner."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HWTOOLS = Path(os.environ.get("HWTOOLS", REPO / "hwtools"))
ENV = HWTOOLS / "env" / "bin"                      # conda env: openroad, sta, iverilog (litex-hub)
DEPS = Path(os.environ.get("HW_DEPS", "/tmp/hwdeps"))  # scripts/hw/fetch_hw_deps.sh
PLATFORM = HWTOOLS / "platforms" / "nangate45"     # ORFS nangate45 platform files (LEF, RC, PDN)
BUILD = HWTOOLS / "build"
RESULTS = REPO / "results" / "hw" / "array"
YOSYS = str(Path(sys.executable).parent / "yowasp-yosys")
LIBS = {
    "nangate45": DEPS / "nangate45_typ.lib",
    "sky130hd": DEPS / "sky130hd_tt.lib",
}
DONT_USE = {
    "nangate45": ["TAPCELL_X1", "FILLCELL_*", "AOI211_X1", "OAI211_X1", "ANTENNA_*", "LOGIC0_*", "LOGIC1_*"],
    "sky130hd": ["sky130_fd_sc_hd__probe_p_*", "sky130_fd_sc_hd__probec_p_*", "sky130_fd_sc_hd__lpflow_*",
                 "sky130_fd_sc_hd__clkdlybuf*", "sky130_fd_sc_hd__dlygate*", "sky130_fd_sc_hd__dlymetal*"],
}
HF_FILES = ["HardFloat_primitives.v", "HardFloat_rawFN.v", "isSigNaNRecFN.v", "addRecFN.v", "mulRecFN.v",
            "mulAddRecFN.v", "fNToRecFN.v", "recFNToFN.v", "recFNToRecFN.v", "HardFloat_specialize.v"]
RTL = HERE / "rtl" / "acc_array.v"

# E6 patch (see scripts/hw/run_synth.py): alignment-shift width of addRecFN is too small for sigWidth=8;
# the same holds for sigWidth = 14 and 16 used here (clog2(14)=clog2(16)=4 < needed 5).
HF_PATCHES = [
    ("addRecFN.v", "    localparam alignDistWidth = clog2(sigWidth);\n",
     "    localparam alignDistWidth = clog2(sigWidth + 3);  // E6 patch, see scripts/hw/run_synth.py\n"),
    # portability only (Icarus rejects an unsized constant inside a concatenation); same value
    ("HardFloat_specialize.vi", "`define HardFloat_signDefaultNaN 0\n", "`define HardFloat_signDefaultNaN 1'b0\n"),
]

VARIANTS = ["fp32", "bf16", "fp16", "e8m13", "kahan", "chunk"]
SEP_VARIANTS = ["bf16_sep", "fp16_sep", "e8m13_sep"]
VAR_ID = {"fp32": 0, "bf16": 1, "fp16": 2, "e8m13": 3, "kahan": 4, "chunk": 5,
          "bf16_sep": 11, "fp16_sep": 12, "e8m13_sep": 13}
# (AE, AS) recoded accumulator format, IEEE width at the array edge
ACC_REC = {"fp32": (8, 24), "bf16": (8, 8), "fp16": (5, 11), "e8m13": (8, 14), "kahan": (8, 8),
           "chunk": (8, 8), "bf16_sep": (8, 8), "fp16_sep": (5, 11), "e8m13_sep": (8, 14)}


def sw_bits(var: str) -> int:
    return 34 if var == "kahan" else sum(ACC_REC[var]) + 1


def iw_bits(var: str) -> int:
    return 32 if var == "kahan" else sum(ACC_REC[var])


def prepare_hf(build: Path, patch: bool = True) -> Path:
    """Copy HardFloat (+RISC-V specialisation) and our RTL into build/, apply patches to the copy."""
    build.mkdir(parents=True, exist_ok=True)
    src = DEPS / "HardFloat-1" / "source"
    for f in list(src.glob("*.v*")) + list((src / "RISCV").glob("*")):
        shutil.copy(f, build / f.name)
    for fn, old, new in HF_PATCHES:
        if not patch and fn == "addRecFN.v":
            continue
        p = build / fn
        t = p.read_text()
        assert t.count(old) == 1, f"patch target not found in {fn}"
        p.write_text(t.replace(old, new))
    shutil.copy(RTL, build / RTL.name)
    for lib in LIBS.values():
        if not (build / lib.name).exists():
            shutil.copy(lib, build / lib.name)
    return build


def yosys(build: Path, script: str, tag: str, timeout: int | None = None) -> None:
    ys = build / f"{tag}.ys"
    ys.write_text(script)
    r = subprocess.run(["nice", "-n", "19", YOSYS, "-q", "-l", f"{tag}.log", "-s", ys.name], cwd=build,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (build / f"{tag}.log").read_text()[-3000:] if (build / f"{tag}.log").exists() else ""
        raise RuntimeError(f"yosys failed for {tag}:\n{tail}\n{r.stderr[-2000:]}")


def read_cmd() -> str:
    return "read_verilog -I. " + " ".join(HF_FILES + [RTL.name]) + "\n"
