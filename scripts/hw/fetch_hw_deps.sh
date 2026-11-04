#!/usr/bin/env bash
# E6: download open-source hardware dependencies for scripts/hw/run_synth.py (CPU only, ~20 MB).
#   * Berkeley HardFloat Release 1 (Verilog, BSD-3)  http://www.jhauser.us/arithmetic/HardFloat.html
#   * Nangate/FreePDK45 Open Cell Library typical liberty and SkyWater sky130_fd_sc_hd tt liberty
#     (both as shipped with OpenROAD-flow-scripts, ~20 MB)
#   * yowasp-yosys (Yosys 0.70 compiled to WebAssembly, pip package, no system install)
set -euo pipefail
DEPS=${HW_DEPS:-/tmp/hwdeps}
PY=${PY:-/root/workspace/paper/edge-inference-research/.venv/bin/python}
mkdir -p "$DEPS"
cd /tmp
if ! "$PY" -c "import yowasp_yosys" 2>/dev/null; then
  uv pip install --no-config -p "$PY" --index-url https://pypi.org/simple yowasp-yosys
fi
if [ ! -d "$DEPS/HardFloat-1" ]; then
  curl -sSL -o "$DEPS/HardFloat-1.zip" http://www.jhauser.us/arithmetic/HardFloat-1.zip
  "$PY" -I -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$DEPS/HardFloat-1.zip" "$DEPS"
fi
if [ ! -f "$DEPS/nangate45_typ.lib" ]; then
  curl -sSL -o "$DEPS/nangate45_typ.lib" \
    https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
fi
if [ ! -f "$DEPS/sky130hd_tt.lib" ]; then
  curl -sSL -o "$DEPS/sky130hd_tt.lib" \
    https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
fi
sha256sum "$DEPS/HardFloat-1.zip" "$DEPS/nangate45_typ.lib" "$DEPS/sky130hd_tt.lib"
echo "deps in $DEPS"
