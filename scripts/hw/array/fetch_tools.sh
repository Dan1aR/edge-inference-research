#!/usr/bin/env bash
# E7: open-source physical-design tools under hwtools/ (no system install, ~0.5 GB after pruning):
#   * OpenROAD 2.0-12381 + OpenSTA 2.5.0 + Icarus Verilog 13 (conda, channel litex-hub, + 'main' for deps)
#   * OpenROAD-flow-scripts nangate45 platform files (tech/cell LEF, tracks, PDN, RC, OpenRCX rules)
# HardFloat, liberty files and yowasp-yosys come from scripts/hw/fetch_hw_deps.sh (E6).
set -euo pipefail
cd "$(dirname "$0")/../../.."
HW=hwtools
export CONDA_PKGS_DIRS=$PWD/$HW/pkgs
if [ ! -x $HW/env/bin/openroad ]; then
  /root/miniconda3/condabin/conda create -y -q -p $PWD/$HW/env -c litex-hub -c main --override-channels \
      "openroad=2.0_12381_g01bba3695" "python=3.10" iverilog
  rm -rf $HW/pkgs $HW/env/include $HW/env/translations $HW/env/share/doc $HW/env/share/man
  find $HW/env -name "*.a" -delete; rm -f $HW/env/lib/libQt5WebEngine*
fi
P=$HW/platforms/nangate45; mkdir -p $P/lef $P/lib
B=https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/nangate45
for f in make_tracks.tcl setRC.tcl tapcell.tcl fastroute.tcl grid_strategy-M1-M4-M7.tcl rcx_patterns.rules config.mk; do
  [ -f $P/$f ] || curl -sSf -o $P/$f $B/$f; done
for f in NangateOpenCellLibrary.tech.lef NangateOpenCellLibrary.macro.mod.lef; do
  [ -f $P/lef/$f ] || curl -sSf -o $P/lef/$f $B/lef/$f; done
cp -n ${HW_DEPS:-/tmp/hwdeps}/nangate45_typ.lib $P/lib/NangateOpenCellLibrary_typical.lib || true
$HW/env/bin/openroad -version; $HW/env/bin/iverilog -V 2>&1 | head -1
