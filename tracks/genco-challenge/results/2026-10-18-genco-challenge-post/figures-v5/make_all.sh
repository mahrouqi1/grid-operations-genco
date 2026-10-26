#!/usr/bin/env bash
# Rebuild the three post figures from the committed result files, into this directory:
#   fig_sweep_v5.png                share of grids where GENCO beats the best interpolator against the measured fraction (post figure 1)
#   fig_voltage_v5.png              voltage inference: GENCO error over best interpolator error per grid at 40% measured (post figure 2)
#   fig_corrected_v51.png           collapse reproduction versus updates, corrected versus original labels (post figure 3)
# The scripts write into scratch "new/" folders next to themselves and re-write their plotted-number files (identical bytes);
# the voltage and corrected-label figures are checked against their source files by number checkers, then this runner copies the
# PNGs here and removes the scratch folders, so a run leaves no extra file in the repo.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
R=$(cd ../.. && pwd)   # the results/ directory
HERE=$(pwd)
D5="$R/2026-10-26-figures-v5"

(cd "$D5" && "$PY" make_fig_sweep_v5.py && "$PY" make_fig_voltage_v5.py && "$PY" make_fig_corrected_v51.py && "$PY" check_numbers_voltage_v5.py && "$PY" check_numbers_corrected_v51.py)
cp "$D5/new/fig_sweep_v5.png" "$HERE/"
cp "$D5/new/fig_voltage_v5.png" "$HERE/"
cp "$D5/new/fig_corrected_v51.png" "$HERE/"
rm -rf "$D5/new"
