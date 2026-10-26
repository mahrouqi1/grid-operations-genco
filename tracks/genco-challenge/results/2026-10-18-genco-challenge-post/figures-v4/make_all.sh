#!/usr/bin/env bash
# Rebuild the two post figures from the committed result files, into this directory:
#   fig1_voltage_inference_v4.png   voltage inference: GENCO error over best interpolator error, per grid (post figure 2)
#   fig2_corrected_labels_v4.png    collapse reproduction versus updates, corrected versus original labels (post figure 1)
# The scripts write into a scratch "new/" folder next to themselves and re-write their plotted-number files (identical bytes);
# each figure is checked against its source files by a number checker, then this runner copies the PNGs here and removes the
# scratch folder, so a run leaves no extra file in the repo.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
R=$(cd ../.. && pwd)   # the results/ directory
HERE=$(pwd)
D="$R/2026-10-26-figures-v4"

(cd "$D" && "$PY" make_fig1_v4.py && "$PY" make_fig2_v4.py && "$PY" check_numbers_v4.py && "$PY" check_numbers_fig2_v4.py)
cp "$D/new/fig1_voltage_inference_v4.png" "$HERE/"
cp "$D/new/fig2_corrected_labels_v4.png" "$HERE/"
rm -rf "$D/new"
