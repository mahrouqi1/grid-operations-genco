#!/usr/bin/env bash
# Rebuild the three post figures from the committed result files, into this directory:
#   fig_a_voltage_inference_v35.png   voltage inference against the interpolators and the WLS state estimator
#   fig_b_transfer_12k.png            12k-update contrast of the continued-pretraining runs
#   fig_c_unlearning.png              collapse reproduction versus updates, with held-out error
# The scripts write into a scratch "new/" folder next to themselves; this runner copies the PNGs here and removes the scratch files,
# so a run leaves no extra file in the repo.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
R=$(cd ../.. && pwd)   # the results/ directory
HERE=$(pwd)

# b and c: make_figs_v3.py also builds the older figure (a) and before/after panels from files that are not shipped,
# so call only its two functions (importing the module only defines them and creates its output folders).
(cd "$R/2026-10-24-figure-redesign" && "$PY" -c "import make_figs_v3 as m; m.fig_b(); m.fig_c()")
cp "$R/2026-10-24-figure-redesign/new/fig_b_transfer_12k.png" "$HERE/"
cp "$R/2026-10-24-figure-redesign/new/fig_c_unlearning.png" "$HERE/"
rm -rf "$R/2026-10-24-figure-redesign/new" "$R/2026-10-24-figure-redesign/compare"

# a: reads the verdict and cell files of the voltage-inference and WLS result directories
(cd "$R/2026-10-25-figure-a-v35" && "$PY" make_fig_a_v35.py)
cp "$R/2026-10-25-figure-a-v35/new/fig_a_voltage_inference_v35.png" "$HERE/"
rm -rf "$R/2026-10-25-figure-a-v35/new" "$R/2026-10-25-figure-a-v35/plotted_numbers_v35.json"
