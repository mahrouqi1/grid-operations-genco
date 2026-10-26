#!/bin/bash
# GENCO section 5.5.3 sanity reproduction: evaluate the released case118 checkpoints on GENCO's own
# 9,952-scenario eval set, exactly as the repro guide prescribes. Published (transfer_case118.csv):
# finetune@1000 1.932 MW, scratch@1000 3.810 MW, zero-shot 11.078 MW, DC-PF 2.295 MW (mean |P| residual).
# Env: grid-operations-genco. Run from the repo root.
set -euo pipefail
REPO=$(pwd)
GK=$REPO/data/external/gridfm-graphkit-repro
T=$REPO/data/external/genco-pf-transfer
OUT=$REPO/tracks/genco-challenge/results/2026-10-06-genco-553-sanity
export MLFLOW_ALLOW_FILE_STORE=true PYTHONNOUSERSITE=1
cd $GK
run() {  # name model [normalizer]
  local extra=()
  [ -n "${3:-}" ] && extra=(--normalizer_stats "$3")
  gridfm_graphkit evaluate --config scripts/pretraining_advantage/case118_eval.yaml \
    --data_path "$T/data" --model_path "$2" "${extra[@]}" --compute_dc_ac_metrics \
    --num_workers 8 --log_dir "$OUT/mlruns" --run_name "$1" > "$OUT/$1.log" 2>&1
}
run finetune_1000 $T/finetune/case118_1000/best_model_state_dict.pt $T/finetune/case118_1000/normalizer_stats.pt
run scratch_1000  $T/scratch/case118_1000/best_model_state_dict.pt  $T/scratch/case118_1000/normalizer_stats.pt
run zeroshot      $T/pretraining/best_model_state_dict.pt

# metrics.json: pull the two metric CSVs of each run's newest mlruns directory (audit genco-eval R1-F9).
python - "$OUT" <<'PY'
import csv, glob, json, sys
from pathlib import Path
out = Path(sys.argv[1])
res = {}
for name in ("finetune_1000", "scratch_1000", "zeroshot"):
    runs = [Path(p).parent.parent for p in glob.glob(str(out / "mlruns/*/*/tags/mlflow.runName"))
            if Path(p).read_text().strip() == name]
    art = max(runs, key=lambda p: p.stat().st_mtime) / "artifacts/test"
    rd = lambda f: {r["Metric"]: float(r["Value"]) for r in csv.DictReader((art / f).open())}
    res[name] = {"model": rd("case118_ieee_metrics.csv"), "reference_and_dc": rd("case118_ieee_ac_dc_metrics.csv")}
(out / "metrics.json").write_text(json.dumps(res, indent=1))
PY
