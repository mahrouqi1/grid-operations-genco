#!/bin/bash
# Family-strict ladder -> datakit conversion, 26 cases, per-case counts from n_map.json (SimBench 2000, GOC 1000, EPIGRIDS 6000, RTE 500). CPU only. Launch:
#   nohup nice -n 10 bash tracks/genco-challenge/results/2026-10-20-ladder-to-datakit/run_full.sh > /dev/null 2>&1 &
cd <REPO> || exit 2
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=core:tracks/powerflow-surrogate/src:tracks/genco-challenge/src
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
OUT=data/external/genco-ladder-datakit
mkdir -p "$OUT"
rm -f "$OUT/CONVERT_DONE" "$OUT/CONVERT_FAILED"
D=tracks/genco-challenge/results/2026-10-20-ladder-to-datakit
CASES=$(cat $D/cases.txt)
echo "start $(date -Is) pid $$" > "$OUT/convert.log"
$PY -m genco_challenge.ladder_to_datakit --cases $CASES --n 2000 --n-map $D/n_map.json --seed 0 --out "$OUT" --workers 16 >> "$OUT/convert.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then date -Is > "$OUT/CONVERT_DONE"; else echo "rc=$rc $(date -Is) (see convert.log for FAILED lines)" > "$OUT/CONVERT_FAILED"; fi
