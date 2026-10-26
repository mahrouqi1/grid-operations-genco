#!/bin/bash
# Runs repairkit check on every local subgrid of data/external/genco-pf-pretraining (CPU only). Run from the repo root.
R=tracks/genco-challenge/results/2026-10-19-repairkit-check31
for d in data/external/genco-pf-pretraining/subnet_*; do
  g=$(basename $d); [ -f $R/$g.csv.summary.json ] && continue
  echo "== $g $(date +%T)"
  nice python -m genco_challenge.repairkit check $d/raw --out $R/$g.csv --workers 4 > $R/$g.log 2>&1 || echo "FAILED $g"
done
echo ALLDONE
