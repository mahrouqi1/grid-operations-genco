#!/bin/bash
# Second stage: re-convert the 5 GOC cases that were rate-fixed before the tie-break improvement, convert
# case3022 with relaxed label tolerance (its OPF dispatch reproduces to 1.8e-6 pu, not 1e-6), then set rate_a
# of every pglib_opf_* case to the PGLib .m values and write the DONE marker.
cd <REPO> || exit 2
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=core:tracks/powerflow-surrogate/src:tracks/genco-challenge/src
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
OUT=data/external/genco-ladder-datakit; D=tracks/genco-challenge/results/2026-10-20-ladder-to-datakit
rm -f $OUT/CONVERT_FAILED
$PY -m genco_challenge.ladder_to_datakit --cases pglib_opf_case500_goc pglib_opf_case793_goc pglib_opf_case2000_goc pglib_opf_case2312_goc pglib_opf_case2742_goc --n 1000 --out $OUT --workers 16 >> $OUT/convert.log 2>&1 || rc=1
$PY -m genco_challenge.ladder_to_datakit --cases pglib_opf_case3022_goc --n 1000 --tol-vm 1e-5 --tol-va 1e-3 --out $OUT --workers 16 >> $OUT/convert.log 2>&1 || rc=1
$PY -m genco_challenge.pglib_rate_a_fix $OUT > $OUT/rate_fix.log 2>&1 || rc=1
if [ -z "$rc" ]; then date -Is > $OUT/CONVERT_DONE; else echo "stage2 failed $(date -Is)" > $OUT/CONVERT_FAILED; fi
