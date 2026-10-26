#!/bin/bash
# Stage 3: reconvert the 15 PGLib cases with datakit-faithful generator semantics (every .m gen a gen row, loads-only
# Pd), then rate_a := .m, verify the whole tree, and only then write CONVERT_DONE.
cd <REPO> || exit 2
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=core:tracks/powerflow-surrogate/src:tracks/genco-challenge/src
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
OUT=data/external/genco-ladder-datakit; D=tracks/genco-challenge/results/2026-10-20-ladder-to-datakit
rm -f $OUT/CONVERT_DONE $OUT/CONVERT_FAILED
echo "=== stage 3 (pglib .m gens) $(date -Is)" >> $OUT/convert.log
GOC="pglib_opf_case500_goc pglib_opf_case793_goc pglib_opf_case2000_goc pglib_opf_case2312_goc pglib_opf_case2742_goc pglib_opf_case3970_goc pglib_opf_case4020_goc pglib_opf_case4601_goc pglib_opf_case4619_goc pglib_opf_case4837_goc pglib_opf_case4917_goc pglib_opf_case9591_goc"
$PY -m genco_challenge.ladder_to_datakit --cases $GOC --n 1000 --out $OUT --workers 16 >> $OUT/convert.log 2>&1 || rc=1
$PY -m genco_challenge.ladder_to_datakit --cases pglib_opf_case3022_goc --n 1000 --tol-vm 1e-5 --tol-va 1e-3 --out $OUT --workers 16 >> $OUT/convert.log 2>&1 || rc=1
$PY -m genco_challenge.ladder_to_datakit --cases pglib_opf_case5658_epigrids pglib_opf_case7336_epigrids --n 6000 --out $OUT --workers 16 >> $OUT/convert.log 2>&1 || rc=1
$PY -m genco_challenge.pglib_rate_a_fix $OUT > $OUT/rate_fix.log 2>&1 || rc=1
$PY $D/verify_tree.py > $OUT/verify.log 2>&1 || rc=1
if [ -z "$rc" ]; then date -Is > $OUT/CONVERT_DONE; else echo "stage3 failed $(date -Is) (see convert.log, rate_fix.log, verify.log)" > $OUT/CONVERT_FAILED; fi
