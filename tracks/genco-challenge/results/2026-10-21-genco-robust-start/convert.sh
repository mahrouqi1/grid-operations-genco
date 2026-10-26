#!/bin/bash
cd <REPO> || exit 2
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=core:tracks/powerflow-surrogate/src:tracks/genco-challenge/src
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
OUT=data/external/genco-robust-start
$PY -m genco_challenge.ladder_to_datakit --cases case2869pegase pglib_opf_case8387_pegase case9241pegase pglib_opf_case2383wp_k pglib_opf_case3012wp_k pglib_opf_case3375wp_k --n 100 --seed 0 --out $OUT --workers 12 > $OUT/convert.log 2>&1
echo "rc=$?" > $OUT/CONVERT_DONE
