#!/bin/bash
# Stage 3b: stage-3 (finish2.sh) ran 3022 and the epigrids in parallel by hand to save wall time; this waits for the
# GOC invocation to end, stops finish2.sh before it repeats those, then runs rate_a := .m, verify, marker.
cd <REPO> || exit 2
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=core:tracks/powerflow-surrogate/src:tracks/genco-challenge/src
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
OUT=data/external/genco-ladder-datakit; D=tracks/genco-challenge/results/2026-10-20-ladder-to-datakit
while kill -0 $(cat /tmp/pgoc.pid) 2>/dev/null; do sleep 5; done
F2=$(head -1 /tmp/pfinish2.pid)
kill -STOP $F2 2>/dev/null    # freeze the script so it cannot start another step
pkill -P $F2 2>/dev/null      # kill what it just launched (its own 3022 run), never the hand-started one
kill -9 $F2 2>/dev/null
sleep 1
while kill -0 $(cat /tmp/p3022.pid) 2>/dev/null; do sleep 5; done
rm -f $OUT/CONVERT_DONE $OUT/CONVERT_FAILED
$PY -m genco_challenge.pglib_rate_a_fix $OUT > $OUT/rate_fix.log 2>&1 || rc=1
$PY $D/verify_tree.py > $OUT/verify.log 2>&1 || rc=1
if [ -z "$rc" ]; then date -Is > $OUT/CONVERT_DONE; else echo "stage3b failed $(date -Is)" > $OUT/CONVERT_FAILED; fi
