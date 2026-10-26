#!/bin/bash
# protocol_amendment_4.json: WLS uncapped (uncapped + flatuncapped, 40/60 iterations), f40_s0 only. CPU only, 1 thread per worker, nice 15.
# Usage: launch_v3.sh <worker-number>   (start 6 of them; see start_v3.sh). Workers claim lines of jobs_v3.txt in order via mkdir locks.
# Lines 1-102 = tier 1 (locked15, control, fvs; up to 100 scenarios). Lines 103-166 = tier 2 (32 whole grids, 30 scenarios).
# Output: per_case_v3/<arm>/<study>/<grid>.csv. Markers: TIER1_DONE / TIER2_DONE (only when every job of the tier finished before the deadline).
# Hard stop: Sunday 2026-10-25 12:00 EDT (--deadline on every call; the script default is in the past).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
DL="2026-10-25 12:00"; DLS=$(date -d "$DL" +%s); T1=102; TOTAL=$(grep -c . jobs_v3.txt)
mkdir -p locks_v3 ok_v3 logs_v3 per_case_v3
W=$1; i=0
marker() {
  n1=$(ls ok_v3 | awk -v t=$T1 '$1<=t' | wc -l); n2=$(ls ok_v3 | awk -v t=$T1 '$1>t' | wc -l)
  [ "$n1" -ge "$T1" ] && [ ! -e TIER1_DONE ] && date > TIER1_DONE
  [ "$n2" -ge $((TOTAL-T1)) ] && [ ! -e TIER2_DONE ] && date > TIER2_DONE
}
while read arm study grid lim cfgs; do
  i=$((i+1))
  [ "$(date +%s)" -ge "$DLS" ] && break
  mkdir "locks_v3/$i" 2>/dev/null || continue
  out="logs_v3/job_$i.log"
  nice -n 15 $PY run_wls_v2.py --arm $arm --study $study --grids $grid --limit $lim --configs $cfgs --deadline "$DL" \
      --out per_case_v3/$arm/$study/$grid.csv > "$out" 2>&1
  rc=$?
  cat "$out" >> logs_v3/${arm}_${study}_w$W.log
  if [ $rc -eq 0 ] && ! grep -q "deadline reached" "$out" && [ -s per_case_v3/$arm/$study/$grid.csv ]; then
    touch ok_v3/$i; marker
  else
    echo "job $i FAILED/INCOMPLETE rc=$rc: $arm $study $grid" >> logs_v3/failures.txt
  fi
done < jobs_v3.txt
echo "worker $W done $(date)" >> logs_v3/workers_done.txt
marker
