#!/usr/bin/env bash
# One command: verify checksums and the pinned env, rebuild the two post figures (figures-v4) from the committed result
# files, run the unit tests, print a PASS or FAIL summary. Exit 0 only if every step passed.
#   bash reproduce.sh              # all four steps, in this order
#   bash reproduce.sh checksums    # or: env | figures | tests (one step)
# Needs the env of environment-genco.lock.yml (see REPRODUCE.md). No private data and no GPU are used.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true MPLBACKEND=Agg
export PYTHONPATH="$PWD/core:$PWD/tracks/genco-challenge/src:$PWD/tracks/powerflow-surrogate/src"
what=${1:-all}
LOG=${REPRODUCE_LOG_DIR:-$(mktemp -d)}
names=(); results=()

step() {      # step <name> <command...>: run, keep the log, record PASS or FAIL
  local name=$1; shift
  echo "#### $name"
  if "$@" >"$LOG/$name.log" 2>&1; then results+=(PASS); else results+=(FAIL); tail -n 15 "$LOG/$name.log"; fi
  names+=("$name")
}
run_figures() {
  PY="$PY" bash tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v4/make_all.sh
}
want() { [ "$what" = all ] || [ "$what" = "$1" ]; }

want checksums && step checksums sha256sum --quiet -c CHECKSUMS.sha256
want env       && step env "$PY" tools/check_env.py
want figures   && step figures run_figures
want tests     && step tests "$PY" -m pytest -q -m "not slow" tests -p no:cacheprovider

echo
echo "==== reproduce.sh summary (logs in $LOG) ===="
fail=0
for i in "${!names[@]}"; do
  printf '  %-10s %s\n' "${names[$i]}" "${results[$i]}"
  [ "${results[$i]}" = PASS ] || fail=1
done
if [ "${#names[@]}" -eq 0 ]; then echo "  no step selected: '$what'"; fail=1; fi
[ "$fail" -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit "$fail"
