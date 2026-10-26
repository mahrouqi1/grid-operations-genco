#!/bin/bash
# Starts 6 detached workers of launch_v3.sh (survive the session). Run once from this directory.
cd "$(dirname "$0")"; mkdir -p logs_v3
for w in 1 2 3 4 5 6; do setsid nohup nice -n 15 bash launch_v3.sh $w > logs_v3/worker_$w.out 2>&1 < /dev/null & echo $! >> logs_v3/pids.txt; done
