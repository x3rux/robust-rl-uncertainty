#!/usr/bin/env bash
# Quick throughput benchmark: CPU vs GPU, 3 SAC runs in parallel each.
source ~/rlenv/bin/activate
cd "$(dirname "$0")/.."
for d in cpu cuda; do
  s=$(date +%s)
  for k in 1 2 3; do
    python -m src.train --env Hopper-v5 --seed 9$k --steps 15000 --eval-freq 5000 --device $d >/dev/null 2>&1 &
  done
  wait
  echo "$d: 3 parallel runs x 15k steps took $(( $(date +%s) - s ))s"
done
rm -rf runs/Hopper-v5/seed9*
