#!/usr/bin/env bash
# Full mid-sem pipeline: train clean SAC agents -> stress test -> figures.
# Run from the project root inside WSL:   bash scripts/run_all.sh
set -e
source ~/rlenv/bin/activate
cd "$(dirname "$0")/.."

ENVS=${ENVS:-"Walker2d-v5 HalfCheetah-v5"}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-300000}
mkdir -p logs

# 1) training: all (env, seed) runs in parallel
for env in $ENVS; do
  for s in $SEEDS; do
    python -m src.train --env "$env" --seed "$s" --steps "$STEPS" --device cpu > "logs/train_${env}_s${s}.log" 2>&1 &
  done
done
wait

# 2) stress tests
for env in $ENVS; do
  for s in $SEEDS; do
    python -m src.evaluate --env "$env" --seed "$s" --episodes 10 > "logs/eval_${env}_s${s}.log" 2>&1 &
  done
done
wait

# 3) figures and tables
python -m src.analyze
