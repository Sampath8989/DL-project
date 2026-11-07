#!/usr/bin/env bash
# Linux/WSL equivalent of run_review3.ps1. Same order, same resumability.
set -u
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p logs
LOG="logs/review3-$(date +%Y%m%d-%H%M%S).log"

step () {
  echo "" | tee -a "$LOG"
  echo "=== $1" | tee -a "$LOG"
  shift
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$LOG"
  echo "    ($(( (SECONDS-t0)/60 )) min)" | tee -a "$LOG"
}

MODELS=(Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct \
        Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-1B-Instruct \
        Qwen/Qwen2.5-0.5B-Instruct)

# verify, never rebuild: the caches on disk were extracted in the order of
# the working set that existed then, so a re-sample would silently attach
# their labels to different questions
step "verify cache alignment" python scripts/09_verify_alignment.py || {
  echo "cache/working-set mismatch; refusing to run the analysis" | tee -a "$LOG"; exit 1; }
for m in "${MODELS[@]}"; do
  for d in popqa triviaqa; do
    step "extract $m $d" python scripts/03_run_experiment.py \
        --dataset "$d" --model "$m" --arch linear --device cuda
    step "baselines $m $d" python scripts/06_baselines.py \
        --model "$m" --dataset "$d" --device cuda --k 5
  done
done
step "final analysis" python scripts/07_final_analysis.py --crossfit
step "transfer" python scripts/08_transfer.py
echo "out/review3_results.json and out/review3_transfer.json written; log $LOG"
