#!/bin/sh
# usage: logs/run_model.sh <hf-model-id>  -- extraction + all four probes on popqa
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1
M="$1"
cd "$(dirname "$0")/.."
for A in layer_attn linear mlp traj; do
  echo "=== $M $A $(date +%T)"
  python scripts/03_run_experiment.py --dataset popqa --model "$M" --arch $A --device cuda || echo "FAILED $M $A"
done
echo "=== DONE $M $(date +%T)"
