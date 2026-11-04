#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-$HOME/biornn_feedback_stochastic_bottleneck_pfc01_seeds/pfc_0p1}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/biornn_feedback_stochastic_bottleneck_pfc01_seeds/summaries/training_audit}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "training_root=$TRAINING_ROOT"
echo "output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" scripts/audit_training_and_feedback.py \
  --training-root "$TRAINING_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model-glob '*feedback_stochastic_hidden_switch*' \
  --variants baseline pfc_vip_error sr_vip_error both_vip_error \
  --seeds 1 2 3 4 5 \
  --tail-n 5 \
  --smooth-window 25 \
  --require-complete

echo "Training and feedback audit finished."
