#!/usr/bin/env bash
# Run feedback-gating training jobs sequentially on one allocated GPU.

set -euo pipefail

VARIANTS_DEFAULT="baseline pfc_vip_error sr_vip_error both_vip_error vip_only_error"
SEEDS_DEFAULT="1 2 3 4 5"

OUTPUT_DIR=${OUTPUT_DIR:-"${SCRATCH:-$PWD}/biornn_feedback_runs"}
PYTHON_BIN=${PYTHON_BIN:-python}
VARIANTS=${VARIANTS:-$VARIANTS_DEFAULT}
SEEDS=${SEEDS:-$SEEDS_DEFAULT}
JOB_PREFIX=${JOB_PREFIX:-feedback_gating}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-47}

echo "output_dir=${OUTPUT_DIR}"
echo "python=${PYTHON_BIN}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "job_prefix=${JOB_PREFIX}"

for SEED in ${SEEDS}; do
  for VARIANT in ${VARIANTS}; do
    echo "============================================================"
    echo "Starting variant=${VARIANT} seed=${SEED}"
    echo "============================================================"

    "${PYTHON_BIN}" scripts/train_feedback_variant.py \
      --variant "${VARIANT}" \
      --seed "${SEED}" \
      --output-dir "${OUTPUT_DIR}" \
      --job-prefix "${JOB_PREFIX}" \
      --batch-size 10 \
      --block-len 20 \
      --n-switches 3 \
      --checkpoint-interval 10 \
      --max-runtime-hours "${MAX_RUNTIME_HOURS}"
  done
done

"${PYTHON_BIN}" scripts/summarize_training_results.py \
  --output-dir "${OUTPUT_DIR}"

echo "All sequential feedback-gating jobs finished."
