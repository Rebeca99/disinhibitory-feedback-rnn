#!/usr/bin/env bash
# Run warm-start feedback-gating fine-tuning jobs sequentially on one allocated GPU.

set -euo pipefail

VARIANTS_DEFAULT="baseline pfc_vip_error sr_vip_error both_vip_error vip_only_error"
SEEDS_DEFAULT="1 2 3 4 5"

if [[ -z "${PRETRAINED_PATH:-}" ]]; then
  echo "ERROR: PRETRAINED_PATH must point to one Liu/Wang pretrained checkpoint file." >&2
  exit 1
fi

OUTPUT_DIR=${OUTPUT_DIR:-"${SCRATCH:-$HOME}/biornn_feedback_finetune_runs"}
PYTHON_BIN=${PYTHON_BIN:-python}
VARIANTS=${VARIANTS:-$VARIANTS_DEFAULT}
SEEDS=${SEEDS:-$SEEDS_DEFAULT}
JOB_PREFIX=${JOB_PREFIX:-feedback_finetune}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-2000}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-12}

if [[ "${CURRICULUM:-0}" == "1" ]]; then
  CURRICULUM_ARG="--curriculum"
else
  CURRICULUM_ARG=""
fi

echo "pretrained_path=${PRETRAINED_PATH}"
echo "output_dir=${OUTPUT_DIR}"
echo "python=${PYTHON_BIN}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "job_prefix=${JOB_PREFIX}"
echo "learning_rate=${LEARNING_RATE}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"

for SEED in ${SEEDS}; do
  for VARIANT in ${VARIANTS}; do
    echo "============================================================"
    echo "Starting warm-start variant=${VARIANT} seed=${SEED}"
    echo "============================================================"

    "${PYTHON_BIN}" scripts/finetune_feedback_variant.py \
      --variant "${VARIANT}" \
      --seed "${SEED}" \
      --pretrained-path "${PRETRAINED_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --job-prefix "${JOB_PREFIX}" \
      --learning-rate "${LEARNING_RATE}" \
      --max-train-batches "${MAX_TRAIN_BATCHES}" \
      --checkpoint-interval 10 \
      --max-runtime-hours "${MAX_RUNTIME_HOURS}" \
      ${CURRICULUM_ARG}
  done
done

"${PYTHON_BIN}" scripts/summarize_training_results.py \
  --output-dir "${OUTPUT_DIR}"

echo "All sequential warm-start feedback-gating jobs finished."
