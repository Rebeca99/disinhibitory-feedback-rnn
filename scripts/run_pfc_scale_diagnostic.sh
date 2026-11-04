#!/usr/bin/env bash
# Run noisy-feedback warm-start fine-tuning while weakening direct PFC feedback.

set -euo pipefail

if [[ -z "${PRETRAINED_PATH:-}" ]]; then
  echo "ERROR: PRETRAINED_PATH must point to one Liu/Wang pretrained checkpoint file." >&2
  exit 1
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRATCH:-$HOME}/biornn_feedback_pfc_scale_diagnostic"}
PYTHON_BIN=${PYTHON_BIN:-python}
PFC_SCALES=${PFC_SCALES:-"0.1 0.0"}
VARIANTS=${VARIANTS:-"baseline sr_vip_error both_vip_error"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
JOB_PREFIX_BASE=${JOB_PREFIX_BASE:-feedback_noisy_task_pfcscale}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-1000}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-24}
FEEDBACK_NOISE_MODE=${FEEDBACK_NOISE_MODE:-mixed}
FEEDBACK_NOISE_PROB=${FEEDBACK_NOISE_PROB:-0.2}
FEEDBACK_NOISE_FLIP_FRACTION=${FEEDBACK_NOISE_FLIP_FRACTION:-0.5}
VIP_FEEDBACK_SCALE=${VIP_FEEDBACK_SCALE:-5.0}
VIP_FEEDBACK_INIT=${VIP_FEEDBACK_INIT:-0.02}
PREV_STIM_MAG=${PREV_STIM_MAG:-1.0}
PREV_CHOICE_MAG=${PREV_CHOICE_MAG:-1.0}
PREV_REW_MAG=${PREV_REW_MAG:-1.0}
NO_PREV_STIM=${NO_PREV_STIM:-0}
NO_PREV_CHOICE=${NO_PREV_CHOICE:-0}
NO_PREV_REW=${NO_PREV_REW:-0}

mkdir -p "${OUTPUT_ROOT}"

echo "pretrained_path=${PRETRAINED_PATH}"
echo "output_root=${OUTPUT_ROOT}"
echo "python=${PYTHON_BIN}"
echo "pfc_scales=${PFC_SCALES}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "learning_rate=${LEARNING_RATE}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_runtime_hours=${MAX_RUNTIME_HOURS}"
echo "feedback_noise=${FEEDBACK_NOISE_MODE}, prob=${FEEDBACK_NOISE_PROB}, flip_fraction=${FEEDBACK_NOISE_FLIP_FRACTION}"
echo "vip_feedback_scale=${VIP_FEEDBACK_SCALE}"
echo "vip_feedback_init=${VIP_FEEDBACK_INIT}"
echo "prev_stim_mag=${PREV_STIM_MAG}, prev_choice_mag=${PREV_CHOICE_MAG}, prev_rew_mag=${PREV_REW_MAG}"
echo "no_prev_stim=${NO_PREV_STIM}, no_prev_choice=${NO_PREV_CHOICE}, no_prev_rew=${NO_PREV_REW}"

for PFC_SCALE in ${PFC_SCALES}; do
  SCALE_TAG=$(echo "${PFC_SCALE}" | tr '-' 'm' | tr '.' 'p')
  SCALE_OUTPUT_DIR="${OUTPUT_ROOT}/pfc_${SCALE_TAG}"
  JOB_PREFIX="${JOB_PREFIX_BASE}_pfc${SCALE_TAG}"

  echo "============================================================"
  echo "Starting PFC-scale diagnostic pfc_feedback_scale=${PFC_SCALE}"
  echo "Output: ${SCALE_OUTPUT_DIR}"
  echo "============================================================"

  PRETRAINED_PATH="${PRETRAINED_PATH}" \
  OUTPUT_DIR="${SCALE_OUTPUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  VARIANTS="${VARIANTS}" \
  SEEDS="${SEEDS}" \
  JOB_PREFIX="${JOB_PREFIX}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES}" \
  MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS}" \
  FEEDBACK_NOISE_MODE="${FEEDBACK_NOISE_MODE}" \
  FEEDBACK_NOISE_PROB="${FEEDBACK_NOISE_PROB}" \
  FEEDBACK_NOISE_FLIP_FRACTION="${FEEDBACK_NOISE_FLIP_FRACTION}" \
  PFC_FEEDBACK_SCALE="${PFC_SCALE}" \
  VIP_FEEDBACK_SCALE="${VIP_FEEDBACK_SCALE}" \
  VIP_FEEDBACK_INIT="${VIP_FEEDBACK_INIT}" \
  PREV_STIM_MAG="${PREV_STIM_MAG}" \
  PREV_CHOICE_MAG="${PREV_CHOICE_MAG}" \
  PREV_REW_MAG="${PREV_REW_MAG}" \
  NO_PREV_STIM="${NO_PREV_STIM}" \
  NO_PREV_CHOICE="${NO_PREV_CHOICE}" \
  NO_PREV_REW="${NO_PREV_REW}" \
  bash scripts/run_sequential_noisy_task_finetune.sh
done

"${PYTHON_BIN}" scripts/summarize_pfc_scale_diagnostic.py \
  --output-root "${OUTPUT_ROOT}"

echo "PFC-scale diagnostic jobs finished."
