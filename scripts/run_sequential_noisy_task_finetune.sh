#!/usr/bin/env bash
# Run warm-start noisy-feedback WCST fine-tuning jobs sequentially on one allocated GPU.

set -euo pipefail

VARIANTS_DEFAULT="baseline both_vip_error"
SEEDS_DEFAULT="1 2 3 4 5"

if [[ -z "${PRETRAINED_PATH:-}" ]]; then
  echo "ERROR: PRETRAINED_PATH must point to one Liu/Wang pretrained checkpoint file." >&2
  exit 1
fi

OUTPUT_DIR=${OUTPUT_DIR:-"${SCRATCH:-$HOME}/biornn_feedback_noisy_task_runs"}
PYTHON_BIN=${PYTHON_BIN:-python}
VARIANTS=${VARIANTS:-$VARIANTS_DEFAULT}
SEEDS=${SEEDS:-$SEEDS_DEFAULT}
JOB_PREFIX=${JOB_PREFIX:-feedback_noisy_task_finetune}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-2000}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-12}
FEEDBACK_NOISE_MODE=${FEEDBACK_NOISE_MODE:-mixed}
FEEDBACK_NOISE_PROB=${FEEDBACK_NOISE_PROB:-0.2}
FEEDBACK_NOISE_FLIP_FRACTION=${FEEDBACK_NOISE_FLIP_FRACTION:-0.5}
PFC_FEEDBACK_SCALE=${PFC_FEEDBACK_SCALE:-1.0}
VIP_FEEDBACK_SCALE=${VIP_FEEDBACK_SCALE:-1.0}
VIP_FEEDBACK_INIT=${VIP_FEEDBACK_INIT:-0.0}
PREV_STIM_MAG=${PREV_STIM_MAG:-1.0}
PREV_CHOICE_MAG=${PREV_CHOICE_MAG:-1.0}
PREV_REW_MAG=${PREV_REW_MAG:-1.0}
NO_PREV_STIM=${NO_PREV_STIM:-0}
NO_PREV_CHOICE=${NO_PREV_CHOICE:-0}
NO_PREV_REW=${NO_PREV_REW:-0}

if [[ "${CURRICULUM:-0}" == "1" ]]; then
  CURRICULUM_ARG="--curriculum"
else
  CURRICULUM_ARG=""
fi

HISTORY_ARGS=(
  --prev-stim-mag "${PREV_STIM_MAG}"
  --prev-choice-mag "${PREV_CHOICE_MAG}"
  --prev-rew-mag "${PREV_REW_MAG}"
)
if [[ "${NO_PREV_STIM}" == "1" ]]; then
  HISTORY_ARGS+=(--no-prev-stim)
fi
if [[ "${NO_PREV_CHOICE}" == "1" ]]; then
  HISTORY_ARGS+=(--no-prev-choice)
fi
if [[ "${NO_PREV_REW}" == "1" ]]; then
  HISTORY_ARGS+=(--no-prev-rew)
fi

echo "pretrained_path=${PRETRAINED_PATH}"
echo "output_dir=${OUTPUT_DIR}"
echo "python=${PYTHON_BIN}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "job_prefix=${JOB_PREFIX}"
echo "learning_rate=${LEARNING_RATE}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "feedback_noise_mode=${FEEDBACK_NOISE_MODE}"
echo "feedback_noise_prob=${FEEDBACK_NOISE_PROB}"
echo "feedback_noise_flip_fraction=${FEEDBACK_NOISE_FLIP_FRACTION}"
echo "pfc_feedback_scale=${PFC_FEEDBACK_SCALE}"
echo "vip_feedback_scale=${VIP_FEEDBACK_SCALE}"
echo "vip_feedback_init=${VIP_FEEDBACK_INIT}"
echo "prev_stim_mag=${PREV_STIM_MAG}, prev_choice_mag=${PREV_CHOICE_MAG}, prev_rew_mag=${PREV_REW_MAG}"
echo "no_prev_stim=${NO_PREV_STIM}, no_prev_choice=${NO_PREV_CHOICE}, no_prev_rew=${NO_PREV_REW}"

for SEED in ${SEEDS}; do
  for VARIANT in ${VARIANTS}; do
    echo "============================================================"
    echo "Starting noisy-task warm-start variant=${VARIANT} seed=${SEED}"
    echo "============================================================"

    "${PYTHON_BIN}" scripts/finetune_feedback_noisy_task_variant.py \
      --variant "${VARIANT}" \
      --seed "${SEED}" \
      --pretrained-path "${PRETRAINED_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --job-prefix "${JOB_PREFIX}" \
      --learning-rate "${LEARNING_RATE}" \
      --max-train-batches "${MAX_TRAIN_BATCHES}" \
      --checkpoint-interval 10 \
      --max-runtime-hours "${MAX_RUNTIME_HOURS}" \
      --feedback-noise-mode "${FEEDBACK_NOISE_MODE}" \
      --feedback-noise-prob "${FEEDBACK_NOISE_PROB}" \
      --feedback-noise-flip-fraction "${FEEDBACK_NOISE_FLIP_FRACTION}" \
      --pfc-feedback-scale "${PFC_FEEDBACK_SCALE}" \
      --vip-feedback-scale "${VIP_FEEDBACK_SCALE}" \
      --vip-feedback-init "${VIP_FEEDBACK_INIT}" \
      "${HISTORY_ARGS[@]}" \
      ${CURRICULUM_ARG}
  done
done

"${PYTHON_BIN}" scripts/summarize_training_results.py \
  --output-dir "${OUTPUT_DIR}"

echo "All sequential noisy-task feedback-gating jobs finished."
