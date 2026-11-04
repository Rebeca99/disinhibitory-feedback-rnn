#!/usr/bin/env bash
# Run PFC-scale diagnostic jobs in parallel across multiple visible GPUs.

set -euo pipefail

if [[ -z "${PRETRAINED_PATH:-}" ]]; then
  echo "ERROR: PRETRAINED_PATH must point to one Liu/Wang pretrained checkpoint file." >&2
  exit 1
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRATCH:-$HOME}/biornn_feedback_pfc_scale_diagnostic"}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-"0 1"}
PFC_SCALES=${PFC_SCALES:-"0.1"}
VARIANTS=${VARIANTS:-"baseline sr_vip_error both_vip_error"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
JOB_PREFIX_BASE=${JOB_PREFIX_BASE:-feedback_noisy_task_pfcscale}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-1000}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-24}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10}
TEST_N_TRIALS=${TEST_N_TRIALS:-200}
TEST_SWITCH_EVERY=${TEST_SWITCH_EVERY:-20}
BLOCK_LEN=${BLOCK_LEN:-}
TRAIN_SWITCH_MODE=${TRAIN_SWITCH_MODE:-fixed_n}
TEST_SWITCH_MODE=${TEST_SWITCH_MODE:-fixed_interval}
STOCHASTIC_MIN_BLOCK_LEN=${STOCHASTIC_MIN_BLOCK_LEN:-10}
STOCHASTIC_MAX_BLOCK_LEN=${STOCHASTIC_MAX_BLOCK_LEN:-35}
STOCHASTIC_SWITCH_PROB=${STOCHASTIC_SWITCH_PROB:-0.08}
TEST_STOCHASTIC_MIN_BLOCK_LEN=${TEST_STOCHASTIC_MIN_BLOCK_LEN:-${STOCHASTIC_MIN_BLOCK_LEN}}
TEST_STOCHASTIC_MAX_BLOCK_LEN=${TEST_STOCHASTIC_MAX_BLOCK_LEN:-${STOCHASTIC_MAX_BLOCK_LEN}}
TEST_STOCHASTIC_SWITCH_PROB=${TEST_STOCHASTIC_SWITCH_PROB:-${STOCHASTIC_SWITCH_PROB}}
FEEDBACK_NOISE_MODE=${FEEDBACK_NOISE_MODE:-mixed}
FEEDBACK_NOISE_PROB=${FEEDBACK_NOISE_PROB:-0.2}
FEEDBACK_NOISE_FLIP_FRACTION=${FEEDBACK_NOISE_FLIP_FRACTION:-0.5}
VIP_FEEDBACK_SCALE=${VIP_FEEDBACK_SCALE:-5.0}
VIP_FEEDBACK_INIT=${VIP_FEEDBACK_INIT:-0.02}
PREV_STIM_MAG=${PREV_STIM_MAG:-1.0}
PREV_CHOICE_MAG=${PREV_CHOICE_MAG:-1.0}
PREV_REW_MAG=${PREV_REW_MAG:-1.0}
FREEZE_BACKBONE_MODE=${FREEZE_BACKBONE_MODE:-none}
SHORT_JOB_NAMES=${SHORT_JOB_NAMES:-0}
NO_PREV_STIM=${NO_PREV_STIM:-0}
NO_PREV_CHOICE=${NO_PREV_CHOICE:-0}
NO_PREV_REW=${NO_PREV_REW:-0}

read -r -a GPU_ARRAY <<< "${GPU_IDS}"
N_GPUS=${#GPU_ARRAY[@]}
if [[ "${N_GPUS}" -lt 1 ]]; then
  echo "ERROR: GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"

echo "pretrained_path=${PRETRAINED_PATH}"
echo "output_root=${OUTPUT_ROOT}"
echo "python=${PYTHON_BIN}"
echo "gpu_ids=${GPU_IDS}"
echo "pfc_scales=${PFC_SCALES}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "learning_rate=${LEARNING_RATE}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_runtime_hours=${MAX_RUNTIME_HOURS}"
echo "checkpoint_interval=${CHECKPOINT_INTERVAL}"
echo "test_n_trials=${TEST_N_TRIALS}"
echo "test_switch_every=${TEST_SWITCH_EVERY}"
echo "block_len=${BLOCK_LEN:-checkpoint_default}"
echo "train_switch_mode=${TRAIN_SWITCH_MODE}"
echo "test_switch_mode=${TEST_SWITCH_MODE}"
echo "stochastic_switch=min:${STOCHASTIC_MIN_BLOCK_LEN}, max:${STOCHASTIC_MAX_BLOCK_LEN}, p:${STOCHASTIC_SWITCH_PROB}"
echo "test_stochastic_switch=min:${TEST_STOCHASTIC_MIN_BLOCK_LEN}, max:${TEST_STOCHASTIC_MAX_BLOCK_LEN}, p:${TEST_STOCHASTIC_SWITCH_PROB}"
echo "feedback_noise=${FEEDBACK_NOISE_MODE}, prob=${FEEDBACK_NOISE_PROB}, flip_fraction=${FEEDBACK_NOISE_FLIP_FRACTION}"
echo "vip_feedback_scale=${VIP_FEEDBACK_SCALE}"
echo "vip_feedback_init=${VIP_FEEDBACK_INIT}"
echo "prev_stim_mag=${PREV_STIM_MAG}, prev_choice_mag=${PREV_CHOICE_MAG}, prev_rew_mag=${PREV_REW_MAG}"
echo "freeze_backbone_mode=${FREEZE_BACKBONE_MODE}"
echo "short_job_names=${SHORT_JOB_NAMES}"
echo "no_prev_stim=${NO_PREV_STIM}, no_prev_choice=${NO_PREV_CHOICE}, no_prev_rew=${NO_PREV_REW}"

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

SWITCH_ARGS=(
  --train-switch-mode "${TRAIN_SWITCH_MODE}"
  --test-switch-mode "${TEST_SWITCH_MODE}"
  --stochastic-min-block-len "${STOCHASTIC_MIN_BLOCK_LEN}"
  --stochastic-max-block-len "${STOCHASTIC_MAX_BLOCK_LEN}"
  --stochastic-switch-prob "${STOCHASTIC_SWITCH_PROB}"
  --test-stochastic-min-block-len "${TEST_STOCHASTIC_MIN_BLOCK_LEN}"
  --test-stochastic-max-block-len "${TEST_STOCHASTIC_MAX_BLOCK_LEN}"
  --test-stochastic-switch-prob "${TEST_STOCHASTIC_SWITCH_PROB}"
)
if [[ -n "${BLOCK_LEN}" ]]; then
  SWITCH_ARGS+=(--block-len "${BLOCK_LEN}")
fi

JOB_NAME_ARGS=()
if [[ "${SHORT_JOB_NAMES}" == "1" ]]; then
  JOB_NAME_ARGS+=(--short-job-name)
fi

scale_tag() {
  echo "$1" | tr '-' 'm' | tr '.' 'p'
}

run_job() {
  local pfc_scale="$1"
  local variant="$2"
  local seed="$3"
  local gpu_id="$4"
  local tag
  local scale_output_dir
  local job_prefix
  local log_file

  tag=$(scale_tag "${pfc_scale}")
  scale_output_dir="${OUTPUT_ROOT}/pfc_${tag}"
  job_prefix="${JOB_PREFIX_BASE}_pfc${tag}"
  log_file="${OUTPUT_ROOT}/logs/pfc${tag}_${variant}_seed${seed}_gpu${gpu_id}.log"
  mkdir -p "${scale_output_dir}"

  echo "[$(date)] START gpu=${gpu_id} pfc=${pfc_scale} variant=${variant} seed=${seed} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" scripts/finetune_feedback_noisy_task_variant.py \
    --variant "${variant}" \
    --seed "${seed}" \
    --pretrained-path "${PRETRAINED_PATH}" \
    --output-dir "${scale_output_dir}" \
    --job-prefix "${job_prefix}" \
    --learning-rate "${LEARNING_RATE}" \
    --max-train-batches "${MAX_TRAIN_BATCHES}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    --max-runtime-hours "${MAX_RUNTIME_HOURS}" \
    --test-n-trials "${TEST_N_TRIALS}" \
    --test-switch-every "${TEST_SWITCH_EVERY}" \
    "${JOB_NAME_ARGS[@]}" \
    "${SWITCH_ARGS[@]}" \
    --feedback-noise-mode "${FEEDBACK_NOISE_MODE}" \
    --feedback-noise-prob "${FEEDBACK_NOISE_PROB}" \
    --feedback-noise-flip-fraction "${FEEDBACK_NOISE_FLIP_FRACTION}" \
    --pfc-feedback-scale "${pfc_scale}" \
    --vip-feedback-scale "${VIP_FEEDBACK_SCALE}" \
    --vip-feedback-init "${VIP_FEEDBACK_INIT}" \
    --freeze-backbone-mode "${FREEZE_BACKBONE_MODE}" \
    "${HISTORY_ARGS[@]}" \
    > "${log_file}" 2>&1
  echo "[$(date)] DONE  gpu=${gpu_id} pfc=${pfc_scale} variant=${variant} seed=${seed}"
}

pids=()
for worker_idx in "${!GPU_ARRAY[@]}"; do
  (
    gpu_id="${GPU_ARRAY[$worker_idx]}"
    job_idx=0
    for pfc_scale in ${PFC_SCALES}; do
      for seed in ${SEEDS}; do
        for variant in ${VARIANTS}; do
          if (( job_idx % N_GPUS == worker_idx )); then
            run_job "${pfc_scale}" "${variant}" "${seed}" "${gpu_id}"
          fi
          job_idx=$((job_idx + 1))
        done
      done
    done
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one GPU worker failed. Check logs under ${OUTPUT_ROOT}/logs." >&2
  exit 1
fi

for pfc_scale in ${PFC_SCALES}; do
  tag=$(scale_tag "${pfc_scale}")
  "${PYTHON_BIN}" scripts/summarize_training_results.py \
    --output-dir "${OUTPUT_ROOT}/pfc_${tag}"
done

"${PYTHON_BIN}" scripts/summarize_pfc_scale_diagnostic.py \
  --output-root "${OUTPUT_ROOT}"

echo "Parallel PFC-scale diagnostic jobs finished."
