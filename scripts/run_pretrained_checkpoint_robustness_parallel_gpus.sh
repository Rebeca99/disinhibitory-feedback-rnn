#!/usr/bin/env bash
# Manuscript robustness test: rerun the final stochastic hidden-switch bottleneck
# across multiple pretrained Liu/Wang checkpoints.
#
# Checkpoint input:
#   PRETRAINED_PATHS_FILE=/path/to/checkpoints.txt
#     One checkpoint per line. Lines can be either:
#       /absolute/path/to/checkpoint
#       label=/absolute/path/to/checkpoint
#
#   or PRETRAINED_PATHS="/path/ckpt1,/path/ckpt2"
#
# The script creates one subfolder per pretrained checkpoint under OUTPUT_ROOT and
# reuses run_pfc_scale_diagnostic_parallel_gpus.sh for the actual two-GPU launch.

set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRATCH:-$HOME}/biornn_feedback_pretrained_checkpoint_robustness"}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-"0 1"}
PFC_SCALES=${PFC_SCALES:-"0.1"}
VARIANTS=${VARIANTS:-"baseline pfc_vip_error sr_vip_error both_vip_error"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
JOB_PREFIX_BASE=${JOB_PREFIX_BASE:-ckptrobust_stoch}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-1000}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-24}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10}
TEST_N_TRIALS=${TEST_N_TRIALS:-200}
TEST_SWITCH_EVERY=${TEST_SWITCH_EVERY:-20}
SHORT_JOB_NAMES=${SHORT_JOB_NAMES:-1}

# Final manuscript stochastic hidden-switch bottleneck defaults.
BLOCK_LEN=${BLOCK_LEN:-30}
TRAIN_SWITCH_MODE=${TRAIN_SWITCH_MODE:-stochastic_hazard}
TEST_SWITCH_MODE=${TEST_SWITCH_MODE:-stochastic_hazard}
STOCHASTIC_MIN_BLOCK_LEN=${STOCHASTIC_MIN_BLOCK_LEN:-8}
STOCHASTIC_MAX_BLOCK_LEN=${STOCHASTIC_MAX_BLOCK_LEN:-22}
STOCHASTIC_SWITCH_PROB=${STOCHASTIC_SWITCH_PROB:-0.10}
TEST_STOCHASTIC_MIN_BLOCK_LEN=${TEST_STOCHASTIC_MIN_BLOCK_LEN:-8}
TEST_STOCHASTIC_MAX_BLOCK_LEN=${TEST_STOCHASTIC_MAX_BLOCK_LEN:-35}
TEST_STOCHASTIC_SWITCH_PROB=${TEST_STOCHASTIC_SWITCH_PROB:-0.08}

# Keep feedback informative while reducing stimulus/choice history shortcuts.
PREV_STIM_MAG=${PREV_STIM_MAG:-0.10}
PREV_CHOICE_MAG=${PREV_CHOICE_MAG:-0.10}
PREV_REW_MAG=${PREV_REW_MAG:-1.0}
NO_PREV_STIM=${NO_PREV_STIM:-0}
NO_PREV_CHOICE=${NO_PREV_CHOICE:-0}
NO_PREV_REW=${NO_PREV_REW:-0}

# Ambiguous feedback, weak direct PFC route, amplified VIP route.
FEEDBACK_NOISE_MODE=${FEEDBACK_NOISE_MODE:-mixed}
FEEDBACK_NOISE_PROB=${FEEDBACK_NOISE_PROB:-0.10}
FEEDBACK_NOISE_FLIP_FRACTION=${FEEDBACK_NOISE_FLIP_FRACTION:-0.5}
VIP_FEEDBACK_SCALE=${VIP_FEEDBACK_SCALE:-5.0}
VIP_FEEDBACK_INIT=${VIP_FEEDBACK_INIT:-0.02}
FREEZE_BACKBONE_MODE=${FREEZE_BACKBONE_MODE:-none}

sanitize_label() {
  basename "$1" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_*//; s/_*$//'
}

add_checkpoint_spec() {
  local raw="$1"
  local label=""
  local path=""

  raw="${raw%%#*}"
  raw="$(echo "${raw}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  [[ -z "${raw}" ]] && return 0

  if [[ "${raw}" == *=* ]]; then
    label="${raw%%=*}"
    path="${raw#*=}"
  else
    path="${raw}"
    label="$(sanitize_label "${path}")"
  fi

  label="$(echo "${label}" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_*//; s/_*$//')"
  if [[ -z "${label}" ]]; then
    label="checkpoint"
  fi
  CHECKPOINT_SPECS+=("${label}|${path}")
}

declare -a CHECKPOINT_SPECS=()

if [[ -n "${PRETRAINED_PATHS_FILE:-}" ]]; then
  if [[ ! -f "${PRETRAINED_PATHS_FILE}" ]]; then
    echo "ERROR: PRETRAINED_PATHS_FILE not found: ${PRETRAINED_PATHS_FILE}" >&2
    exit 1
  fi
  while IFS= read -r line; do
    add_checkpoint_spec "${line}"
  done < "${PRETRAINED_PATHS_FILE}"
fi

if [[ -n "${PRETRAINED_PATHS:-}" ]]; then
  IFS=',' read -r -a path_array <<< "${PRETRAINED_PATHS}"
  for checkpoint_path in "${path_array[@]}"; do
    add_checkpoint_spec "${checkpoint_path}"
  done
fi

if [[ "${#CHECKPOINT_SPECS[@]}" -eq 0 ]]; then
  echo "ERROR: provide PRETRAINED_PATHS_FILE or PRETRAINED_PATHS." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "output_root=${OUTPUT_ROOT}"
echo "python=${PYTHON_BIN}"
echo "gpu_ids=${GPU_IDS}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "pfc_scales=${PFC_SCALES}"
echo "n_pretrained_checkpoints=${#CHECKPOINT_SPECS[@]}"
echo "task=stochastic_hidden_switch"
echo "switch=train min/max/p ${STOCHASTIC_MIN_BLOCK_LEN}/${STOCHASTIC_MAX_BLOCK_LEN}/${STOCHASTIC_SWITCH_PROB}; test ${TEST_STOCHASTIC_MIN_BLOCK_LEN}/${TEST_STOCHASTIC_MAX_BLOCK_LEN}/${TEST_STOCHASTIC_SWITCH_PROB}"
echo "history=prev_stim ${PREV_STIM_MAG}, prev_choice ${PREV_CHOICE_MAG}, prev_rew ${PREV_REW_MAG}"
echo "feedback=noise ${FEEDBACK_NOISE_MODE} p=${FEEDBACK_NOISE_PROB}, pfc_scales=${PFC_SCALES}, vip_scale=${VIP_FEEDBACK_SCALE}, vip_init=${VIP_FEEDBACK_INIT}"

idx=0
for spec in "${CHECKPOINT_SPECS[@]}"; do
  idx=$((idx + 1))
  label="${spec%%|*}"
  checkpoint_path="${spec#*|}"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "ERROR: checkpoint not found: ${checkpoint_path}" >&2
    exit 1
  fi

  checkpoint_root="${OUTPUT_ROOT}/ckpt_${idx}_${label}"
  mkdir -p "${checkpoint_root}"
  printf '%s\n' "${checkpoint_path}" > "${checkpoint_root}/pretrained_checkpoint_path.txt"
  printf '%s\n' "${label}" > "${checkpoint_root}/pretrained_checkpoint_label.txt"

  echo
  echo "============================================================"
  echo "Starting checkpoint ${idx}/${#CHECKPOINT_SPECS[@]}: ${label}"
  echo "path=${checkpoint_path}"
  echo "checkpoint_root=${checkpoint_root}"
  echo "============================================================"

  PRETRAINED_PATH="${checkpoint_path}" \
  OUTPUT_ROOT="${checkpoint_root}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  GPU_IDS="${GPU_IDS}" \
  PFC_SCALES="${PFC_SCALES}" \
  VARIANTS="${VARIANTS}" \
  SEEDS="${SEEDS}" \
  JOB_PREFIX_BASE="${JOB_PREFIX_BASE}_ckpt${idx}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES}" \
  MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS}" \
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
  TEST_N_TRIALS="${TEST_N_TRIALS}" \
  TEST_SWITCH_EVERY="${TEST_SWITCH_EVERY}" \
  BLOCK_LEN="${BLOCK_LEN}" \
  TRAIN_SWITCH_MODE="${TRAIN_SWITCH_MODE}" \
  TEST_SWITCH_MODE="${TEST_SWITCH_MODE}" \
  STOCHASTIC_MIN_BLOCK_LEN="${STOCHASTIC_MIN_BLOCK_LEN}" \
  STOCHASTIC_MAX_BLOCK_LEN="${STOCHASTIC_MAX_BLOCK_LEN}" \
  STOCHASTIC_SWITCH_PROB="${STOCHASTIC_SWITCH_PROB}" \
  TEST_STOCHASTIC_MIN_BLOCK_LEN="${TEST_STOCHASTIC_MIN_BLOCK_LEN}" \
  TEST_STOCHASTIC_MAX_BLOCK_LEN="${TEST_STOCHASTIC_MAX_BLOCK_LEN}" \
  TEST_STOCHASTIC_SWITCH_PROB="${TEST_STOCHASTIC_SWITCH_PROB}" \
  PREV_STIM_MAG="${PREV_STIM_MAG}" \
  PREV_CHOICE_MAG="${PREV_CHOICE_MAG}" \
  PREV_REW_MAG="${PREV_REW_MAG}" \
  NO_PREV_STIM="${NO_PREV_STIM}" \
  NO_PREV_CHOICE="${NO_PREV_CHOICE}" \
  NO_PREV_REW="${NO_PREV_REW}" \
  FEEDBACK_NOISE_MODE="${FEEDBACK_NOISE_MODE}" \
  FEEDBACK_NOISE_PROB="${FEEDBACK_NOISE_PROB}" \
  FEEDBACK_NOISE_FLIP_FRACTION="${FEEDBACK_NOISE_FLIP_FRACTION}" \
  VIP_FEEDBACK_SCALE="${VIP_FEEDBACK_SCALE}" \
  VIP_FEEDBACK_INIT="${VIP_FEEDBACK_INIT}" \
  FREEZE_BACKBONE_MODE="${FREEZE_BACKBONE_MODE}" \
  SHORT_JOB_NAMES="${SHORT_JOB_NAMES}" \
  bash scripts/run_pfc_scale_diagnostic_parallel_gpus.sh
done

"${PYTHON_BIN}" scripts/summarize_pretrained_checkpoint_robustness.py \
  --output-root "${OUTPUT_ROOT}"

echo "Pretrained-checkpoint robustness jobs finished."
