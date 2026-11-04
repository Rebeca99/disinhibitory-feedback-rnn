#!/usr/bin/env bash
# Convenience wrapper for the feedback-dependent history bottleneck diagnostic.

set -euo pipefail

export OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRATCH:-$HOME}/biornn_feedback_history_bottleneck_pfc01"}
export GPU_IDS=${GPU_IDS:-"0 1"}
export PFC_SCALES=${PFC_SCALES:-"0.1"}
export VARIANTS=${VARIANTS:-"baseline sr_vip_error both_vip_error"}
export SEEDS=${SEEDS:-"1 2 3 4 5"}
export MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-1000}
export MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-24}
export LEARNING_RATE=${LEARNING_RATE:-0.0001}
export VIP_FEEDBACK_SCALE=${VIP_FEEDBACK_SCALE:-5.0}
export VIP_FEEDBACK_INIT=${VIP_FEEDBACK_INIT:-0.02}
export FEEDBACK_NOISE_MODE=${FEEDBACK_NOISE_MODE:-mixed}
export FEEDBACK_NOISE_PROB=${FEEDBACK_NOISE_PROB:-0.2}
export FEEDBACK_NOISE_FLIP_FRACTION=${FEEDBACK_NOISE_FLIP_FRACTION:-0.5}

# Keep history available but weaken stimulus/choice shortcuts.
export PREV_STIM_MAG=${PREV_STIM_MAG:-0.25}
export PREV_CHOICE_MAG=${PREV_CHOICE_MAG:-0.25}
export PREV_REW_MAG=${PREV_REW_MAG:-1.0}
export NO_PREV_STIM=${NO_PREV_STIM:-0}
export NO_PREV_CHOICE=${NO_PREV_CHOICE:-0}
export NO_PREV_REW=${NO_PREV_REW:-0}

bash scripts/run_pfc_scale_diagnostic_parallel_gpus.sh
