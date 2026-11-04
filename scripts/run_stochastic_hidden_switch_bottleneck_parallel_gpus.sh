#!/usr/bin/env bash
# Convenience wrapper for the stochastic hidden-switch error-gated bottleneck pilot.

set -euo pipefail

export OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRATCH:-$HOME}/biornn_feedback_stochastic_bottleneck_pfc01_pilot"}
export GPU_IDS=${GPU_IDS:-"0 1"}
export PFC_SCALES=${PFC_SCALES:-"0.1"}
export VARIANTS=${VARIANTS:-"baseline sr_vip_error both_vip_error"}
export SEEDS=${SEEDS:-"1"}
export JOB_PREFIX_BASE=${JOB_PREFIX_BASE:-feedback_stochastic_hidden_switch}
export MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-1000}
export MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-24}
export LEARNING_RATE=${LEARNING_RATE:-0.0001}

# Stronger online error-driven bottleneck.
export BLOCK_LEN=${BLOCK_LEN:-30}
export TRAIN_SWITCH_MODE=${TRAIN_SWITCH_MODE:-stochastic_hazard}
export TEST_SWITCH_MODE=${TEST_SWITCH_MODE:-stochastic_hazard}
export STOCHASTIC_MIN_BLOCK_LEN=${STOCHASTIC_MIN_BLOCK_LEN:-8}
export STOCHASTIC_MAX_BLOCK_LEN=${STOCHASTIC_MAX_BLOCK_LEN:-22}
export STOCHASTIC_SWITCH_PROB=${STOCHASTIC_SWITCH_PROB:-0.10}
export TEST_STOCHASTIC_MIN_BLOCK_LEN=${TEST_STOCHASTIC_MIN_BLOCK_LEN:-8}
export TEST_STOCHASTIC_MAX_BLOCK_LEN=${TEST_STOCHASTIC_MAX_BLOCK_LEN:-35}
export TEST_STOCHASTIC_SWITCH_PROB=${TEST_STOCHASTIC_SWITCH_PROB:-0.08}

# Keep reward/error informative while weakening stimulus and choice shortcuts.
export PREV_STIM_MAG=${PREV_STIM_MAG:-0.10}
export PREV_CHOICE_MAG=${PREV_CHOICE_MAG:-0.10}
export PREV_REW_MAG=${PREV_REW_MAG:-1.0}
export NO_PREV_STIM=${NO_PREV_STIM:-0}
export NO_PREV_CHOICE=${NO_PREV_CHOICE:-0}
export NO_PREV_REW=${NO_PREV_REW:-0}

# Make feedback ambiguous but not hopeless.
export FEEDBACK_NOISE_MODE=${FEEDBACK_NOISE_MODE:-mixed}
export FEEDBACK_NOISE_PROB=${FEEDBACK_NOISE_PROB:-0.10}
export FEEDBACK_NOISE_FLIP_FRACTION=${FEEDBACK_NOISE_FLIP_FRACTION:-0.5}

# Preserve weak direct PFC route and amplified VIP route for the causal comparison.
export VIP_FEEDBACK_SCALE=${VIP_FEEDBACK_SCALE:-5.0}
export VIP_FEEDBACK_INIT=${VIP_FEEDBACK_INIT:-0.02}

bash scripts/run_pfc_scale_diagnostic_parallel_gpus.sh
