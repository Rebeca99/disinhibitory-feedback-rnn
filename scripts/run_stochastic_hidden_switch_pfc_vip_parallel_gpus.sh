#!/usr/bin/env bash
# Train the missing PFC-VIP-only error-feedback variant for the stochastic hidden-switch bottleneck.
#
# This is intentionally narrow: it reuses the exact stochastic bottleneck defaults
# from run_stochastic_hidden_switch_bottleneck_parallel_gpus.sh, but only launches
# pfc_vip_error into the same default output root used for the 5-seed stochastic run.

set -euo pipefail

export OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRATCH:-$HOME}/biornn_feedback_stochastic_bottleneck_pfc01_seeds"}
export GPU_IDS=${GPU_IDS:-"0 1"}
export PFC_SCALES=${PFC_SCALES:-"0.1"}
export VARIANTS=${VARIANTS:-"pfc_vip_error"}
export SEEDS=${SEEDS:-"1 2 3 4 5"}

bash scripts/run_stochastic_hidden_switch_bottleneck_parallel_gpus.sh
