#!/usr/bin/env bash
# Run the missing frozen-backbone seeds after the first long-name run produced
# only one seed's saved models. This keeps the same output root and uses short
# job names so seed-specific checkpoint/model names cannot collide.

set -euo pipefail

export PRETRAINED_PATH=${PRETRAINED_PATH:-"$HOME/disinhib_rnn/pretrained_wcst_17"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-$HOME/biornn_feedback_stochastic_frozen_backbone_pfc01_seeds}
export PYTHON_BIN=${PYTHON_BIN:-python}
export GPU_IDS=${GPU_IDS:-"0 1"}
export SEEDS=${SEEDS:-"1 2 3 4"}
export VARIANTS=${VARIANTS:-"baseline pfc_vip_error sr_vip_error both_vip_error"}
export FREEZE_BACKBONE_MODE=${FREEZE_BACKBONE_MODE:-feedback_readout}
export SHORT_JOB_NAMES=1
export JOB_PREFIX_BASE=${JOB_PREFIX_BASE:-fzbb_stoch}

echo "Running remaining frozen-backbone seeds with short job names."
echo "output_root=${OUTPUT_ROOT}"
echo "seeds=${SEEDS}"
echo "variants=${VARIANTS}"

bash scripts/run_stochastic_hidden_switch_frozen_backbone_parallel_gpus.sh
