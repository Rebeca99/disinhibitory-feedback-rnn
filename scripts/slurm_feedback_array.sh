#!/usr/bin/env bash
#SBATCH --job-name=biornn_feedback
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-24

set -euo pipefail

mkdir -p logs

VARIANTS=(baseline pfc_vip_error sr_vip_error both_vip_error vip_only_error)
SEEDS=(1 2 3 4 5)

N_VARIANTS=${#VARIANTS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
VARIANT=${VARIANTS[$((TASK_ID % N_VARIANTS))]}
SEED=${SEEDS[$((TASK_ID / N_VARIANTS))]}

OUTPUT_DIR=${OUTPUT_DIR:-"${SCRATCH:-$PWD}/biornn_feedback_runs"}
PYTHON_BIN=${PYTHON_BIN:-python}

echo "variant=${VARIANT}"
echo "seed=${SEED}"
echo "output_dir=${OUTPUT_DIR}"
echo "python=${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/train_feedback_variant.py \
  --variant "${VARIANT}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --job-prefix feedback_gating \
  --batch-size 10 \
  --block-len 20 \
  --n-switches 3 \
  --checkpoint-interval 10 \
  --max-runtime-hours 47
