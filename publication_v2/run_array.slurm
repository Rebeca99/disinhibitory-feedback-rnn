#!/usr/bin/env bash
#SBATCH --job-name=biornn-controls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=biornn-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
PLAN=${1:?Pass the absolute plan.json path to sbatch}
PYTHON=${PYTHON:-python}
# Activate your department's environment before sbatch. Add the site's required
# partition/account directives to this template, not to the Python experiments.
exec "$PYTHON" train_publication.py --plan "$PLAN" --job-index "$SLURM_ARRAY_TASK_ID" --device cuda:0
