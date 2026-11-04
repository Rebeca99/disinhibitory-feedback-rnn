#!/usr/bin/env bash
# Run a lightweight hidden-rule decoding validation on final stochastic models.

set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-"$HOME/biornn_feedback_stochastic_bottleneck_pfc01_seeds/pfc_0p1"}
ANALYSIS_DIR=${ANALYSIS_DIR:-"$HOME/biornn_feedback_stochastic_bottleneck_pfc01_seeds/summaries/rule_decoding_small"}
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_GLOB=${MODEL_GLOB:-"*feedback_stochastic_hidden_switch*"}
VARIANTS=${VARIANTS:-"baseline pfc_vip_error sr_vip_error both_vip_error"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
N_TRIALS=${N_TRIALS:-300}
TEST_REPEATS=${TEST_REPEATS:-3}
TEST_SEED_BASE=${TEST_SEED_BASE:-14000}
SWITCH_EVERY=${SWITCH_EVERY:-20}
INIT_RULES=${INIT_RULES:-"color shape"}
WINDOWS=${WINDOWS:-"feedback early_post_feedback pre_stim_delay response_epoch"}
FEATURE_SETS=${FEATURE_SETS:-"pfc sr pfc_sr pfc_motif sr_motif"}
CONDITIONS=${CONDITIONS:-"all_nonfirst previous_error stable_trials true_switch_trial"}
MAX_PCA_COMPONENTS=${MAX_PCA_COMPONENTS:-20}
CV_FOLDS=${CV_FOLDS:-5}
DEVICE=${DEVICE:-auto}
DRY_RUN=${DRY_RUN:-0}

echo "output_dir=${OUTPUT_DIR}"
echo "analysis_dir=${ANALYSIS_DIR}"
echo "model_glob=${MODEL_GLOB}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "n_trials=${N_TRIALS}"
echo "test_repeats=${TEST_REPEATS}"
echo "test_seed_base=${TEST_SEED_BASE}"
echo "switch_every=${SWITCH_EVERY}"
echo "init_rules=${INIT_RULES}"
echo "windows=${WINDOWS}"
echo "feature_sets=${FEATURE_SETS}"
echo "conditions=${CONDITIONS}"
echo "max_pca_components=${MAX_PCA_COMPONENTS}"
echo "cv_folds=${CV_FOLDS}"
echo "device=${DEVICE}"

ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

"${PYTHON_BIN}" scripts/analyze_rule_decoding.py \
  --output-dir "${OUTPUT_DIR}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --model-glob "${MODEL_GLOB}" \
  --variants ${VARIANTS} \
  --seeds ${SEEDS} \
  --n-trials "${N_TRIALS}" \
  --test-repeats "${TEST_REPEATS}" \
  --test-seed-base "${TEST_SEED_BASE}" \
  --switch-every "${SWITCH_EVERY}" \
  --init-rules ${INIT_RULES} \
  --windows ${WINDOWS} \
  --feature-sets ${FEATURE_SETS} \
  --conditions ${CONDITIONS} \
  --max-pca-components "${MAX_PCA_COMPONENTS}" \
  --cv-folds "${CV_FOLDS}" \
  --device "${DEVICE}" \
  "${ARGS[@]}"

echo "Rule-decoding analysis finished."
