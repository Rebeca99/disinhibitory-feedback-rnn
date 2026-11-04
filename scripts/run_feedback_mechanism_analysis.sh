#!/usr/bin/env bash
# Analyze feedback-gating trained models for VIP/SST mechanism signatures.

set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-"${SCRATCH:-$HOME}/biornn_feedback_engaged_pfc05_pilot"}
PYTHON_BIN=${PYTHON_BIN:-python}
ANALYSIS_DIR=${ANALYSIS_DIR:-"${OUTPUT_DIR}/summaries/mechanism"}
VARIANTS=${VARIANTS:-"baseline both_vip_error both_vip_all_feedback"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
N_TRIALS=${N_TRIALS:-200}
TEST_REPEATS=${TEST_REPEATS:-1}
TEST_SEED_BASE=${TEST_SEED_BASE:-12345}
SWITCH_EVERY=${SWITCH_EVERY:-20}
INIT_RULES=${INIT_RULES:-"color shape"}
CAUSAL_TESTS=${CAUSAL_TESTS:-"none"}
WEIGHT_ABLATIONS=${WEIGHT_ABLATIONS:-"none"}
OPTO_WINDOW=${OPTO_WINDOW:-feedback}
DEVICE=${DEVICE:-auto}

echo "output_dir=${OUTPUT_DIR}"
echo "analysis_dir=${ANALYSIS_DIR}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "n_trials=${N_TRIALS}"
echo "test_repeats=${TEST_REPEATS}"
echo "test_seed_base=${TEST_SEED_BASE}"
echo "switch_every=${SWITCH_EVERY}"
echo "init_rules=${INIT_RULES}"
echo "causal_tests=${CAUSAL_TESTS}"
echo "weight_ablations=${WEIGHT_ABLATIONS}"
echo "opto_window=${OPTO_WINDOW}"
echo "device=${DEVICE}"

"${PYTHON_BIN}" scripts/analyze_feedback_mechanism.py \
  --output-dir "${OUTPUT_DIR}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --variants ${VARIANTS} \
  --seeds ${SEEDS} \
  --n-trials "${N_TRIALS}" \
  --test-repeats "${TEST_REPEATS}" \
  --test-seed-base "${TEST_SEED_BASE}" \
  --switch-every "${SWITCH_EVERY}" \
  --init-rules ${INIT_RULES} \
  --causal-tests ${CAUSAL_TESTS} \
  --weight-ablations ${WEIGHT_ABLATIONS} \
  --opto-window "${OPTO_WINDOW}" \
  --device "${DEVICE}"

echo "Mechanism analysis finished."
