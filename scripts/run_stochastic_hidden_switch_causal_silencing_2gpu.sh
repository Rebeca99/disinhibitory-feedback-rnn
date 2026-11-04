#!/usr/bin/env bash
# Causal silencing analysis for the stochastic hidden-switch bottleneck.
#
# This tests whether the error-linked VIP/SST/dendritic motifs are functionally
# necessary during the feedback and immediate post-feedback window.

set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-"$HOME/biornn_feedback_stochastic_bottleneck_pfc01_seeds/pfc_0p1"}
ANALYSIS_ROOT=${ANALYSIS_ROOT:-"$HOME/biornn_feedback_stochastic_bottleneck_pfc01_seeds/summaries"}
ANALYSIS_NAME=${ANALYSIS_NAME:-"causal_silencing_feedback_plus_early_r3_n300_2gpu"}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}

VARIANTS=${VARIANTS:-"baseline pfc_vip_error sr_vip_error both_vip_error"}
SEEDS_GPU0=${SEEDS_GPU0:-"1 2 3"}
SEEDS_GPU1=${SEEDS_GPU1:-"4 5"}
N_TRIALS=${N_TRIALS:-300}
TEST_REPEATS=${TEST_REPEATS:-3}
TEST_SEED_BASE=${TEST_SEED_BASE:-14000}
SWITCH_EVERY=${SWITCH_EVERY:-20}
INIT_RULES=${INIT_RULES:-"color shape"}
CAUSAL_TESTS=${CAUSAL_TESTS:-"none sr_vip_silence_feedback sr_sst_silence_feedback sr_edend_silence_feedback pfc_vip_silence_feedback pfc_sst_silence_feedback"}
WEIGHT_ABLATIONS=${WEIGHT_ABLATIONS:-"none"}
OPTO_WINDOW=${OPTO_WINDOW:-feedback_plus_early_post}
DEVICE=${DEVICE:-cuda}

ANALYSIS0="${ANALYSIS_ROOT}/${ANALYSIS_NAME}_gpu0"
ANALYSIS1="${ANALYSIS_ROOT}/${ANALYSIS_NAME}_gpu1"
COMBINED="${ANALYSIS_ROOT}/${ANALYSIS_NAME}"

mkdir -p "${ANALYSIS0}" "${ANALYSIS1}" "${COMBINED}"

echo "model_root=${MODEL_ROOT}"
echo "analysis_root=${ANALYSIS_ROOT}"
echo "analysis_name=${ANALYSIS_NAME}"
echo "variants=${VARIANTS}"
echo "causal_tests=${CAUSAL_TESTS}"
echo "opto_window=${OPTO_WINDOW}"
echo "gpu0=${GPU0}, seeds=${SEEDS_GPU0}, analysis=${ANALYSIS0}"
echo "gpu1=${GPU1}, seeds=${SEEDS_GPU1}, analysis=${ANALYSIS1}"
echo "n_trials=${N_TRIALS}, test_repeats=${TEST_REPEATS}, test_seed_base=${TEST_SEED_BASE}"

CUDA_VISIBLE_DEVICES="${GPU0}" "${PYTHON_BIN}" scripts/analyze_feedback_mechanism.py \
  --output-dir "${MODEL_ROOT}" \
  --analysis-dir "${ANALYSIS0}" \
  --model-glob '*feedback_stochastic_hidden_switch*' \
  --variants ${VARIANTS} \
  --seeds ${SEEDS_GPU0} \
  --n-trials "${N_TRIALS}" \
  --test-repeats "${TEST_REPEATS}" \
  --test-seed-base "${TEST_SEED_BASE}" \
  --switch-every "${SWITCH_EVERY}" \
  --init-rules ${INIT_RULES} \
  --causal-tests ${CAUSAL_TESTS} \
  --weight-ablations ${WEIGHT_ABLATIONS} \
  --opto-window "${OPTO_WINDOW}" \
  --pair-causal-seeds \
  --device "${DEVICE}" \
  > "${ANALYSIS0}/run.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" "${PYTHON_BIN}" scripts/analyze_feedback_mechanism.py \
  --output-dir "${MODEL_ROOT}" \
  --analysis-dir "${ANALYSIS1}" \
  --model-glob '*feedback_stochastic_hidden_switch*' \
  --variants ${VARIANTS} \
  --seeds ${SEEDS_GPU1} \
  --n-trials "${N_TRIALS}" \
  --test-repeats "${TEST_REPEATS}" \
  --test-seed-base "${TEST_SEED_BASE}" \
  --switch-every "${SWITCH_EVERY}" \
  --init-rules ${INIT_RULES} \
  --causal-tests ${CAUSAL_TESTS} \
  --weight-ablations ${WEIGHT_ABLATIONS} \
  --opto-window "${OPTO_WINDOW}" \
  --pair-causal-seeds \
  --device "${DEVICE}" \
  > "${ANALYSIS1}/run.log" 2>&1 &
PID1=$!

echo "Started GPU0 PID=${PID0} and GPU1 PID=${PID1}"

failed=0
if ! wait "${PID0}"; then
  failed=1
fi
if ! wait "${PID1}"; then
  failed=1
fi

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one causal-analysis worker failed. Check ${ANALYSIS0}/run.log and ${ANALYSIS1}/run.log." >&2
  exit 1
fi

"${PYTHON_BIN}" - "${ANALYSIS0}" "${ANALYSIS1}" "${COMBINED}" <<'PY'
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

inputs = [Path(sys.argv[1]), Path(sys.argv[2])]
output = Path(sys.argv[3])
output.mkdir(parents=True, exist_ok=True)

csv_names = [
    'model_summary.csv',
    'feedback_weight_summary.csv',
    'frozen_test_metrics.csv',
    'activity_by_condition.csv',
    'activity_error_minus_correct.csv',
    'behavior_by_condition.csv',
    'switch_position_behavior.csv',
    'paired_behavior_deltas_vs_baseline.csv',
    'weight_ablation_behavior_deltas.csv',
    'causal_behavior_deltas.csv',
]

for name in csv_names:
    frames = []
    for path in inputs:
        csv_path = path / name
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(csv_path))
        except EmptyDataError:
            continue
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output / name, index=False)
        print(f'wrote {output / name}')
    else:
        print(f'skipped empty/missing {name}')

print(f'combined outputs written to {output}')
PY

"${PYTHON_BIN}" scripts/summarize_causal_silencing_results.py \
  --analysis-dir "${COMBINED}"

echo "Causal silencing analysis finished."
