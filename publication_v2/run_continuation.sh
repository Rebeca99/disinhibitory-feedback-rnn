#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
SOURCE_PLAN=${SOURCE_PLAN:-"$HOME/biornn_publication_v2/pilot/plan.json"}
SOURCE_CODE_DIR=${SOURCE_CODE_DIR:-"$PWD/../publication_v2_2026-10-08"}
TOTAL_UPDATES=${TOTAL_UPDATES:-1000}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/biornn_publication_v2/continuation_${TOTAL_UPDATES}"}
PLAN="$OUTPUT_ROOT/plan.json"

"$PYTHON" prepare_continuation.py --source-plan "$SOURCE_PLAN" \
  --source-code-dir "$SOURCE_CODE_DIR" --output-root "$OUTPUT_ROOT" \
  --updates "$TOTAL_UPDATES" --plan "$PLAN"
"$PYTHON" -m unittest discover -s tests -q
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared and verified only: $PLAN"
  exit 0
fi
if [[ -n "${GPU_IDS:-}" ]]; then
  read -r -a DEVICES <<< "$GPU_IDS"
  "$PYTHON" launch.py --plan "$PLAN" --gpus "${DEVICES[@]}"
else
  "$PYTHON" launch.py --plan "$PLAN"
fi
echo "Fine-tuning continuation finished. Plan: $PLAN"
"$PYTHON" summarize_publication.py --plan "$PLAN" --output "$OUTPUT_ROOT/training_check"
echo "Training audit and full learning curves: $OUTPUT_ROOT/training_check"
echo "Held-out evaluation is a separate step; see CONTINUATION.md."
