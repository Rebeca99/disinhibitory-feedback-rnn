#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PHASE=${1:-pilot}
if [[ $# -gt 0 ]]; then shift; fi
PYTHON=${PYTHON:-python}
PARENT_DIR=${PARENT_DIR:-"$PWD/parents"}
ANCHOR=${ANCHOR:-"$PWD/parents/parent_01_wcst_6.pt"}
PARENT_COUNT=${PARENT_COUNT:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRATCH:-$HOME}/biornn_publication_v2/$PHASE"}
mkdir -p "$OUTPUT_ROOT"
PLAN="$OUTPUT_ROOT/plan.json"
OPTIONS=()
case "$PHASE" in
  smoke)
    OPTIONS=(--seeds 1 --updates 2 --batch-size 2 --block-trials 3 --validation-every 1
             --validation-trials 12 --routes baseline sr_vip)
    ;;
  pilot)
    OPTIONS=(--seeds 1 --updates 300)
    ;;
  core)
    OPTIONS=(--seeds 1 2 3 --updates 1000)
    ;;
  extended)
    OPTIONS=(--seeds 1 2 3 --updates 1000 --with-motif-training-control
             --with-normmatched-combined --with-restricted-control)
    ;;
  *) echo "Use smoke, pilot, core, or extended" >&2; exit 2 ;;
esac
"$PYTHON" prepare_plan.py --parent-dir "$PARENT_DIR" --anchor "$ANCHOR" \
  --parent-count "$PARENT_COUNT" --output-root "$OUTPUT_ROOT" --plan "$PLAN" "${OPTIONS[@]}" "$@"
"$PYTHON" -m unittest discover -s tests -q
"$PYTHON" preflight.py --plan "$PLAN" --device cpu --trials 100
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared plan only: $PLAN"
  exit 0
fi
if [[ -n "${GPU_IDS:-}" ]]; then
  read -r -a DEVICES <<< "$GPU_IDS"
  "$PYTHON" launch.py --plan "$PLAN" --gpus "${DEVICES[@]}"
else
  "$PYTHON" launch.py --plan "$PLAN"
fi
echo "Training finished. Plan: $PLAN"
echo "Run evaluate_publication.py, then decode_publication.py and summarize_publication.py. See README.md."
