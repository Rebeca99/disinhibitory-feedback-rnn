#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
BASE_ROOT=${BASE_ROOT:-"$HOME/biornn_publication_v2"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$BASE_ROOT/replication_2500_seeds_2_4"}
PLAN=${PLAN:-"$OUTPUT_ROOT/plan.json"}
EXISTING_SEED1_PLAN=${EXISTING_SEED1_PLAN:-"$BASE_ROOT/continuation_2500/plan.json"}
COMBINED_PLAN=${COMBINED_PLAN:-"$OUTPUT_ROOT/combined_seeds_1_4_plan.json"}
COMBINED_SUMMARY=${COMBINED_SUMMARY:-"$OUTPUT_ROOT/combined_training_check"}
PARENT_DIR=${PARENT_DIR:-"$PWD/parents"}
ANCHOR=${ANCHOR:-"$PARENT_DIR/parent_01_wcst_6.pt"}
SEEDS=${SEEDS:-"2 3 4"}
GPU_IDS=${GPU_IDS:-"0 1"}
UPDATES=${UPDATES:-2500}
MIN_FREE_GB=${MIN_FREE_GB:-3}

ROUTES=(baseline pfc_vip sr_vip both_vip sr_dend10 sr_pv10 sr_dend_pv10)
read -r -a seed_values <<< "$SEEDS"
read -r -a devices <<< "$GPU_IDS"

if [[ ${#seed_values[@]} -lt 1 ]]; then
  echo "No fine-tuning seeds supplied. Set SEEDS, for example: SEEDS=\"2 3 4\"." >&2
  exit 2
fi
if [[ ${#devices[@]} -lt 1 ]]; then
  echo "No GPUs supplied. Set GPU_IDS, for example: GPU_IDS=\"0 1\"." >&2
  exit 2
fi
if [[ $UPDATES -lt 1 ]]; then
  echo "UPDATES must be positive." >&2
  exit 2
fi
if [[ ! -f "$ANCHOR" ]]; then
  echo "Missing parent anchor: $ANCHOR" >&2
  exit 1
fi
if [[ ! -f "$EXISTING_SEED1_PLAN" ]]; then
  echo "Missing completed seed-1 plan: $EXISTING_SEED1_PLAN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
free_kb=$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
min_free_kb=$((MIN_FREE_GB * 1024 * 1024))
if [[ -n "$free_kb" && "$free_kb" -lt "$min_free_kb" ]]; then
  echo "Only $((free_kb / 1024 / 1024)) GiB free under $OUTPUT_ROOT; require at least ${MIN_FREE_GB} GiB." >&2
  exit 1
fi

"$PYTHON" prepare_plan.py \
  --parent-dir "$PARENT_DIR" \
  --anchor "$ANCHOR" \
  --parent-count 2 \
  --seeds "${seed_values[@]}" \
  --routes "${ROUTES[@]}" \
  --updates "$UPDATES" \
  --output-root "$OUTPUT_ROOT" \
  --plan "$PLAN"

"$PYTHON" -m unittest discover -s tests -q
"$PYTHON" preflight.py --plan "$PLAN"

"$PYTHON" - "$PLAN" "$UPDATES" <<'PY'
import json
import sys

plan_path, expected_updates = sys.argv[1], int(sys.argv[2])
plan = json.load(open(plan_path))
jobs = plan["jobs"]
expected_routes = {
    "baseline", "pfc_vip", "sr_vip", "both_vip",
    "sr_dend10", "sr_pv10", "sr_dend_pv10",
}
parents = {job["parent_id"] for job in jobs}
seeds = {job["seed"] for job in jobs}
routes = {job["route"] for job in jobs}
assert len(parents) == 2, parents
assert routes == expected_routes, routes
assert all(job["adaptation"] == "full" for job in jobs)
assert all(job["config"]["updates"] == expected_updates for job in jobs)
assert len(jobs) == len(parents) * len(seeds) * len(routes)
print("Verified replication matrix: {} parents x {} seeds x {} routes = {} jobs at {} updates.".format(
    len(parents), len(seeds), len(routes), len(jobs), expected_updates
))
PY

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared and verified only: $PLAN"
  "$PYTHON" launch.py --plan "$PLAN" --gpus "${devices[@]}" --dry-run
  exit 0
fi

"$PYTHON" launch.py --plan "$PLAN" --gpus "${devices[@]}"
"$PYTHON" summarize_publication.py --plan "$PLAN" --output "$OUTPUT_ROOT/training_check"

"$PYTHON" merge_replication_plans.py \
  --base-plan "$EXISTING_SEED1_PLAN" \
  --replication-plan "$PLAN" \
  --output "$COMBINED_PLAN"
"$PYTHON" summarize_publication.py --plan "$COMBINED_PLAN" --output "$COMBINED_SUMMARY"

echo "============================================================"
echo "Fine-tuning replication complete."
echo "New-seed plan: $PLAN"
echo "Combined seed-1-to-4 analysis plan: $COMBINED_PLAN"
echo "Combined training audit: $COMBINED_SUMMARY"
df -h "$OUTPUT_ROOT"
