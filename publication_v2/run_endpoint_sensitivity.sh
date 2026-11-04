#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
BASE_ROOT=${BASE_ROOT:-"$HOME/biornn_publication_v2"}
EVAL_ENDPOINTS=${EVAL_ENDPOINTS:-"1500 2000"}
REFERENCE_ENDPOINT=${REFERENCE_ENDPOINT:-2500}
REFERENCE_SUMMARY=${REFERENCE_SUMMARY:-"$BASE_ROOT/continuation_${REFERENCE_ENDPOINT}/heldout_2500_summary"}
GPU_IDS=${GPU_IDS:-"0 1"}
EVAL_SEED=${EVAL_SEED:-830001}
SEQUENCES=${SEQUENCES:-6}
TRIALS=${TRIALS:-300}
TAG=${TAG:-"endpoint_sensitivity_${EVAL_SEED}"}
COMBINED_OUTPUT=${COMBINED_OUTPUT:-"$BASE_ROOT/${TAG}_summary"}
MIN_FREE_GB=${MIN_FREE_GB:-2}

read -r -a endpoints <<< "$EVAL_ENDPOINTS"
read -r -a devices <<< "$GPU_IDS"

if [[ ${#endpoints[@]} -eq 0 ]]; then
  echo "No endpoints supplied. Set EVAL_ENDPOINTS, for example: EVAL_ENDPOINTS=\"1500 2000\"" >&2
  exit 2
fi
if [[ ! -d "$REFERENCE_SUMMARY" ]]; then
  echo "Missing completed reference summary: $REFERENCE_SUMMARY" >&2
  exit 1
fi
if [[ ${#devices[@]} -lt 1 ]]; then
  echo "No GPUs supplied. Set GPU_IDS, for example: GPU_IDS=\"0 1\"" >&2
  exit 2
fi
if [[ $SEQUENCES -lt 1 || $TRIALS -lt 1 ]]; then
  echo "SEQUENCES and TRIALS must be positive." >&2
  exit 2
fi

free_kb=$(df -Pk "$BASE_ROOT" | awk 'NR==2 {print $4}')
min_free_kb=$((MIN_FREE_GB * 1024 * 1024))
if [[ -n "$free_kb" && "$free_kb" -lt "$min_free_kb" ]]; then
  echo "Only $((free_kb / 1024 / 1024)) GiB free under $BASE_ROOT; require at least ${MIN_FREE_GB} GiB." >&2
  exit 1
fi

run_worker() {
  local plan=$1
  local gpu=$2
  local start=$3
  local step=$4
  local n_jobs=$5
  local index
  export CUDA_VISIBLE_DEVICES="$gpu"
  for ((index=start; index<n_jobs; index+=step)); do
    "$PYTHON" evaluate_publication.py \
      --plan "$plan" \
      --job-index "$index" \
      --device cuda:0 \
      --sequences "$SEQUENCES" \
      --trials "$TRIALS" \
      --eval-seed "$EVAL_SEED" \
      --interventions intact \
      --no-features \
      --tag "$TAG"
  done
}

for endpoint in "${endpoints[@]}"; do
  root="$BASE_ROOT/continuation_${endpoint}"
  plan="$root/plan.json"
  log_root="$root/evaluation_logs"
  summary="$root/${TAG}_summary"

  if [[ ! -f "$plan" ]]; then
    echo "Missing plan: $plan" >&2
    exit 1
  fi
  n_jobs=$("$PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["jobs"]))' "$plan")
  if [[ $n_jobs -lt 1 ]]; then
    echo "Plan contains no jobs: $plan" >&2
    exit 1
  fi
  expected=$((n_jobs * 2))
  mkdir -p "$log_root"

  echo "============================================================"
  echo "Endpoint: $endpoint updates"
  echo "Plan: $plan"
  echo "Jobs: $n_jobs; expected ID/OOD completions: $expected"
  echo "Evaluation seed: $EVAL_SEED; features: disabled"

  pids=()
  for worker in "${!devices[@]}"; do
    gpu=${devices[$worker]}
    logfile="$log_root/${TAG}_gpu${gpu}.log"
    run_worker "$plan" "$gpu" "$worker" "${#devices[@]}" "$n_jobs" >"$logfile" 2>&1 &
    pids+=("$!")
    echo "START worker $worker on GPU $gpu (PID ${pids[$worker]})"
  done

  failures=0
  for worker in "${!pids[@]}"; do
    if ! wait "${pids[$worker]}"; then
      echo "Worker $worker failed for endpoint $endpoint." >&2
      failures=1
    fi
  done
  if [[ $failures -ne 0 ]]; then
    echo "Inspect logs in $log_root and rerun this script to resume." >&2
    exit 1
  fi

  completed=$(find "$root" -path "*/${TAG}/*/complete.json" -print | wc -l | tr -d ' ')
  if [[ "$completed" -ne "$expected" ]]; then
    echo "Endpoint $endpoint has $completed/$expected complete outputs." >&2
    echo "Inspect logs in $log_root and rerun this script to resume." >&2
    exit 1
  fi

  "$PYTHON" summarize_publication.py --plan "$plan" --tag "$TAG" --output "$summary"
  echo "COMPLETE endpoint $endpoint: $summary"
done

summary_args=()
for endpoint in "${endpoints[@]}"; do
  summary_args+=(--summary "$endpoint=$BASE_ROOT/continuation_${endpoint}/${TAG}_summary")
done
summary_args+=(--summary "$REFERENCE_ENDPOINT=$REFERENCE_SUMMARY")

"$PYTHON" summarize_endpoint_sensitivity.py \
  "${summary_args[@]}" \
  --output "$COMBINED_OUTPUT"

echo "============================================================"
echo "Endpoint sensitivity complete."
echo "Combined summary: $COMBINED_OUTPUT"
df -h "$BASE_ROOT"
