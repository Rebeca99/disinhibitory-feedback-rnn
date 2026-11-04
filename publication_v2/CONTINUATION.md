# Continue the completed pilot to 1,000 total updates

Run continuation from a fixed repository commit. Do not edit or rerun the
completed pilot plan/checkpoints while continuation runs.

## Run on the department machine

Keep your existing `ssn-speech` environment active and use your allocated GPUs:

```bash
cd ~/disinhibitory-feedback-rnn/publication_v2
GPU_IDS="0 1" bash run_continuation.sh
```

The default source is `$HOME/biornn_publication_v2/pilot/plan.json`. The original
code folder must remain at `../publication_v2_2026-10-08`. Override `SOURCE_PLAN`
or `SOURCE_CODE_DIR` if you put them elsewhere.

The wrapper verifies every source condition and runs regression tests before
launching. Expected preparation output:

```text
Verified source jobs: 14
Continue updates: 300 -> 1000 (700 additional/job)
```

Each new worker then prints `CONTINUE ... completed 300, next update 301`.
The launcher console shows START/END; detailed progress is in each job's worker.log.

Output is separate: `$HOME/biornn_publication_v2/continuation_1000/`.
Both parent networks and all seven routes remain in the plan. No new fine-tuning
seeds are added by continuation. Every source job is retained, independent of
its accuracy or the direction of its feedback effect.

## What is restored and unchanged

- All learned model parameters and Adam state, including moment estimates and
  optimizer step counters, are restored from each source final.pt.
- Update numbering continues at 301. The next cards, rule schedule, feedback
  corruption and network noise use the original update-indexed seed streams.
- Model dynamics, masks, routing gains, learning rate, clipping, task timing,
  history attenuation, corruption rates and validation schedule are unchanged.
- Source training and validation records are retained; new curves span updates
  1-1,000. The final 1,000-update parameter state is used, not a best checkpoint.
- No recurrent trial state is imported between optimizer updates. As in the
  original protocol, the next 30-trial training block starts from zero state.

The source code hash must match the verified pilot release, and model/task files
must be byte-identical. Source plans, parents and final checkpoints are checksum
checked. A changed learning rate, route, source file or destination configuration
is rejected. Use the same PyTorch/CUDA environment as the pilot.

`runtime_hours` includes the completed pilot and continuation. The audit also
reports `source_completed_updates`, `continuation_updates`, and
`continuation_runtime_hours`, plus source checkpoint hashes.

## Resume or prepare only

Rerun the same command after interruption; latest.pt restores the continued
model and optimizer, and completed jobs are skipped. SIGTERM/Ctrl+C requests a
checkpoint at the next optimizer boundary, not midway through a 30-trial block.

```bash
GPU_IDS="0 1" bash run_continuation.sh
PREPARE_ONLY=1 bash run_continuation.sh
```

Preparation-only writes the verified manifest and runs tests; it does not launch
GPU training. `TOTAL_UPDATES` changes the target before preparation, and
`OUTPUT_ROOT` changes the destination. Do not change either for an existing run.

The wrapper generates training audit/learning curves automatically at completion.
It does not automatically launch the more expensive held-out analyses.

## Matched endpoint-sensitivity evaluation

After completing the common 1,500, 2,000, and 2,500-update checkpoints, evaluate
the 1,500 and 2,000 endpoints using the same held-out streams as the completed
2,500 evaluation:

```bash
GPU_IDS="0 1" bash run_endpoint_sensitivity.sh
```

The runner is resumable, evaluates intact ID and OOD rollouts, and passes
`--no-features` so that it does not save the large recurrent feature arrays. It
reuses `continuation_2500/heldout_2500_summary`, avoiding duplicate 2,500-update
evaluation. It writes a summary inside each newly evaluated continuation
directory and a combined endpoint summary under
`$HOME/biornn_publication_v2/endpoint_sensitivity_830001_summary`. Override
`EVAL_ENDPOINTS`, `REFERENCE_ENDPOINT`, `REFERENCE_SUMMARY`, `EVAL_SEED`,
`SEQUENCES`, `TRIALS`, `TAG`, or `BASE_ROOT` through environment variables. Run
the script explicitly with `bash` on systems whose login shell is `tcsh`.

## Fine-tuning-seed replication at 2,500 updates

After fixing the common endpoint at 2,500 updates, add three prospectively
declared fine-tuning seeds for both compatible parents and all seven routes:

```bash
GPU_IDS="0 1" PREPARE_ONLY=1 bash run_replication_2500.sh
GPU_IDS="0 1" bash run_replication_2500.sh
```

The default matrix is 2 parents x seeds 2, 3, and 4 x 7 routes = 42 new jobs.
Together with the completed seed-1 jobs, this provides four fine-tuning seeds
nested within each parent-route condition. The script runs tests and preflight,
uses a fixed equal 2,500-update budget, is resumable, and does not select
checkpoints by evaluation performance. On completion it creates an analysis-only
combined plan containing seeds 1-4 and writes a combined training audit. Do not
launch the combined plan; it references already completed jobs.

Override `SEEDS`, `GPU_IDS`, `OUTPUT_ROOT`, or other declared environment
variables only before the first run. Once job metadata exists, rerun the same
command and output root to resume without changing the protocol.

## Evaluate the continued models

Start with intact ID/OOD evaluation and its summary:

```bash
PLAN="$HOME/biornn_publication_v2/continuation_1000/plan.json"
python evaluate_publication.py --plan "$PLAN" --device cuda:0 \
  --sequences 6 --trials 300 --interventions intact --tag intact_check
python summarize_publication.py --plan "$PLAN" --tag intact_check \
  --output "$HOME/biornn_publication_v2/continuation_1000/intact_check_summary"
```

For the full intervention analysis, use a different evaluation tag:

```bash
python evaluate_publication.py --plan "$PLAN" --device cuda:0 \
  --sequences 6 --trials 300 --tag evaluation
python evaluate_publication.py --plan "$PLAN" --device cuda:0 \
  --sequences 6 --trials 300 --tag replay --replay --no-features \
  --interventions ablate_added block_sr_vip_sst_both_negative
python decode_publication.py --plan "$PLAN" --distribution id --permutations 20
python decode_publication.py --plan "$PLAN" --distribution ood --permutations 20
python summarize_publication.py --plan "$PLAN" \
  --output "$HOME/biornn_publication_v2/continuation_1000/summary"
python summarize_publication.py --plan "$PLAN" --tag replay \
  --output "$HOME/biornn_publication_v2/continuation_1000/replay_summary"
```

Twenty permutations remain a pilot diagnostic. This is still a two-parent,
one-fine-tuning-seed-per-parent pilot, not the final multi-seed publication matrix.
See README.md for the unchanged scientific conditions and evaluation scope.

## Verification

Tests compare an actual completed run from the bundled, unmodified pilot trainer
against uninterrupted training and continuation, including temporal jitter,
baseline/VIP/direct targeting and restricted adaptation. Model tensors, Adam
state and learning histories must match exactly on CPU. Tests also check resumed
continuation and rejection of altered/missing sources. CUDA continuation has
not been run locally; retain the existing department environment.
