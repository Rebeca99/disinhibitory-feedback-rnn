# Disinhibitory RNN: publication-control version

**This copy is the continuation release.** To extend the completed 300-update
pilot to 1,000 total updates, use `run_continuation.sh` and CONTINUATION.md.
Do not overwrite the original pilot code folder. The scientific protocol below
is unchanged; `run_cluster.sh` starts new experiments, not pilot continuation.

This isolated version keeps the two-rule task and addresses the most immediate
publication controls before a separate multi-rule extension. It does not edit
the Chapter 6 scripts. It is a new experiment, not a substitute for the thesis
results. CPU smoke tests are not evidence for any scientific hypothesis.

## Start on the department GPU machine

Unpack the supplied archive in a new directory. It includes two compact parent
parameter archives, so you do not need to download the original models again.
Activate your existing `ssn-speech` environment, then:

```bash
cd ~/disinhib_rnn/publication_v2_2026-10-08
GPU_IDS="0 1" bash run_cluster.sh smoke
GPU_IDS="0 1" bash run_cluster.sh pilot
```

Use only GPUs allocated to you. Inside a scheduler allocation, omit `GPU_IDS`:
the launcher uses `CUDA_VISIBLE_DEVICES`. It runs one job per GPU, with individual
logs and resume checkpoints. A failed job does not silently count as completed.
Rerun the same command to resume; completed jobs are skipped.
Configuration and code hashes are recorded; do not edit experiment code between
interruption and resume. A deliberately changed protocol needs a new run.
Environment activation and CUDA/PyTorch compatibility are site-specific. Do not upgrade a
working CUDA environment merely to satisfy optional plotting dependencies.

`smoke`: 4 short training jobs, two parents x two routes x one seed.
`pilot`: 14 jobs, two parents x seven routes x one seed, 300 updates each.
`core`: 42 jobs, two parents x seven routes x three seeds, 1,000 updates each.
`extended`: 84 jobs, adding motif-blocked training controls, a norm-matched
combined route, and the four restricted-adaptation controls.

The pilot is for verifying learning, timings, memory and route effects; do not
treat two parents and one fine-tuning seed as the final robustness claim. The
1,000-update core budget is a starting protocol, not a claim that convergence
is assured. Inspect curves and decide on a common, longer budget before final
comparisons if learning is still changing substantially. No best-checkpoint
selection or route-specific early stopping is performed.

Output defaults to `$SCRATCH/biornn_publication_v2/<phase>`, or
`$HOME/biornn_publication_v2/<phase>` if SCRATCH is absent. Override it with
`OUTPUT_ROOT`. To prepare without training, set `PREPARE_ONLY=1`.

For Slurm, prepare the plan, add your site's partition/account to
`run_array.slurm`, and submit a job array with one index per plan job. Do not run
both launch methods on the same plan. A file lock prevents concurrent duplicate
training workers. The template requests one GPU per array element and saves on
SIGTERM at the next optimizer boundary.

## Scientific conditions

Every condition keeps both inherited feedback channels directed to PFC
dendrites/PV at fixed multiplicative gain 0.1. The added pathways use only
observed negative feedback, after the shared corruption process. Both positive
and negative history channels still reach the inherited PFC route. Previous
stimulus and choice magnitudes are 0.1; feedback magnitude is 1.

| Route | Newly enabled negative-channel targets | Initial added norm |
|---|---|---|
| baseline | none | 0 |
| pfc_vip | 10 PFC VIP | sqrt(10) x 0.1 |
| sr_vip | 10 SR VIP | sqrt(10) x 0.1 |
| both_vip | 20 VIP across modules | sqrt(20) x 0.1 |
| sr_dend10 | 10 randomly selected SR PYR dendritic compartments | sqrt(10) x 0.1 |
| sr_pv10 | 10 SR PV | sqrt(10) x 0.1 |
| sr_dend_pv10 | 5 SR dendritic compartments + 5 SR PV | sqrt(10) x 0.1 |

The five single-module added-route conditions match active connection count,
raw initialization (0.02), multiplicative gain (5), initial per-connection
coefficient (0.1), and initial added L2 norm. They do not enforce equal learned
currents or equal influence on different cell populations. Sparse dendritic
targeting is a capacity-matched control, not a complete model of cortical
feedback. `--target-seed` permits later checks of this subset choice.

Optional `both_vip_normmatched` divides the gain by sqrt(2): it matches initial
L2 norm, but still has 20 connections. Optional
`sr_direct_dense_normmatched` targets all SR dendrites/PV and normalizes its
initial L2 norm; it deliberately has more active parameters. Enable the latter
with `--with-dense-direct-control`. These conditions are not part of the original
binary PFC-by-SR interaction contrast, which uses only the four original routes.

`--with-motif-training-control` includes *both* baseline and SR-VIP variants
without SR VIP-to-SST connections throughout training and evaluation. Compare
route effects within this matched pair against the intact-motif pair; do not
interpret a lone motif-deletion network as decisive evidence for mediation.
`--with-restricted-control` trains only feedback and readout weights.

## Important inherited-forward-pass finding

The local legacy code's `additive` loop is not equivalent to simply applying
the same leaky ReLU update to every unmodified state at every step. After step
zero, `next_h` and `hidden` share a tensor. The preceding dendritic assignment
then overwrites entries used by the following soma update, and also changes
the previous recorded dendritic sample. Thus the nominal additive branch does
not make the specialized dendritic calculation entirely inactive in this code.

The default `legacy_additive` mode reproduces this forward behaviour
functionally, without in-place mutation of tensors needed by autograd. The
preflight compares its readouts and final state with the actual legacy forward
pass. This discovery warrants checking the exact model_working.py used for the
cluster thesis runs before asserting that those runs had exclusively linear
dendritic integration. It does not, by itself, establish that old numerical
results are wrong. The bundled `legacy/model_working.py` is the inspected local
snapshot, not a verified copy of your department's current file.

`--dynamics pure_additive` is an explicit *different-model* option for a later
controlled comparison. It removes that intermediate overwrite. Do not pool
it with the compatibility runs or present it as an unnoticed implementation fix.

## Parent replication

The local collection contained only **two** archives with matching architecture,
sign-mask counts, SST sparsity, and dendritic nonlinearity for the Chapter 6
parent. They have original training seeds 6 and 7. `wcst_17` was an archive/run
identifier, not necessarily the recorded training seed. `parents/provenance.json`
records source filenames, hashes, and seeds. The compact files preserve the
original learned parameter tensors; they omit pickled models and old recordings.

Do not silently mix the other checkpoints, which change circuit conditions.
To obtain additional genuinely independent parents, this version also supplies:

```bash
python fresh_parents.py initialize \
  --template parents/parent_01_wcst_6.pt --seeds 101 102 103 \
  --updates 5000 --output-root "$HOME/biornn_new_parents" \
  --plan parent_training_plan.json
python launch.py --plan parent_training_plan.json --gpus 0 1
python fresh_parents.py promote --plan parent_training_plan.json \
  --output additional_parents
```

These parents copy the architectural masks/fixed couplings, **not learned
weights**, and train from independent random initializations. Their pretraining
task has uncued 20-trial rule blocks, informative history, and no feedback
corruption. It is a declared new parent-training protocol, not a reproduction
of all Liu-Wang optimization details. Default export requires final held-out
choice and rule accuracy >= 0.75. Report failed seeds and extend their training
if appropriate; do not selectively retain parents that later favour VIP routes.

After all three pass, prepare the same controls within this fresh-parent cohort:

```bash
python prepare_plan.py --parent-dir additional_parents \
  --anchor additional_parents/fresh_parent_101.pt --parent-count 3 \
  --seeds 1 2 3 --updates 1000 --with-motif-training-control \
  --output-root "$HOME/biornn_publication_fresh" --plan fresh_controls_plan.json
python preflight.py --plan fresh_controls_plan.json
python launch.py --plan fresh_controls_plan.json --gpus 0 1
```

Analyze the original-parent and fresh-parent cohorts separately first. A shared
task across fine-tuning runs does not replace independent parent training.

## Training and reproducibility

The saved `config` and training audit are authoritative for this protocol.
`hp` retains inherited architectural metadata and can also contain unused
parent-training fields; do not infer the new training schedule from those fields.

Each update comprises 30 consecutive trials x 10 parallel card examples. As in
the inherited training organization, these examples share the block's latent
rule schedule; they are not ten independent rule sequences. Initial rule varies
between training blocks. State and gradients carry across trials within a
block. Choice, outcome and explicit history are detached; state resets between
blocks. Adam uses 1e-4, no explicit L1/L2/weight-decay penalty, and global gradient
norm clipping at 1. The MSE head losses have equal coefficients, with zero
choice targets outside the response window included.

Per-update task, corruption, and network-noise streams are paired across routes.
Validation does not advance them; resume restarts at the next update's stream.
Actual choices and hence feedback events can diverge between routes. Inputs do
not expose the current latent rule. Physical cards are sampled independently
of rule identity. Training hazard: minimum 8, maximum 22, intermediate p=0.1.
ID tests use that hazard; OOD tests use minimum 8, maximum 35, p=0.08.

Three explicit protocol changes relative to legacy analysis are audited:
single argmax produces one choice even on ties (legacy multi-hot tie behaviour
is reported alongside); previous stimulus history is extracted using the
preceding trial's actual response window, not the current jittered window;
and activity windows follow actual jittered reference/response onsets.
The parent-like preflight is an informative-history diagnostic, not an exact
reproduction of the original paper's entire test distribution.

## Evaluation, decoding, and summary

After training, use the plan path printed by run_cluster.sh. On an allocated
GPU, run whole-sequence rollouts and separate single-trial common-state replay:

```bash
PLAN="$HOME/biornn_publication_v2/pilot/plan.json"  # replace with actual output path
python evaluate_publication.py --plan "$PLAN" --device cuda:0 \
  --sequences 6 --trials 300 --tag evaluation
python evaluate_publication.py --plan "$PLAN" --device cuda:0 \
  --sequences 6 --trials 300 --tag replay --replay --no-features \
  --interventions ablate_added block_sr_vip_sst_both_negative
python decode_publication.py --plan "$PLAN" --distribution id --permutations 20
python decode_publication.py --plan "$PLAN" --distribution ood --permutations 20
python summarize_publication.py --plan "$PLAN" --output publication_summary
python summarize_publication.py --plan "$PLAN" --tag replay --output replay_summary
```

The default interventions are intact, added-feedback removal, and SR VIP-to-SST
transmission block over 0-200 ms only for examples receiving observed negative
feedback. Other accepted names include `ablate_pfc_vip`, `ablate_sr_vip`,
`block_sr_vip_sst_feedback_negative`, `block_sr_vip_sst_early_negative`,
`block_sr_vip_sst_both`, and `silence_sr_sst_feedback` (or other named population
and feedback/early/both window). Silencing zeros the updated state before noise,
as in the inherited implementation.

In a free rollout, an intervention can alter later choices and feedback. In
replay, each trial is forked from the intact preceding state/history, with
matched noise, and only that trial's effect is measured; the intact trajectory
continues afterwards. `delta_*` in replay always means intervention minus
intact on the same examples. Replay is not a test of the intervention's
cumulative effects. Rollout deltas must be computed from matched whole runs;
previous-error conditional subsets remain endogenous.

Trial tables separate true preceding errors, observed positive/negative events,
and omitted/reversed outcomes. `activity_events.csv` gives the true-outcome by
observed-feedback decomposition, with omitted events as a third category.
Activity reports both true-outcome and
observed-feedback contrasts. Post-switch curves report lag 0 (before feedback
about the new rule), subsequent accuracy, old-rule/neither choices, and
first-three-correct recovery with explicit censoring. They do not automatically
turn every old-rule choice into a diagnostic perseverative error.

Feature arrays keep individual units and examples, averaging only over time
within each window. Every example from a held-out sequence stays out of the
training folds. Scaling and optional PCA are fitted inside folds. Logistic
regression uses L2, C=1, liblinear, 1,000 iterations, no class weights. All models
share sequence folds. The block-preserving null randomly relabels complete rule
blocks, not individual trials. Twenty null repeats are a quick diagnostic;
use a larger predefined count (e.g. 200) for final reporting. Decode with
`--pca-components 0` for the no-PCA sensitivity check. Decoding is a linear
accessibility measure, not evidence that the RNN uses the decoded direction.

The Bayesian filter is a declared information-access reference. It knows the
hazard, corruption rates, exact cards and actual chosen location, and filters
each RNN's own feedback history. Choosing the neither-matching card provides no
rule discrimination: it is wrong under both rules. This is not a separately
acting optimal policy, and its information access exceeds the attenuated
explicit history delivered to the RNN. No softmax of MSE readouts is treated
as a calibrated posterior.

`training_audit.csv` includes updates, termination, final losses, validation
accuracies, and effective feedback norms. Per-trial files contain actual added
and inherited current norms. Summaries average fine-tuning seeds within each
parent before across-parent statistics; they also retain all seed values and
the four-condition interaction. The learning-curve PDF and absolute-accuracy
PDF display numerical chance. A small number of parents is still a limitation.

## Scope and deferred work

This does not claim biological local learning or fit neural recordings. It
does not yet implement the four-rule task, an upstream ACC error circuit,
branch-specific afferent alignment, or a calibrated competing behavioural
policy. Those are separate extensions after the matched-route and independent-
parent results are understood. The default test suite and CPU integration
smokes check implementation, not CUDA speed or publication conclusions.
