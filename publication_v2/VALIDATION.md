# Continuation-release verification (8 October 2026)

Continuation schema: publication-continuation-v1. Model/task protocol remains
publication-v2.2. Continuation experiment-code SHA-256:
`321df732b62e36f4e552c785b9bcc900ed77ed85aa573870f34ad36cea0aa026`.

## New checks passed

- 31 CPU unit/regression tests: the original 23 plus eight continuation tests.
- Actual runs from the bundled original pilot trainer continued to the same
  endpoint as uninterrupted training. Model tensors, Adam moment/step state,
  training records and validation records matched exactly, including temporal
  jitter, baseline/VIP/direct-SR routes and restricted adaptation.
- Interrupted continuation restored the correct subsequent update and matched
  uninterrupted weights, Adam state and histories.
- Source checksum/configuration guards rejected changed plans, altered final
  checkpoints, changed routes/learning rates, missing Adam state, incomplete
  source jobs and output directories overlapping the pilot.
- A two-parent x seven-route tiny CPU matrix completed through the queue wrapper,
  continuing all 14 jobs from two to four updates, then automatically generated
  cumulative audit CSVs and full learning-curve PDFs. Figures were visually
  inspected. These fixtures are implementation tests, not scientific results.
- Full 480-state, 30-trial x 10-example source training and continuation optimizer
  updates completed on CPU. Adam step counters advanced from one to two rather
  than resetting, and source learning records were retained.
- Continued checkpoints passed intact ID/OOD evaluation and sequence-held-out
  decoding integration smokes. Python and shell syntax checks passed.
- core.py, protocol.py and the legacy model files remain identical to the original
  pilot snapshot. Its original code SHA-256 remains unchanged.

CUDA continuation, full 1,000-update scientific runs and convergence are not
established locally. Use the pilot's existing department environment. Exact CPU
equivalence is not a guarantee of GPU bitwise determinism across environments.

## Original protocol verification record

The record below applies to the original pilot release and its unchanged
model/task implementation. Its code hash is not the continuation release hash.

Version: publication-v2.2.
Experiment-code SHA-256:
`c424e037efce9c1618a42263765ff9e518680c216138b50e00bcaee2e6073db8`.

Verified on CPU with Python 3.9, PyTorch 1.13.1, NumPy 1.26.4,
SciPy 1.13.1, pandas 2.3.3, scikit-learn 1.6.1, and Matplotlib 3.9.4.
CUDA was unavailable locally. No department GPU training was launched.

## Passed

- 23 unit/regression tests, including inherited forward-pass and gradient
  equivalence, recorded dendritic aliasing, unique argmax/tie reporting,
  route norms/counts, input independence from the latent rule, hazard bounds,
  corruption events, cross-trial gradients, detached history, selective edge
  blocking, actual ablated current norms, and exact interrupted/resumed weights.
- Two-update CPU training smokes for all seven core routes, the combined-norm
  control, broad direct-SR control, both motif-blocked training controls, and
  all four restricted-adaptation controls.
- Additional baseline, SR-VIP and sparse direct-dendrite smokes with the second
  independently pretrained parent.
- A full 30-trial x 10-example optimizer update, with temporal jitter and
  cross-trial backpropagation, followed by validation and final-state saving.
- Independent random-parent initialization, one full pretraining update,
  export/reload mechanics and preparation of a subsequent fine-tuning plan.
  The export mechanics used a zero threshold ONLY in this smoke test; the
  released default remains 0.75 for both accuracies. No random parent is
  presented as a successfully pretrained model in the package.
- ID/OOD evaluations, post-switch/recovery tables, true-outcome/observed-feedback
  activity decomposition, added-route removal, selective VIP-to-SST blocking,
  and common-state replay including population silencing.
- Sequence-held-out linear decoding with fold-local standardization/PCA and a
  small structured-null smoke test.
- Training/statistical summary generation and visual inspection of the learning
  curves and absolute-accuracy plot layouts.
- Python syntax checks and shell syntax checks.

## Not established by these tests

Scientific convergence, route effects, robustness across a larger parent
cohort, GPU speed/memory/determinism, department scheduler compatibility,
or empirical validation of the circuit hypothesis. Short smoke-test outputs
are not publication results and are not included as scientific findings.

## Important source-code issue

The selected local legacy additive loop overwrites dendritic state through a
shared tensor. The compatibility implementation reproduces the measured
forward pass and gradients. Check the department copy of model_working.py
before applying this finding retrospectively to the thesis simulations.
See README.md for the precise distinction and the explicit pure-additive option.
