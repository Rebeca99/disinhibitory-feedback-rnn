# Disinhibitory feedback routing in a biologically constrained RNN

This repository extends the model introduced by Liu and Wang in
[Flexible gating between subspaces in a neural network model of internally
guided task switching](https://www.nature.com/articles/s41467-024-50501-y).
It studies how routing noisy negative feedback to different cell populations
and circuit modules affects hidden-rule inference, task behaviour, population
activity, and the recurrent solution acquired during continued supervised
training.

The original model and analysis notebooks remain in the repository history.
The extensions here were developed for the disinhibitory-feedback RNN project
and are organized into two complementary workflows.

## Repository map

- `functions.py`, `model_working.py`, and `train.py`: extensions to the parent
  implementation used by the thesis-era experiments.
- `scripts/`: fine-tuning, noisy hidden-switch tasks, feedback-routing controls,
  activity analyses, pathway ablation, restricted-window population silencing,
  linear decoding, statistical audits, and figure-generation scripts.
- `publication_v2/`: the newer reproducible workflow for matched feedback-route
  controls, two compatible parent checkpoints, continuation to fixed training
  endpoints, held-out evaluation, endpoint sensitivity, and additional
  fine-tuning-seed replication.
- `publication_v2/parents/`: compact parameter archives for the two compatible
  Liu-Wang parent networks used in the current extension. File hashes and
  source identifiers are recorded in `provenance.json`.
- `publication_v2/tests/`: regression and integration tests for the current
  workflow.

Generated model states, logs, feature arrays, and summaries are deliberately
excluded from version control. The repository contains code, compact parent
parameters, protocol documentation, and provenance, not completed experiment
outputs.

## Current experimental design

The current workflow compares baseline, PFC-VIP, SR-VIP, combined PFC+SR-VIP,
and matched direct sensorimotor feedback controls. It supports:

- stochastic hidden-rule switches and corrupted or omitted feedback;
- continued supervised training from either compatible parent network;
- fixed-budget training and endpoint-sensitivity analysis;
- intact rollouts and within-network evaluation-time interventions;
- post-switch behavioural trajectories and route-specific activity contrasts;
- sequence-held-out linear decoding of latent-rule information;
- replication across fine-tuning seeds nested within each parent network.

The model tests the consequences of cell-type-specific feedback access. It is
not a biologically local learning model and its population dynamics have not
been fitted to cell-type-resolved neural recordings.

## Running the current workflow

Start with [`publication_v2/README.md`](publication_v2/README.md) and
[`publication_v2/VALIDATION.md`](publication_v2/VALIDATION.md). A short cluster
smoke test should precede longer GPU runs. Use only GPUs allocated by the local
scheduler or system administrator.

The two compact parent files are approximately 2.8 MB each and are tracked
directly. Their SHA-256 hashes should be verified against
`publication_v2/parents/provenance.json` before running experiments.

## Parent project

The original Liu-Wang pretrained networks and preprocessed data are available
from the authors' [Google Drive archive](https://drive.google.com/drive/folders/17LqvmcBynX0a4OtUgnkODGJKK_JFRB2Q?usp=drive_link).
The parent code was tested by its authors with Python 3.10.8 and PyTorch 1.13.1.
Environment requirements for this extension are listed in
`publication_v2/requirements.txt`.

## Attribution and license

Original BioRNN_WCST code: Copyright 2024 Yue Liu.

The parent code is distributed under the Apache License, Version 2.0. Changes
and additional experiment code in this branch retain that license and should be
cited together with the Liu-Wang paper and the corresponding project paper or
thesis chapter when used.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy at
<https://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.
