# Continue a completed pilot

Clone the repository on the cluster and enter `publication_v2` inside your
existing Python environment and persistent terminal session:

```bash
git clone https://github.com/Rebeca99/disinhibitory-feedback-rnn.git
cd disinhibitory-feedback-rnn/publication_v2
GPU_IDS="0 1" bash run_continuation.sh
```

Use only allocated GPUs. This restores all 14 completed pilot models and their
Adam optimizers and adds 700 updates each, reaching 1,000 total. It leaves
the original pilot code/checkpoints unchanged.

New results: `$HOME/biornn_publication_v2/continuation_1000/`.
Learning curves and training audit are generated automatically in
`continuation_1000/training_check/`. Held-out analysis remains a separate step.

See CONTINUATION.md for checks, resume instructions and evaluation commands.
