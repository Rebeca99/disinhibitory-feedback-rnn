"""Sequence-held-out decoding with fold-local transforms and structured nulls."""
import argparse
import csv
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from core import stream_seed


def decoder_score(x, y, splits, pca_components, seed):
    scores, converged, component_counts = [], [], []
    for train, test in splits:
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("A held-out fold lacks both rules; collect longer/more sequences")
        steps = [("standardize", StandardScaler())]
        components = min(pca_components, x.shape[1], len(train) - 1)
        if pca_components:
            steps.append(("pca", PCA(n_components=components, svd_solver="full", random_state=seed)))
        steps.append(("classifier", LogisticRegression(C=1, penalty="l2", solver="liblinear", max_iter=1000,
                                                      class_weight=None, random_state=seed)))
        decoder = Pipeline(steps)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            decoder.fit(x[train], y[train])
        scores.append(balanced_accuracy_score(y[test], decoder.predict(x[test])))
        converged.append(not any(issubclass(w.category, ConvergenceWarning) for w in caught))
        component_counts.append(components if pca_components else 0)
    return scores, converged, component_counts


def structured_null(y, block_groups, seed):
    # Independently relabel entire true rule blocks. Every temporal replicate of
    # a block keeps the same null label; this does not shuffle individual trials.
    rng = np.random.RandomState(seed)
    labels = {block: int(rng.randint(2)) for block in np.unique(block_groups)}
    return np.asarray([labels[block] for block in block_groups])


def decode(path, folds=5, pca_components=20, permutations=20, seed=730002):
    data = np.load(path, allow_pickle=False)
    features, y, groups = data["features"], data["labels"], data["groups"]
    if len(np.unique(groups)) < folds:
        raise ValueError("Need at least {} complete evaluation sequences; found {}".format(folds, len(np.unique(groups))))
    populations = json.loads(str(data["populations"]))
    feature_groups = {"pfc": np.concatenate([v for k, v in populations.items() if k.startswith("pfc_") and k in
        ["pfc_esoma", "pfc_edend", "pfc_pv", "pfc_sst", "pfc_vip"]]),
        "sr": np.concatenate([populations[k] for k in ["sr_esoma", "sr_edend", "sr_pv", "sr_sst", "sr_vip"]]),
        "both": np.arange(features.shape[-1]),
        "pfc_motif": np.concatenate([populations[k] for k in ["pfc_vip", "pfc_sst", "pfc_edend"]]),
        "sr_motif": np.concatenate([populations[k] for k in ["sr_vip", "sr_sst", "sr_edend"]])}
    splits = list(GroupKFold(n_splits=folds).split(features[:, 0], y, groups))
    split_audit = []
    for i, (train, test) in enumerate(splits):
        train_groups, test_groups = set(groups[train]), set(groups[test])
        assert not train_groups & test_groups
        split_audit.append(dict(fold=i, train_sequences=sorted(map(int, train_groups)),
                                test_sequences=sorted(map(int, test_groups)),
                                train_trial_examples=len(train), test_trial_examples=len(test)))
    null_labels = [structured_null(y, data["block_groups"], stream_seed(seed, "null", i))
                   for i in range(permutations)]
    rows = []
    for window_index, window in enumerate(data["windows"]):
        for name, idx in feature_groups.items():
            x = features[:, window_index, idx]
            scores, converged, counts = decoder_score(x, y, splits, pca_components, seed)
            null_scores = []
            for null in null_labels:
                try:
                    null_scores.append(np.mean(decoder_score(x, null, splits, pca_components, seed)[0]))
                except ValueError:
                    continue
            mean = float(np.mean(scores))
            rows.append(dict(window=str(window), features=name, balanced_accuracy=mean,
                             fold_sd=float(np.std(scores, ddof=1)), n_sequences=len(np.unique(groups)),
                             n_trial_examples=len(y), n_features=len(idx), pca_requested=pca_components,
                             pca_actual_min=min(counts), all_folds_converged=all(converged),
                             null_mean=float(np.mean(null_scores)) if null_scores else float("nan"),
                             null_repeats=len(null_scores),
                             structured_null_p=(1 + sum(s >= mean for s in null_scores)) / (1 + len(null_scores))
                             if null_scores else float("nan")))
    return rows, split_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--tag", default="evaluation")
    p.add_argument("--distribution", choices=["id", "ood"], default="id")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--pca-components", type=int, default=20, help="0 disables PCA")
    p.add_argument("--permutations", type=int, default=20)
    p.add_argument("--job-index", type=int)
    args = p.parse_args()
    jobs = json.loads(Path(args.plan).read_text())["jobs"]
    for job in ([jobs[args.job_index]] if args.job_index is not None else jobs):
        base = Path(job["output"]) / args.tag / (args.distribution + "_intact_rollout")
        path = base / "features.npz"
        rows, audit = decode(path, args.folds, args.pca_components, args.permutations)
        prefix = "decoding_pca{}".format(args.pca_components)
        with open(base / (prefix + ".csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        from core import write_json
        write_json(base / (prefix + "_folds.json"), audit)
        print("DECODED", job["id"], args.distribution, flush=True)


if __name__ == "__main__":
    main()
