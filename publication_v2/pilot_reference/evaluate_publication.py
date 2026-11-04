"""ID/OOD evaluation, post-switch curves, feedback events, and paired interventions."""
import argparse
import csv
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from core import VERSION, build_model, code_digest, digest_file, stream_seed, trusted_load, write_json
from protocol import validation

PRIMARY = ["intact", "ablate_added", "block_sr_vip_sst_both_negative"]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summaries(rows):
    metrics = ["choice_accuracy", "rule_accuracy", "legacy_choice_accuracy", "legacy_rule_accuracy",
               "choice_tie", "rule_tie", "delta_choice", "delta_rule", "chose_old_rule",
               "chose_neither", "bayes_rule_accuracy", "bayes_true_rule_probability",
               "added_current_l2", "inherited_current_l2"]
    metrics += ["intervention_added_current_l2"]
    result = []
    subsets = {"all_nonfirst": lambda r: r["nonfirst"], "previous_error": lambda r: r["previous_error"],
               "stable": lambda r: r["stable"], "true_switch": lambda r: r["true_switch"],
               "reported_true_error": lambda r: r["previous_error"] and r["observed_feedback"] == "negative",
               "false_negative_after_correct": lambda r: r["previous_correct"] is True and r["observed_feedback"] == "negative",
               "true_error_feedback_reversed": lambda r: r["previous_error"] and r["corruption"] == "reversed",
               "true_error_feedback_omitted": lambda r: r["previous_error"] and r["corruption"] == "omitted"}
    for truth in ["error", "correct"]:
        for observed in ["positive", "negative", "omitted"]:
            subsets[truth + "_observed_" + observed] = lambda r, truth=truth, observed=observed: (
                r["nonfirst"] and r["previous_error"] == (truth == "error") and r["observed_feedback"] == observed)
    for name, select in subsets.items():
        subset = [r for r in rows if select(r)]
        result.append(dict(condition=name, n_trial_examples=len(subset),
                           **{k: float(np.mean([r[k] for r in subset])) if subset else float("nan") for k in metrics}))
    return result


def switch_curves(rows, max_lag=20):
    # Exclude the first unswitched block; censor at the next true switch.
    switched = set()
    selected = []
    for row in rows:
        key = (row["sequence"], row["example"])
        if row["true_switch"]:
            switched.add(key)
        if key in switched:
            selected.append(row)
    result = []
    for lag in range(max_lag + 1):
        subset = [r for r in selected if r["age"] == lag]
        result.append(dict(post_switch_lag=lag, n_trial_examples=len(subset),
                           **{k: float(np.mean([r[k] for r in subset])) if subset else float("nan")
                              for k in ["choice_accuracy", "rule_accuracy", "chose_old_rule", "chose_neither", "delta_choice", "delta_rule"]}))
    # First run of three correct choices, reported with censoring, not dropped failures.
    blocks = {}
    for r in selected:
        key = (r["sequence"], r["example"], r["trial"] - r["age"])
        blocks.setdefault(key, []).append(r)
    recovery = []
    for key, block in blocks.items():
        block.sort(key=lambda r: r["age"])
        first = None
        for index in range(len(block) - 2):
            trio = block[index:index+3]
            if all(r["choice_accuracy"] == 1 for r in trio) and trio[-1]["age"] - trio[0]["age"] == 2:
                first = trio[0]["age"]
                break
        recovery.append(dict(sequence=key[0], example=key[1], switch_trial=key[2],
                             first_three_correct_lag=first, recovered=first is not None,
                             last_observed_lag=block[-1]["age"]))
    return result, recovery


def activity_contrasts(rows):
    columns = [k for k in rows[0] if k.startswith("activity_")] if rows else []
    groups = {"true_error_minus_correct": (
        [r for r in rows if r["previous_error"]], [r for r in rows if r["previous_correct"] is True]),
        "observed_negative_minus_positive": (
        [r for r in rows if r["observed_feedback"] == "negative"],
        [r for r in rows if r["observed_feedback"] == "positive"])}
    result = []
    for contrast, (left, right) in groups.items():
        for column in columns:
            result.append(dict(contrast=contrast, feature=column, n_left=len(left), n_right=len(right),
                               difference=float(np.mean([r[column] for r in left]) - np.mean([r[column] for r in right]))
                               if left and right else float("nan")))
    return result


def activity_events(rows):
    columns = [k for k in rows[0] if k.startswith("activity_")] if rows else []
    result = []
    for true_outcome in ["error", "correct"]:
        for observed in ["positive", "negative", "omitted"]:
            subset = [r for r in rows if r["nonfirst"] and r["previous_error"] == (true_outcome == "error")
                      and r["observed_feedback"] == observed]
            for column in columns:
                result.append(dict(true_previous_outcome=true_outcome, observed_feedback=observed,
                                   feature=column, n_trial_examples=len(subset),
                                   mean_activity=float(np.mean([r[column] for r in subset])) if subset else float("nan")))
    return result


def evaluate_job(job, args):
    root = Path(job["output"])
    if not (root / "final.pt").exists():
        raise FileNotFoundError("Training incomplete: " + str(root))
    saved = trusted_load(root / "final.pt")
    if saved["updates"] != job["config"]["updates"]:
        raise ValueError("Final checkpoint does not have the planned equal update count")
    if digest_file(job["parent"]) != job["parent_sha256"]:
        raise ValueError("Parent checksum mismatch")
    model, _ = build_model(trusted_load(job["parent"]), job["route"], job["config"],
                           job["adaptation"], args.device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    output = root / args.tag
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(version=VERSION, sequences=args.sequences, trials=args.trials, seed=args.eval_seed,
                code_sha256=code_digest(),
                interventions=args.interventions, replay=args.replay, features=not args.no_features,
                checkpoint_sha256=digest_file(root / "final.pt"))
    if (output / "evaluation.json").exists() and json.loads((output / "evaluation.json").read_text()) != spec:
        raise ValueError("Evaluation tag already has a different spec; use a new --tag")
    write_json(output / "evaluation.json", spec)
    for distribution in ["id", "ood"]:
        hazard = job["config"]["train_hazard"] if distribution == "id" else job["config"]["ood_hazard"]
        # All parents/routes/FT seeds see the same held-out exogenous streams.
        seed = stream_seed(args.eval_seed, distribution, "heldout")
        for name in args.interventions:
            mode = "replay" if args.replay and name != "intact" else "rollout"
            target = output / (distribution + "_" + name + "_" + mode)
            if (target / "complete.json").exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            result = validation(model, job["config"], seed, args.sequences, args.trials, hazard,
                                intervention=name, replay=args.replay and name != "intact",
                                collect_features=(name == "intact" and not args.no_features))
            write_csv(target / "trials.csv", result["rows"])
            write_csv(target / "summary.csv", summaries(result["rows"]))
            curves, recovery = switch_curves(result["rows"])
            write_csv(target / "post_switch.csv", curves)
            write_csv(target / "recovery.csv", recovery)
            write_csv(target / "activity_contrasts.csv", activity_contrasts(result["rows"]))
            write_csv(target / "activity_events.csv", activity_events(result["rows"]))
            if "features" in result:
                np.savez_compressed(target / "features.npz", features=result["features"], labels=result["labels"],
                                    groups=np.array([r["sequence"] for r in result["rows"]]),
                                    block_groups=np.array([r["sequence"] * (args.trials + 1) + r["trial"] - r["age"]
                                                           for r in result["rows"]]),
                                    populations=json.dumps(saved["populations"]),
                                    windows=np.array(["feedback", "early", "delay", "response"]))
            write_json(target / "complete.json", dict(loss=result["losses"][0], choice_loss=result["losses"][1],
                       rule_loss=result["losses"][2], event_subsets="network-specific in rollout; common intact events in replay",
                       bayes_scope="known-hazard, exact-card information filter on this network's own feedback history"))
            print("EVALUATED", job["id"], distribution, name, mode, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--job-index", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sequences", type=int, default=6)
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--eval-seed", type=int, default=730001)
    p.add_argument("--interventions", nargs="+", default=PRIMARY)
    p.add_argument("--replay", action="store_true", help="Common intact state and event sets; one-trial intervention only")
    p.add_argument("--no-features", action="store_true")
    p.add_argument("--tag", default="evaluation")
    args = p.parse_args()
    if min(args.sequences, args.trials) < 1:
        p.error("Sequences and trials must be positive")
    torch.set_num_threads(1)
    jobs = json.loads(Path(args.plan).read_text())["jobs"]
    selected = [jobs[args.job_index]] if args.job_index is not None else jobs
    for job in selected:
        evaluate_job(job, args)


if __name__ == "__main__":
    main()
