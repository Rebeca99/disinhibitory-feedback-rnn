#!/usr/bin/env python3
"""Audit training endpoints, learning curves, and final feedback currents."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "biornn_mplconfig")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(REPO_ROOT))

# These imports are required when torch unpickles saved full-model objects.
import model_working  # noqa: F401,E402
import train  # noqa: F401,E402


VARIANT_ORDER = [
    "baseline",
    "pfc_vip_error",
    "sr_vip_error",
    "both_vip_error",
]

VARIANT_LABELS = {
    "baseline": "Baseline",
    "pfc_vip_error": "PFC-VIP",
    "sr_vip_error": "SR-VIP",
    "both_vip_error": "PFC+SR VIP",
}

VARIANT_COLORS = {
    "baseline": "#8C8C8C",
    "pfc_vip_error": "#555555",
    "sr_vip_error": "#4C78A8",
    "both_vip_error": "#C65D62",
}

TERMINATION_LABELS = {
    "success": "performance criterion",
    "noConverge": "update limit",
    "timesup": "runtime limit",
    "gradExplode": "gradient instability",
}

ROUTE_GROUPS = {
    "pfc_direct": ("pfc_edend", "pfc_pv"),
    "pfc_vip": ("pfc_vip",),
    "sr_vip": ("sr_vip",),
    "added_vip_total": ("pfc_vip", "sr_vip"),
}

CHANNELS = {
    0: "positive",
    1: "negative",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        required=True,
        help="Training directory containing saved_models/ (for example pfc_0p1).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to TRAINING_ROOT/summaries/training_audit.",
    )
    parser.add_argument(
        "--model-glob",
        default="*feedback_stochastic_hidden_switch*",
        help="Glob applied inside saved_models/.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=VARIANT_ORDER,
        choices=VARIANT_ORDER,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--tail-n",
        type=int,
        default=5,
        help="Number of endpoint observations used for late-window means.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Moving-average window used only for the learning-curve figure.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every requested variant-seed combination is present exactly once.",
    )
    return parser.parse_args()


def torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=torch.device("cpu"))


def infer_variant(hp):
    pfc_vip = bool(hp.get("prev_rew_to_pfc_vip", False))
    sr_vip = bool(hp.get("prev_rew_to_sr_vip", False))
    if pfc_vip and sr_vip:
        return "both_vip_error"
    if pfc_vip:
        return "pfc_vip_error"
    if sr_vip:
        return "sr_vip_error"
    return "baseline"


def infer_seed(hp):
    jobname = str(hp.get("jobname", ""))
    match = re.search(r"seed(\d+)", jobname)
    if match:
        return int(match.group(1))
    seed = hp.get("torch_seed")
    return int(seed) if seed is not None else None


def status_from_path(path):
    for status in TERMINATION_LABELS:
        if path.name.startswith(status + "_") or path.name == status:
            return status
    return path.name.split("_", 1)[0]


def as_float_array(values):
    if values is None:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def endpoint(values):
    arr = as_float_array(values)
    return float(arr[-1]) if arr.size else np.nan


def tail_mean(values, n):
    arr = as_float_array(values)
    return float(np.nanmean(arr[-n:])) if arr.size else np.nan


def get_rnn(saved):
    model = saved.get("model")
    if model is None:
        raise KeyError("Saved model does not contain the full 'model' object.")
    return model.rnn


def combined_indices(rnn, populations):
    arrays = []
    for population in populations:
        if population not in rnn.cg_idx:
            raise KeyError("Cell group {!r} is absent from saved model.".format(population))
        idx = rnn.cg_idx[population]
        if torch.is_tensor(idx):
            idx = idx.detach().cpu().numpy()
        arrays.append(np.asarray(idx, dtype=int).reshape(-1))
    return np.unique(np.concatenate(arrays))


def feedback_rows(saved, file_name, variant, seed):
    rnn = get_rnn(saved)
    raw = rnn.w_rew.detach().cpu()
    mask = rnn.mask_rew.detach().cpu()
    effective = (torch.abs(raw) * mask).detach().cpu()
    rows = []

    for channel_idx, channel in CHANNELS.items():
        for route, populations in ROUTE_GROUPS.items():
            idx_np = combined_indices(rnn, populations)
            idx = torch.as_tensor(idx_np, dtype=torch.long)
            raw_values = raw[channel_idx, idx]
            mask_values = mask[channel_idx, idx]
            effective_values = effective[channel_idx, idx]
            active = mask_values != 0
            active_raw = raw_values[active]
            active_effective = effective_values[active]

            rows.append(
                {
                    "file": file_name,
                    "variant": variant,
                    "seed": seed,
                    "feedback_channel": channel,
                    "route": route,
                    "populations": ";".join(populations),
                    "n_available_targets": int(idx.numel()),
                    "n_enabled_targets": int(active.sum().item()),
                    "routing_mask_gain_mean_enabled": (
                        float(mask_values[active].mean().item()) if active.any() else 0.0
                    ),
                    "raw_parameter_abs_mean_enabled": (
                        float(active_raw.abs().mean().item()) if active.any() else 0.0
                    ),
                    "raw_parameter_l2_enabled": (
                        float(torch.linalg.vector_norm(active_raw).item())
                        if active.any()
                        else 0.0
                    ),
                    "effective_weight_abs_mean_enabled": (
                        float(active_effective.abs().mean().item()) if active.any() else 0.0
                    ),
                    "effective_weight_l1": float(effective_values.abs().sum().item()),
                    "effective_weight_l2": float(
                        torch.linalg.vector_norm(effective_values).item()
                    ),
                    "effective_weight_max_abs": (
                        float(effective_values.abs().max().item())
                        if effective_values.numel()
                        else 0.0
                    ),
                    # For a unit one-hot feedback channel, input @ W_eff equals
                    # this route-specific effective-weight vector.
                    "unit_feedback_current_l1": float(
                        effective_values.abs().sum().item()
                    ),
                    "unit_feedback_current_l2": float(
                        torch.linalg.vector_norm(effective_values).item()
                    ),
                }
            )
    return rows


def audit_row(path, saved, variant, seed, tail_n):
    hp = saved.get("hp", {})
    status = status_from_path(path)
    train_choice = as_float_array(saved.get("perf_list", []))
    train_rule = as_float_array(saved.get("perf_rule_list", []))
    train_loss = as_float_array(saved.get("loss_list", []))
    eval_choice = as_float_array(saved.get("test_perf_list", []))
    eval_rule = as_float_array(saved.get("test_perf_rule_list", []))
    eval_loss = as_float_array(saved.get("test_loss_list", []))

    lengths = {train_choice.size, train_rule.size, train_loss.size}
    history_lengths_match = len(lengths) == 1

    return {
        "file": path.name,
        "jobname": hp.get("jobname", ""),
        "variant": variant,
        "condition": VARIANT_LABELS[variant],
        "seed": seed,
        "status": status,
        "termination_reason": TERMINATION_LABELS.get(status, "unknown"),
        "convergence_criterion_reached": status == "success",
        "checkpoint_selection": "final parameter state at termination",
        "completed_updates": int(train_loss.size),
        "last_recorded_update_index": int(train_loss.size - 1) if train_loss.size else -1,
        "training_history_lengths_match": history_lengths_match,
        "n_periodic_evaluations": int(eval_loss.size),
        "final_train_choice_accuracy": endpoint(train_choice),
        "final_train_rule_head_accuracy": endpoint(train_rule),
        "final_train_loss": endpoint(train_loss),
        "late_train_choice_accuracy": tail_mean(train_choice, tail_n),
        "late_train_rule_head_accuracy": tail_mean(train_rule, tail_n),
        "late_train_loss": tail_mean(train_loss, tail_n),
        "final_evaluation_choice_accuracy": endpoint(eval_choice),
        "final_evaluation_rule_head_accuracy": endpoint(eval_rule),
        "final_evaluation_loss": endpoint(eval_loss),
        "late_evaluation_choice_accuracy": tail_mean(eval_choice, tail_n),
        "late_evaluation_rule_head_accuracy": tail_mean(eval_rule, tail_n),
        "late_evaluation_loss": tail_mean(eval_loss, tail_n),
        "learning_rate": hp.get("learning_rate", np.nan),
        "configured_max_train_batches": hp.get("max_train_batches", np.nan),
        "configured_max_runtime_seconds": hp.get("max_runtime_seconds", np.nan),
        "block_length_trials": hp.get("block_len", np.nan),
        "minibatch_size": hp.get("batch_size", np.nan),
        "torch_seed": hp.get("torch_seed", np.nan),
        "feedback_noise_probability": hp.get("feedback_noise_prob", np.nan),
        "feedback_flip_fraction": hp.get("feedback_noise_flip_fraction", np.nan),
        "pfc_feedback_gain": hp.get("prev_rew_to_pfc_scale", np.nan),
        "vip_feedback_gain": hp.get("prev_rew_to_vip_scale", np.nan),
    }


def history_rows(saved, variant, seed):
    metric_values = {
        "train_choice_accuracy": saved.get("perf_list", []),
        "train_rule_head_accuracy": saved.get("perf_rule_list", []),
        "train_loss": saved.get("loss_list", []),
    }
    rows = []
    for metric, values in metric_values.items():
        for update, value in enumerate(as_float_array(values), start=1):
            rows.append(
                {
                    "variant": variant,
                    "condition": VARIANT_LABELS[variant],
                    "seed": seed,
                    "update": update,
                    "metric": metric,
                    "value": float(value),
                }
            )
    return rows


def evaluation_rows(saved, variant, seed):
    metric_values = {
        "evaluation_choice_accuracy": saved.get("test_perf_list", []),
        "evaluation_rule_head_accuracy": saved.get("test_perf_rule_list", []),
        "evaluation_loss": saved.get("test_loss_list", []),
    }
    rows = []
    for metric, values in metric_values.items():
        for event, value in enumerate(as_float_array(values), start=1):
            rows.append(
                {
                    "variant": variant,
                    "condition": VARIANT_LABELS[variant],
                    "seed": seed,
                    "evaluation_event": event,
                    "metric": metric,
                    "value": float(value),
                }
            )
    return rows


def mean_sd(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else np.nan


def aggregate_audit(seed_df, variants):
    metrics = [
        "completed_updates",
        "n_periodic_evaluations",
        "final_train_choice_accuracy",
        "final_train_rule_head_accuracy",
        "final_train_loss",
        "late_train_choice_accuracy",
        "late_train_rule_head_accuracy",
        "late_train_loss",
        "final_evaluation_choice_accuracy",
        "final_evaluation_rule_head_accuracy",
        "final_evaluation_loss",
        "late_evaluation_choice_accuracy",
        "late_evaluation_rule_head_accuracy",
        "late_evaluation_loss",
    ]
    rows = []
    for variant in variants:
        group = seed_df[seed_df["variant"] == variant]
        if group.empty:
            continue
        row = {
            "variant": variant,
            "condition": VARIANT_LABELS[variant],
            "n_runs": int(len(group)),
            "n_converged": int(group["convergence_criterion_reached"].sum()),
            "termination_reasons": "; ".join(
                "{} ({})".format(reason, int(count))
                for reason, count in group["termination_reason"].value_counts().items()
            ),
            "checkpoint_selection": "final parameter state at termination",
            "completed_updates_min": int(group["completed_updates"].min()),
            "completed_updates_max": int(group["completed_updates"].max()),
            "periodic_evaluations_min": int(group["n_periodic_evaluations"].min()),
            "periodic_evaluations_max": int(group["n_periodic_evaluations"].max()),
        }
        for metric in metrics:
            mean, sd = mean_sd(group[metric])
            row[metric + "_mean"] = mean
            row[metric + "_sd"] = sd
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_feedback(feedback_df, variants):
    rows = []
    for variant in variants:
        subset = feedback_df[feedback_df["variant"] == variant]
        for channel in ["positive", "negative"]:
            for route in ROUTE_GROUPS:
                group = subset[
                    (subset["feedback_channel"] == channel)
                    & (subset["route"] == route)
                ]
                if group.empty:
                    continue
                row = {
                    "variant": variant,
                    "condition": VARIANT_LABELS[variant],
                    "feedback_channel": channel,
                    "route": route,
                    "n_seeds": int(group["seed"].nunique()),
                }
                for metric in [
                    "n_enabled_targets",
                    "routing_mask_gain_mean_enabled",
                    "raw_parameter_abs_mean_enabled",
                    "raw_parameter_l2_enabled",
                    "effective_weight_abs_mean_enabled",
                    "effective_weight_l1",
                    "effective_weight_l2",
                    "unit_feedback_current_l1",
                    "unit_feedback_current_l2",
                ]:
                    mean, sd = mean_sd(group[metric])
                    row[metric + "_mean"] = mean
                    row[metric + "_sd"] = sd
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_learning_curves(history_df):
    grouped = (
        history_df.groupby(["variant", "condition", "metric", "update"], as_index=False)[
            "value"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    return grouped


def smooth(values, window):
    if window <= 1:
        return np.asarray(values, dtype=float)
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def plot_learning_curves(curves, variants, output_dir, smooth_window):
    metric_specs = [
        ("train_choice_accuracy", "Training choice accuracy", (0.0, 1.0)),
        ("train_rule_head_accuracy", "Training rule-head accuracy", (0.0, 1.0)),
        ("train_loss", "Training loss", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65), constrained_layout=True)

    for panel_idx, (ax, (metric, title, ylim)) in enumerate(zip(axes, metric_specs)):
        for variant in variants:
            group = curves[
                (curves["variant"] == variant) & (curves["metric"] == metric)
            ].sort_values("update")
            if group.empty:
                continue
            x = group["update"].to_numpy(dtype=float)
            mean = smooth(group["mean"], smooth_window)
            std = smooth(group["std"].fillna(0.0), smooth_window)
            color = VARIANT_COLORS[variant]
            ax.plot(
                x,
                mean,
                color=color,
                linewidth=1.8,
                label=VARIANT_LABELS[variant],
            )
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Parameter update")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.text(
            -0.16,
            1.06,
            "{}.".format(chr(ord("a") + panel_idx)),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
        )

    axes[0].set_ylabel("Accuracy")
    axes[2].set_ylabel("Mean-squared error")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    caption = (
        "Lines show means across training seeds; shaded bands show one standard "
        "deviation. Curves use a {}-update centred moving average for display."
    ).format(smooth_window)
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8)
    fig.savefig(output_dir / "training_learning_curves.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "training_learning_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def format_mean_sd(mean, sd, digits=3):
    if pd.isna(mean):
        return "--"
    if pd.isna(sd):
        return "{:.{d}f}".format(mean, d=digits)
    return "{:.{d}f} ({:.{d}f})".format(mean, sd, d=digits)


def format_value(value, digits=3):
    if pd.isna(value):
        return "--"
    return "{:.{d}f}".format(value, d=digits)


def update_range(row):
    low = int(row["completed_updates_min"])
    high = int(row["completed_updates_max"])
    return str(low) if low == high else "{}--{}".format(low, high)


def write_chapter_table(path, aggregate_df):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption[Training audit across feedback-routing conditions]{%",
        r"Training endpoints across feedback-routing conditions. Accuracy and loss",
        r"values are the last available periodic-evaluation measurements recorded",
        r"during training, reported as mean (SD) across independently trained",
        r"seeds. The final parameter state at termination was used for every model;",
        r"no best-checkpoint selection was applied. Training loss is the final",
        r"recorded optimization loss.}",
        r"\label{tab:training_audit}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccllll}",
        r"\toprule",
        r"\textbf{Condition} & \textbf{\(n\)} & \textbf{Updates} &",
        r"\textbf{Criterion} & \textbf{Termination} & \textbf{Train loss} &",
        r"\textbf{Eval. loss} & \textbf{Choice acc.} &",
        r"\textbf{Rule-head acc.} \\",
        r"\midrule",
    ]
    for _, row in aggregate_df.iterrows():
        lines.append(
            "{} & {} & {} & {}/{} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["condition"]),
                int(row["n_runs"]),
                update_range(row),
                int(row["n_converged"]),
                int(row["n_runs"]),
                latex_escape(row["termination_reasons"]),
                format_mean_sd(
                    row["final_train_loss_mean"],
                    row["final_train_loss_sd"],
                ),
                format_mean_sd(
                    row["final_evaluation_loss_mean"],
                    row["final_evaluation_loss_sd"],
                ),
                format_mean_sd(
                    row["final_evaluation_choice_accuracy_mean"],
                    row["final_evaluation_choice_accuracy_sd"],
                ),
                format_mean_sd(
                    row["final_evaluation_rule_head_accuracy_mean"],
                    row["final_evaluation_rule_head_accuracy_sd"],
                ),
            )
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    )
    path.write_text("\n".join(lines))


def feedback_value(summary_df, variant, route):
    match = summary_df[
        (summary_df["variant"] == variant)
        & (summary_df["feedback_channel"] == "negative")
        & (summary_df["route"] == route)
    ]
    if match.empty:
        return np.nan, np.nan
    row = match.iloc[0]
    return row["unit_feedback_current_l2_mean"], row["unit_feedback_current_l2_sd"]


def write_feedback_table(path, feedback_summary, variants, n_seeds):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption[Post-training feedback-current norms]{%",
        r"Post-training feedback-current magnitudes by routing condition.",
        r"Values are the \(L_2\) norm of the current produced by a unit observed",
        r"negative-feedback input, reported as mean (SD) across five training",
        r"seeds. These values describe the final implemented routes and were not",
        r"constrained to match across conditions.}",
        r"\label{tab:feedback_current_norms}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Condition} & \textbf{PFC direct} & \textbf{PFC-VIP} &",
        r"\textbf{SR-VIP} & \textbf{Added VIP total} \\",
        r"\midrule",
    ]
    for variant in variants:
        values = [
            feedback_value(feedback_summary, variant, route)
            for route in ["pfc_direct", "pfc_vip", "sr_vip", "added_vip_total"]
        ]
        lines.append(
            "{} & {} & {} & {} & {} \\\\".format(
                latex_escape(VARIANT_LABELS[variant]),
                *[format_mean_sd(mean, sd) for mean, sd in values],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    text = "\n".join(lines).replace(
        "across five training", "across {} training".format(n_seeds)
    )
    path.write_text(text)


def write_seed_table(path, seed_df, feedback_df):
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption[Seed-level training audit]{Seed-level training endpoints.",
        r"Evaluation columns show the last available periodic-evaluation measurement",
        r"recorded during training.",
        r"The final parameter state at termination was analysed in every run.}",
        r"\label{tab:training_audit_seed_level}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrlrrrr}",
        r"\toprule",
        r"\textbf{Condition} & \textbf{Seed} & \textbf{Updates} &",
        r"\textbf{Eval. points} & \textbf{Termination} & \textbf{Train loss} &",
        r"\textbf{Eval. loss} & \textbf{Choice acc.} &",
        r"\textbf{Rule-head acc.} \\",
        r"\midrule",
    ]
    for _, row in seed_df.iterrows():
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["condition"]),
                int(row["seed"]),
                int(row["completed_updates"]),
                int(row["n_periodic_evaluations"]),
                latex_escape(row["termination_reason"]),
                format_value(row["final_train_loss"]),
                format_value(row["final_evaluation_loss"]),
                format_value(row["final_evaluation_choice_accuracy"]),
                format_value(row["final_evaluation_rule_head_accuracy"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_feedback_seed_table(path, feedback_df, variants):
    subset = feedback_df[feedback_df["feedback_channel"] == "negative"]
    pivot = subset.pivot_table(
        index=["variant", "seed"],
        columns="route",
        values="unit_feedback_current_l2",
        aggfunc="first",
    ).reset_index()
    order = {variant: idx for idx, variant in enumerate(variants)}
    pivot["condition"] = pivot["variant"].map(VARIANT_LABELS)
    pivot["variant_order"] = pivot["variant"].map(order)
    pivot = pivot.sort_values(["variant_order", "seed"])

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption[Seed-level post-training feedback currents]{Seed-level",
        r"post-training feedback-current magnitudes. Values are \(L_2\) norms of",
        r"the current produced by a unit observed negative-feedback input,",
        r"restricted to each route.}",
        r"\label{tab:feedback_current_norms_seed_level}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Condition} & \textbf{Seed} & \textbf{PFC direct} &",
        r"\textbf{PFC-VIP} & \textbf{SR-VIP} & \textbf{Added VIP total} \\",
        r"\midrule",
    ]
    for _, row in pivot.iterrows():
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["condition"]),
                int(row["seed"]),
                format_value(row.get("pfc_direct")),
                format_value(row.get("pfc_vip")),
                format_value(row.get("sr_vip")),
                format_value(row.get("added_vip_total")),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def write_results_text(path, seed_df):
    n_models = len(seed_df)
    n_converged = int(seed_df["convergence_criterion_reached"].sum())
    reasons = seed_df["termination_reason"].value_counts()
    updates = sorted(seed_df["completed_updates"].unique().tolist())
    if len(updates) == 1:
        update_text = "{} parameter updates".format(updates[0])
    else:
        update_text = "{}--{} parameter updates".format(min(updates), max(updates))
    reason_text = ", ".join(
        "{} runs at the {}".format(int(count), reason)
        for reason, count in reasons.items()
    )
    text = (
        "Across the {n_models} models included in the main analysis, each run "
        "completed {updates}. The convergence criterion was reached by "
        "{n_converged} runs; termination comprised {reasons}. The final "
        "parameter state at termination was retained in every case, without "
        "selection of the best-performing intermediate checkpoint. Absolute "
        "final training loss and the last available periodic-evaluation choice "
        "accuracy, rule-head accuracy, and loss are reported in "
        "Table~\\ref{{tab:training_audit}}.\n"
    ).format(
        n_models=n_models,
        updates=update_text,
        n_converged=n_converged,
        reasons=reason_text,
    )
    path.write_text(text)


def expected_pairs(variants, seeds):
    return {(variant, seed) for variant in variants for seed in seeds}


def write_audit_report(path, seed_df, feedback_df, variants, seeds, model_glob):
    observed = set(zip(seed_df["variant"], seed_df["seed"]))
    expected = expected_pairs(variants, seeds)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = (
        seed_df.groupby(["variant", "seed"]).size().loc[lambda values: values > 1]
    )
    no_evaluation = seed_df.loc[
        seed_df["n_periodic_evaluations"] == 0, ["variant", "seed"]
    ]

    lines = [
        "TRAINING AND FEEDBACK AUDIT",
        "",
        "Model glob: {}".format(model_glob),
        "Requested variants: {}".format(", ".join(variants)),
        "Requested seeds: {}".format(", ".join(map(str, seeds))),
        "Models included: {}".format(len(seed_df)),
        "Expected models: {}".format(len(expected)),
        "",
        "Termination counts:",
        seed_df["termination_reason"].value_counts().to_string(),
        "",
        "Completed-update counts:",
        seed_df.groupby("variant")["completed_updates"]
        .agg(["min", "max", "count"])
        .to_string(),
        "",
        "Periodic-evaluation counts:",
        seed_df.groupby("variant")["n_periodic_evaluations"]
        .agg(["min", "max", "count"])
        .to_string(),
        "",
        "Missing variant-seed pairs: {}".format(missing or "none"),
        "Extra variant-seed pairs: {}".format(extra or "none"),
        "Duplicate variant-seed pairs: {}".format(
            duplicates.to_dict() if not duplicates.empty else "none"
        ),
        "Runs without periodic evaluations: {}".format(
            no_evaluation.to_dict("records") if not no_evaluation.empty else "none"
        ),
        "",
        "Definitions:",
        "- completed_updates is the number of recorded optimizer-update entries.",
        "- final metrics use the last recorded training or periodic-evaluation value.",
        "- checkpoint selection is the final parameter state saved at termination.",
        "- unit_feedback_current_l2 is ||u W_eff||_2 for a unit one-hot feedback",
        "  channel, restricted to the named route; W_eff = |W_raw| * routing mask.",
        "- periodic evaluations were triggered conditionally during training. They",
        "  are not treated as a formal validation set or used for checkpoint selection.",
        "- sharing a seed label does not by itself prove that route conditions saw",
        "  identical online-generated sequences or corruption draws.",
        "",
        "Feedback rows included: {}".format(len(feedback_df)),
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    training_root = Path(args.training_root).expanduser().resolve()
    saved_models_dir = training_root / "saved_models"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else training_root / "summaries" / "training_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not saved_models_dir.exists():
        raise FileNotFoundError("saved_models directory not found: {}".format(saved_models_dir))

    model_paths = sorted(path for path in saved_models_dir.glob(args.model_glob) if path.is_file())
    if not model_paths:
        raise FileNotFoundError(
            "No saved models matched {!r} in {}".format(args.model_glob, saved_models_dir)
        )

    records = []
    for path in model_paths:
        saved = torch_load(path)
        hp = saved.get("hp", {})
        variant = infer_variant(hp)
        seed = infer_seed(hp)
        if variant not in args.variants or seed not in args.seeds:
            continue
        records.append((path, saved, variant, seed))

    if not records:
        raise RuntimeError("Models were found, but none matched the requested variants and seeds.")

    observed_pairs = [(variant, seed) for _, _, variant, seed in records]
    duplicate_pairs = sorted(
        pair for pair in set(observed_pairs) if observed_pairs.count(pair) > 1
    )
    missing_pairs = sorted(expected_pairs(args.variants, args.seeds) - set(observed_pairs))
    if duplicate_pairs:
        raise RuntimeError("Duplicate variant-seed models found: {}".format(duplicate_pairs))
    if args.require_complete and missing_pairs:
        raise RuntimeError("Missing requested variant-seed models: {}".format(missing_pairs))

    audit_rows = []
    feedback = []
    histories = []
    evaluations = []
    for path, saved, variant, seed in records:
        audit_rows.append(audit_row(path, saved, variant, seed, args.tail_n))
        feedback.extend(feedback_rows(saved, path.name, variant, seed))
        histories.extend(history_rows(saved, variant, seed))
        evaluations.extend(evaluation_rows(saved, variant, seed))

    seed_df = pd.DataFrame(audit_rows)
    seed_df["variant_order"] = seed_df["variant"].map(
        {variant: idx for idx, variant in enumerate(args.variants)}
    )
    seed_df = seed_df.sort_values(["variant_order", "seed"]).drop(
        columns="variant_order"
    )
    feedback_df = pd.DataFrame(feedback)
    history_df = pd.DataFrame(histories)
    evaluation_df = pd.DataFrame(evaluations)

    aggregate_df = aggregate_audit(seed_df, args.variants)
    feedback_summary = aggregate_feedback(feedback_df, args.variants)
    learning_summary = aggregate_learning_curves(history_df)

    seed_df.to_csv(output_dir / "training_audit_seed_level.csv", index=False)
    aggregate_df.to_csv(output_dir / "training_audit_by_condition.csv", index=False)
    history_df.to_csv(output_dir / "training_history_seed_level.csv", index=False)
    learning_summary.to_csv(output_dir / "training_learning_curve_summary.csv", index=False)
    evaluation_df.to_csv(output_dir / "periodic_evaluation_history_seed_level.csv", index=False)
    feedback_df.to_csv(output_dir / "feedback_norms_seed_level.csv", index=False)
    feedback_summary.to_csv(output_dir / "feedback_norms_by_condition.csv", index=False)

    write_chapter_table(output_dir / "training_audit_chapter_table.tex", aggregate_df)
    write_feedback_table(
        output_dir / "feedback_norms_chapter_table.tex",
        feedback_summary,
        args.variants,
        len(args.seeds),
    )
    write_seed_table(
        output_dir / "training_audit_seed_table.tex",
        seed_df,
        feedback_df,
    )
    write_feedback_seed_table(
        output_dir / "feedback_norms_seed_table.tex",
        feedback_df,
        args.variants,
    )
    write_results_text(output_dir / "training_audit_results_text.tex", seed_df)
    plot_learning_curves(
        learning_summary,
        args.variants,
        output_dir,
        args.smooth_window,
    )
    write_audit_report(
        output_dir / "audit_report.txt",
        seed_df,
        feedback_df,
        args.variants,
        args.seeds,
        args.model_glob,
    )

    manifest = {
        "training_root": str(training_root),
        "saved_models_dir": str(saved_models_dir),
        "output_dir": str(output_dir),
        "model_glob": args.model_glob,
        "variants": args.variants,
        "seeds": args.seeds,
        "n_models": len(seed_df),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== Training audit by condition ===")
    display_columns = [
        "condition",
        "n_runs",
        "completed_updates_min",
        "completed_updates_max",
        "n_converged",
        "termination_reasons",
        "final_evaluation_choice_accuracy_mean",
        "final_evaluation_rule_head_accuracy_mean",
        "final_evaluation_loss_mean",
    ]
    print(aggregate_df[display_columns].round(4).to_string(index=False))
    print("\n=== Unit negative-feedback current L2 norms ===")
    current_display = feedback_summary[
        feedback_summary["feedback_channel"] == "negative"
    ][
        [
            "condition",
            "route",
            "n_seeds",
            "n_enabled_targets_mean",
            "unit_feedback_current_l2_mean",
            "unit_feedback_current_l2_sd",
        ]
    ]
    print(current_display.round(4).to_string(index=False))
    print("\nWrote training audit to {}".format(output_dir))


if __name__ == "__main__":
    main()
