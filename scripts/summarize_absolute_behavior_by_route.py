#!/usr/bin/env python3
"""Summarize absolute rule-readout and choice accuracy by feedback route."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VARIANT_LABELS = {
    "baseline": "Baseline",
    "pfc_vip_error": "PFC-VIP",
    "sr_vip_error": "SR-VIP",
    "both_vip_error": "PFC+SR VIP",
}
CONDITION_LABELS = {
    "all_nonfirst": "All non-first",
    "previous_error": "Previous error",
    "stable_trials": "Stable",
    "true_switch_trial": "True switch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine existing behavior_by_condition.csv files and report "
            "absolute accuracies with training seed as the independent unit."
        )
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Analysis directories containing behavior_by_condition.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_inputs(input_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for directory in input_dirs:
        path = directory / "behavior_by_condition.csv"
        if not path.exists() or path.stat().st_size == 0:
            print(f"Skipping missing or empty file: {path}")
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            print(f"Skipping empty CSV: {path}")

    if not frames:
        raise FileNotFoundError(
            "No non-empty behavior_by_condition.csv files were found."
        )
    return pd.concat(frames, ignore_index=True)


def filter_intact_evaluations(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "variant",
        "seed",
        "condition",
        "mean_perf",
        "mean_rule_perf",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    keep = df[
        df["variant"].isin(VARIANT_LABELS)
        & df["condition"].isin(CONDITION_LABELS)
    ].copy()

    if "causal_test" in keep.columns:
        keep = keep[keep["causal_test"] == "none"]
    if "weight_ablation" in keep.columns:
        keep = keep[keep["weight_ablation"] == "none"]

    if keep.empty:
        raise ValueError("No intact evaluations remained after filtering.")
    return keep


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_level = (
        df.groupby(["variant", "condition", "seed"], as_index=False)[
            ["mean_rule_perf", "mean_perf"]
        ]
        .mean()
        .rename(
            columns={
                "mean_rule_perf": "rule_readout_accuracy",
                "mean_perf": "choice_accuracy",
            }
        )
    )

    grouped = seed_level.groupby(["variant", "condition"])
    summary = grouped.agg(
        rule_mean=("rule_readout_accuracy", "mean"),
        rule_sd=("rule_readout_accuracy", "std"),
        choice_mean=("choice_accuracy", "mean"),
        choice_sd=("choice_accuracy", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    expected_seeds = 5
    incomplete = summary[summary["n_seeds"] != expected_seeds]
    if not incomplete.empty:
        print("WARNING: Some route/condition cells do not contain five seeds:")
        print(incomplete[["variant", "condition", "n_seeds"]].to_string(index=False))

    variant_order = list(VARIANT_LABELS)
    condition_order = list(CONDITION_LABELS)
    summary["variant"] = pd.Categorical(
        summary["variant"], categories=variant_order, ordered=True
    )
    summary["condition"] = pd.Categorical(
        summary["condition"], categories=condition_order, ordered=True
    )
    summary = summary.sort_values(["variant", "condition"]).reset_index(drop=True)
    return seed_level, summary


def format_latex(summary: pd.DataFrame) -> str:
    rows = []
    previous_variant = None
    for row in summary.itertuples(index=False):
        variant = str(row.variant)
        if previous_variant is not None and variant != previous_variant:
            rows.append(r"\addlinespace")
        route = VARIANT_LABELS[variant]
        condition = CONDITION_LABELS[str(row.condition)]
        rule = f"{row.rule_mean:.3f} ({row.rule_sd:.3f})"
        choice = f"{row.choice_mean:.3f} ({row.choice_sd:.3f})"
        rows.append(f"{route} & {condition} & {rule} & {choice} \\\\")
        previous_variant = variant

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption[Absolute rule-readout and choice accuracy by feedback route]
{{Absolute rule-readout and choice accuracy for each feedback-routing
condition. Values are mean (SD) across five stochastic continued-training
runs initialized from the same pretrained checkpoint. Repeated initial-rule
conditions and stochastic evaluation sequences were averaged within seed
before group-level summarization. Chance accuracy was \(0.500\) for the
two-class rule readout and \(0.333\) for the three-location choice readout.}}
\label{{tab:absolute_behaviour}}
\small
\begin{{tabular}}{{llcc}}
\toprule
\textbf{{Feedback route}} &
\textbf{{Trial class}} &
\textbf{{Rule-readout accuracy}} &
\textbf{{Choice accuracy}} \\
\midrule
Nominal chance & -- & 0.500 & 0.333 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_inputs(args.input_dirs)
    intact = filter_intact_evaluations(raw)
    seed_level, summary = summarize(intact)

    seed_path = args.output_dir / "absolute_behavior_seed_level.csv"
    summary_path = args.output_dir / "absolute_behavior_summary.csv"
    latex_path = args.output_dir / "absolute_behavior_table.tex"

    seed_level.to_csv(seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    latex_path.write_text(format_latex(summary), encoding="utf-8")

    print("\nAbsolute performance summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote {seed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
