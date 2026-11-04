#!/usr/bin/env python3
"""Audit and summarize thesis results using training seed as the replicate."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


TABLE_SPECS = {
    "behavior_delta_seed_level.csv": {
        "family": "behavior",
        "value": "delta_vs_baseline",
        "groups": ["variant", "condition"],
    },
    "activity_motif_seed_level.csv": {
        "family": "activity",
        "value": "error_minus_correct_activity",
        "groups": ["variant", "window", "population"],
    },
    "causal_delta_seed_level.csv": {
        "family": "causal",
        "value": "delta_vs_intact",
        "groups": ["window_run", "variant", "causal_test", "metric", "condition"],
    },
    "weight_ablation_delta_seed_level.csv": {
        "family": "route_ablation",
        "value": "delta_vs_intact",
        "groups": ["variant", "weight_ablation", "metric", "condition"],
    },
    "rule_decoding_delta_seed_level.csv": {
        "family": "rule_decoding",
        "value": "delta_vs_baseline",
        "groups": ["variant", "feature_set", "window", "condition"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify that inferential summaries use one value per training seed "
            "and calculate seed-level descriptive and one-sample statistics."
        )
    )
    parser.add_argument(
        "--tables-dir",
        required=True,
        type=Path,
        help="Directory containing one or more recognized seed-level CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <tables-dir>/seed_level_statistics.",
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Training-seed identifiers expected in every comparison.",
    )
    parser.add_argument(
        "--route-ablation-file",
        type=Path,
        default=None,
        help=(
            "Optional raw weight_ablation_behavior_deltas.csv. Repeated "
            "evaluations are averaged within training seed before inference."
        ),
    )
    parser.add_argument(
        "--decoding-file",
        type=Path,
        default=None,
        help="Optional rule_decoding_delta_seed_level.csv from the decoder analysis.",
    )
    return parser.parse_args()


def finite_values(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def safe_wilcoxon(values):
    if stats is None or len(values) < 2 or np.allclose(values, 0.0):
        return np.nan
    try:
        return float(stats.wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def safe_sign_test(values):
    if stats is None:
        return np.nan
    nonzero = values[~np.isclose(values, 0.0)]
    if len(nonzero) == 0:
        return np.nan
    n_positive = int((nonzero > 0).sum())
    return float(
        stats.binomtest(
            n_positive,
            n=len(nonzero),
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def summarize(values):
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    sem = sd / math.sqrt(n) if n > 1 else np.nan
    if stats is not None and n > 1 and np.isfinite(sem):
        critical = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = mean - critical * sem
        ci_high = mean + critical * sem
        ttest_p = float(stats.ttest_1samp(values, 0.0).pvalue)
    else:
        ci_low = np.nan
        ci_high = np.nan
        ttest_p = np.nan

    return {
        "n_seeds": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median": float(np.median(values)) if n else np.nan,
        "minimum": float(np.min(values)) if n else np.nan,
        "maximum": float(np.max(values)) if n else np.nan,
        "n_negative": int((values < 0).sum()),
        "n_zero": int(np.isclose(values, 0.0).sum()),
        "n_positive": int((values > 0).sum()),
        "ttest_p": ttest_p,
        "wilcoxon_p": safe_wilcoxon(values),
        "sign_test_p": safe_sign_test(values),
    }


def benjamini_hochberg(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return adjusted

    ordered = valid[np.argsort(values[valid])]
    m = len(ordered)
    running = 1.0
    for rank_from_end, index in enumerate(ordered[::-1], start=1):
        rank = m - rank_from_end + 1
        running = min(running, values[index] * m / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def audit_table(path, spec, expected_seeds):
    df = pd.read_csv(path)
    required = set(spec["groups"] + ["seed", spec["value"]])
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "{} is missing columns: {}".format(path, ", ".join(missing_columns))
        )

    df = df.copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df[spec["value"]] = pd.to_numeric(df[spec["value"]], errors="coerce")
    df = df.dropna(subset=["seed", spec["value"]])
    df["seed"] = df["seed"].astype(int)

    duplicate_keys = spec["groups"] + ["seed"]
    duplicates = df.duplicated(duplicate_keys, keep=False)
    duplicate_rows = int(duplicates.sum())
    if duplicate_rows:
        df = (
            df.groupby(duplicate_keys, as_index=False, dropna=False)[spec["value"]]
            .mean()
        )

    rows = []
    audit_rows = []
    expected = set(expected_seeds)
    grouped = df.groupby(spec["groups"], dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = dict(zip(spec["groups"], keys))
        observed = set(group["seed"].astype(int))
        values = finite_values(group[spec["value"]])

        row = {
            "family": spec["family"],
            "source_file": path.name,
            **labels,
            "seeds": " ".join(str(x) for x in sorted(observed)),
            **summarize(values),
        }
        rows.append(row)

        audit_rows.append(
            {
                "family": spec["family"],
                "source_file": path.name,
                **labels,
                "status": (
                    "ok"
                    if observed == expected and len(values) == len(expected)
                    else "check"
                ),
                "n_seed_rows": len(values),
                "missing_seeds": " ".join(str(x) for x in sorted(expected - observed)),
                "unexpected_seeds": " ".join(
                    str(x) for x in sorted(observed - expected)
                ),
                "duplicate_rows_averaged_in_source": duplicate_rows,
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["ttest_q_bh_within_family"] = benjamini_hochberg(
            summary["ttest_p"]
        )
        summary["wilcoxon_q_bh_within_family"] = benjamini_hochberg(
            summary["wilcoxon_p"]
        )
    return summary, pd.DataFrame(audit_rows)


def prepare_route_ablation(path, output_dir):
    df = pd.read_csv(path)
    groups = ["variant", "weight_ablation", "metric", "condition", "seed"]
    required = set(groups + ["delta_vs_intact"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "{} is missing columns: {}".format(path, ", ".join(missing))
        )
    seed_level = (
        df.groupby(groups, as_index=False, dropna=False)["delta_vs_intact"]
        .mean()
    )
    output_path = output_dir / "weight_ablation_delta_seed_level.csv"
    seed_level.to_csv(output_path, index=False)
    return output_path


def write_report(output_dir, summaries, audits, scipy_available):
    lines = [
        "# Seed-level statistical audit",
        "",
        "Training seed is the inferential unit. Initial rules and repeated test "
        "sequences are averaged within seed before group statistics are calculated.",
        "",
        "- Expected seeds: 1, 2, 3, 4, 5",
        "- Inferential sample size for a complete comparison: n = 5",
        "- Repeated evaluation rows (for example, 30 or 50 rows) are not treated "
        "as independent observations.",
        "- SciPy inferential tests available: {}".format(
            "yes" if scipy_available else "no"
        ),
        "",
    ]

    if audits.empty:
        lines.append("No recognized seed-level input tables were found.")
    else:
        problem = audits[audits["status"] != "ok"]
        lines.append(
            "Audit status: {} comparison(s) checked; {} require attention.".format(
                len(audits), len(problem)
            )
        )
        lines.append("")
        if len(problem):
            lines.append(
                "See `seed_level_audit.csv` for missing seeds, unexpected seeds, "
                "or duplicate seed rows."
            )
            lines.append("")

    lines.extend(
        [
            "The two-sided t-test, Wilcoxon signed-rank test, and exact sign test "
            "are supplied as descriptive support. With only five independent "
            "training seeds, effect magnitude, confidence intervals, and sign "
            "consistency should receive greater emphasis than thresholded p-values.",
            "",
            "Benjamini-Hochberg q-values are calculated separately within each "
            "result family and should be reported if many related tests are "
            "interpreted simultaneously.",
            "",
            "Files:",
            "- `seed_level_statistics.csv`",
            "- `seed_level_audit.csv`",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    tables_dir = args.tables_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else tables_dir / "seed_level_statistics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    audits = []
    table_paths = {
        filename: tables_dir / filename for filename in TABLE_SPECS
    }
    if args.route_ablation_file is not None:
        table_paths["weight_ablation_delta_seed_level.csv"] = (
            prepare_route_ablation(
                args.route_ablation_file.expanduser().resolve(),
                output_dir,
            )
        )
    if args.decoding_file is not None:
        table_paths["rule_decoding_delta_seed_level.csv"] = (
            args.decoding_file.expanduser().resolve()
        )

    for filename, spec in TABLE_SPECS.items():
        path = table_paths[filename]
        if not path.exists() or path.stat().st_size == 0:
            continue
        summary, audit = audit_table(path, spec, args.expected_seeds)
        summaries.append(summary)
        audits.append(audit)
        print("audited {}".format(path))

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    audit_df = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    summary_df.to_csv(output_dir / "seed_level_statistics.csv", index=False)
    audit_df.to_csv(output_dir / "seed_level_audit.csv", index=False)
    write_report(output_dir, summary_df, audit_df, stats is not None)

    print("wrote {}".format(output_dir / "seed_level_statistics.csv"))
    print("wrote {}".format(output_dir / "seed_level_audit.csv"))
    print("wrote {}".format(output_dir / "README.md"))

    if not audit_df.empty and (audit_df["status"] != "ok").any():
        print("WARNING: one or more comparisons failed the seed-count audit")


if __name__ == "__main__":
    main()
