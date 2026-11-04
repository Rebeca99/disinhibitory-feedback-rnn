"""Combine matched held-out summaries across training endpoints."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY_CONDITIONS = ["all_nonfirst", "previous_error", "stable", "true_switch"]


def parse_summary(value):
    try:
        update_text, path_text = value.split("=", 1)
        updates = int(update_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use UPDATE=SUMMARY_DIRECTORY") from exc
    path = Path(path_text).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError("Summary directory not found: {}".format(path))
    return updates, path


def read_summary(updates, path):
    parent_path = path / "parent_level.csv"
    absolute_path = path / "absolute_by_parent_seed.csv"
    post_switch_path = path / "post_switch_by_parent_seed.csv"
    missing = [p for p in [parent_path, absolute_path, post_switch_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing summary files: {}".format(", ".join(map(str, missing))))
    parent = pd.read_csv(parent_path).assign(endpoint_updates=updates)
    absolute = pd.read_csv(absolute_path).assign(endpoint_updates=updates)
    post_switch = pd.read_csv(post_switch_path).assign(endpoint_updates=updates)
    return parent, absolute, post_switch


def aggregate_parent_results(parent):
    identity = ["endpoint_updates", "adaptation", "route", "distribution", "condition"]
    metrics = ["choice_accuracy", "rule_accuracy", "delta_choice_accuracy", "delta_rule_accuracy"]
    rows = []
    for keys, group in parent.groupby(identity, sort=True):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            rows.append(dict(
                zip(identity, keys),
                metric=metric,
                mean=float(np.mean(values)) if len(values) else float("nan"),
                sd_across_parents=float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                n_parents=len(values),
            ))
    return pd.DataFrame(rows)


def endpoint_changes(parent):
    keys = ["parent", "adaptation", "route", "distribution", "condition"]
    metrics = ["choice_accuracy", "rule_accuracy", "delta_choice_accuracy", "delta_rule_accuracy"]
    rows = []
    for identity, group in parent.groupby(keys, sort=True):
        group = group.sort_values("endpoint_updates")
        previous = None
        for row in group.itertuples(index=False):
            if previous is not None:
                result = dict(zip(keys, identity))
                result.update(from_updates=int(previous.endpoint_updates), to_updates=int(row.endpoint_updates))
                for metric in metrics:
                    result["change_" + metric] = float(getattr(row, metric) - getattr(previous, metric))
                rows.append(result)
            previous = row
    return pd.DataFrame(rows)


def plot_trajectories(parent, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    routes = sorted(parent["route"].unique())
    colors = {route: plt.get_cmap("tab10")(i % 10) for i, route in enumerate(routes)}
    parents = sorted(parent["parent"].unique())
    distributions = sorted(parent["distribution"].unique())

    with PdfPages(output / "endpoint_sensitivity.pdf") as pdf:
        for distribution in distributions:
            for parent_name in parents:
                selected = parent[
                    (parent["distribution"] == distribution)
                    & (parent["parent"] == parent_name)
                    & parent["condition"].isin(KEY_CONDITIONS)
                ]
                if selected.empty:
                    continue
                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
                for ax, condition in zip(axes.flat, KEY_CONDITIONS):
                    condition_rows = selected[selected["condition"] == condition]
                    for route in routes:
                        route_rows = condition_rows[condition_rows["route"] == route].sort_values("endpoint_updates")
                        if route_rows.empty:
                            continue
                        ax.plot(
                            route_rows["endpoint_updates"],
                            route_rows["rule_accuracy"],
                            marker="o",
                            linewidth=1.6,
                            markersize=4,
                            color=colors[route],
                            label=route,
                        )
                    ax.set_title(condition.replace("_", " "))
                    ax.set_ylabel("Rule accuracy")
                    ax.grid(alpha=0.2)
                for ax in axes[-1]:
                    ax.set_xlabel("Completed optimizer updates")
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="center right", frameon=False)
                fig.suptitle("{}: {} held-out endpoint sensitivity".format(parent_name, distribution.upper()))
                fig.tight_layout(rect=(0, 0, 0.86, 0.94))
                pdf.savefig(fig)
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        type=parse_summary,
        metavar="UPDATE=DIR",
        help="Summary directory produced by summarize_publication.py; repeat for each endpoint.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-plot", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    supplied = sorted(args.summary)
    updates = [item[0] for item in supplied]
    if len(set(updates)) != len(updates):
        parser.error("Each endpoint update count must be supplied once.")
    if len(supplied) < 2:
        parser.error("At least two endpoints are required.")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    parent_frames, absolute_frames, post_switch_frames = [], [], []
    for endpoint, path in supplied:
        parent, absolute, post_switch = read_summary(endpoint, path)
        parent_frames.append(parent)
        absolute_frames.append(absolute)
        post_switch_frames.append(post_switch)

    parent = pd.concat(parent_frames, ignore_index=True)
    absolute = pd.concat(absolute_frames, ignore_index=True)
    post_switch = pd.concat(post_switch_frames, ignore_index=True)

    parent.to_csv(output / "endpoint_parent_level.csv", index=False, float_format="%.6f")
    absolute.to_csv(output / "endpoint_absolute_by_parent_seed.csv", index=False, float_format="%.6f")
    post_switch.to_csv(output / "endpoint_post_switch_by_parent_seed.csv", index=False, float_format="%.6f")
    aggregate_parent_results(parent).to_csv(
        output / "endpoint_parent_summary.csv", index=False, float_format="%.6f"
    )
    endpoint_changes(parent).to_csv(output / "endpoint_changes.csv", index=False, float_format="%.6f")
    parent[parent["condition"].isin(KEY_CONDITIONS)].to_csv(
        output / "endpoint_key_conditions.csv", index=False, float_format="%.6f"
    )
    if not args.skip_plot:
        plot_trajectories(parent, output)

    (output / "README.txt").write_text(
        "Matched endpoint-sensitivity evaluation. All endpoints use the same evaluation seed and "
        "held-out protocol. Parent checkpoints are the replication unit; fine-tuning seeds remain "
        "nested within parent. Previous-error subsets are endogenous because each rollout can produce "
        "different choices. No inferential test is justified from two parents.\n"
    )
    print("Endpoint sensitivity summary:", output)


if __name__ == "__main__":
    main()
