"""Absolute results, paired effects nested within parents, and training audit figures."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from core import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--tag", default="evaluation")
    p.add_argument("--output", default="publication_summary")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    audits, absolute, intervention_rows, trajectories, decoding, norms = [], [], [], [], [], []
    incomplete = []
    for job in plan["jobs"]:
        root = Path(job["output"])
        meta = dict(parent=job["parent_id"], seed=job["seed"], route=job["route"], adaptation=job["adaptation"])
        if not (root / "final.pt").exists():
            incomplete.append(job["id"])
        if (root / "audit.json").exists():
            audit = json.loads((root / "audit.json").read_text())
            row = dict(meta, completed_updates=audit["updates"], planned_updates=audit["planned_updates"],
                       termination_reason=audit["termination_reason"], criterion_reached=audit["criterion_reached"],
                       runtime_hours=audit["elapsed_seconds"] / 3600, checkpoint_selection="final parameter state",
                       added_weight_l1=audit["feedback"]["added_l1"], added_weight_l2=audit["feedback"]["added_l2"])
            for stage in ["final_training", "last_validation"]:
                if audit[stage]:
                    row.update({stage + "_" + k: v for k, v in audit[stage].items()})
            if audit.get("continuation"):
                origin = audit["continuation"]
                row.update(source_completed_updates=origin["source_updates"],
                           continuation_updates=audit["continuation_updates"],
                           continuation_runtime_hours=audit["continuation_elapsed_seconds"] / 3600,
                           source_checkpoint=origin["source_checkpoint"],
                           source_checkpoint_sha256=origin["source_checkpoint_sha256"],
                           source_code_sha256=origin["source_code_sha256"])
            audits.append(row)
            norms.extend([dict(meta, **r) for r in audit["feedback"]["routes"]])
        for distribution in ["id", "ood"]:
            base = root / args.tag / (distribution + "_intact_rollout")
            if (base / "summary.csv").exists():
                frame = pd.read_csv(base / "summary.csv")
                frame = frame.assign(**meta, distribution=distribution)
                absolute.append(frame)
                if (base / "post_switch.csv").exists():
                    trajectories.append(pd.read_csv(base / "post_switch.csv").assign(**meta, distribution=distribution))
                for decoder_path in base.glob("decoding_pca*.csv"):
                    decoding.append(pd.read_csv(decoder_path).assign(**meta, distribution=distribution))
            for directory in (root / args.tag).glob(distribution + "_*"):
                if directory == base or not (directory / "summary.csv").exists():
                    continue
                frame = pd.read_csv(directory / "summary.csv").assign(**meta, distribution=distribution,
                                                                       intervention=directory.name)
                intervention_rows.append(frame)
    write_json(output / "completion.json", dict(incomplete=incomplete, n_planned=len(plan["jobs"]),
               n_completed=len(plan["jobs"]) - len(incomplete),
               inference_unit="parent: fine-tuning seeds averaged within parent",
               conditional_subsets="rollout previous-error subsets depend on each model's own choices"))
    pd.DataFrame(audits).to_csv(output / "training_audit.csv", index=False, float_format="%.6f")
    pd.DataFrame(norms).to_csv(output / "feedback_weights_by_route_seed.csv", index=False, float_format="%.6f")
    if trajectories:
        pd.concat(trajectories).to_csv(output / "post_switch_by_parent_seed.csv", index=False)
    if intervention_rows:
        pd.concat(intervention_rows).to_csv(output / "interventions_by_parent_seed.csv", index=False)
    if decoding:
        pd.concat(decoding).to_csv(output / "sequence_heldout_decoding_by_parent_seed.csv", index=False)
    if not absolute:
        print("No held-out evaluation yet. Training audit written to", output)
        plot_learning(plan, output)
        return
    absolute = pd.concat(absolute)
    absolute.to_csv(output / "absolute_by_parent_seed.csv", index=False, float_format="%.6f")
    keys = ["parent", "seed", "adaptation", "distribution", "condition"]
    metrics = ["choice_accuracy", "rule_accuracy"]
    baseline = absolute[absolute.route == "baseline"][keys + metrics]
    paired = absolute.merge(baseline, on=keys, how="left", suffixes=("", "_baseline"), validate="many_to_one")
    for metric in metrics:
        paired["delta_" + metric] = paired[metric] - paired[metric + "_baseline"]
    paired.to_csv(output / "paired_by_parent_seed.csv", index=False, float_format="%.6f")
    columns = metrics + ["delta_" + m for m in metrics]
    parent = paired.groupby(["parent", "adaptation", "route", "distribution", "condition"])[columns].mean().reset_index()
    parent.to_csv(output / "parent_level.csv", index=False, float_format="%.6f")
    aggregate = []
    for identity, group in parent.groupby(["adaptation", "route", "distribution", "condition"]):
        for metric in columns:
            values = group[metric].dropna().values
            n = len(values)
            mean = float(np.mean(values)) if n else float("nan")
            sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
            halfwidth = float(stats.t.ppf(0.975, n-1) * sd / np.sqrt(n)) if n > 1 else float("nan")
            aggregate.append(dict(zip(["adaptation", "route", "distribution", "condition"], identity),
                                  metric=metric, mean=mean, sd_across_parents=sd, n_parents=n,
                                  descriptive_t_ci_low=mean-halfwidth, descriptive_t_ci_high=mean+halfwidth))
    pd.DataFrame(aggregate).to_csv(output / "parent_level_summary.csv", index=False, float_format="%.6f")
    interaction = []
    for identity, group in absolute.groupby(keys):
        indexed = group.set_index("route")
        if all(r in indexed.index for r in ["baseline", "pfc_vip", "sr_vip", "both_vip"]):
            for metric in metrics:
                value = indexed.loc["both_vip", metric] - indexed.loc["pfc_vip", metric] - indexed.loc["sr_vip", metric] + indexed.loc["baseline", metric]
                interaction.append(dict(zip(keys, identity), metric=metric, interaction=value))
    pd.DataFrame(interaction).to_csv(output / "binary_interaction_by_parent_seed.csv", index=False)
    plot_learning(plan, output)
    plot_absolute(parent, output)
    print("Summary:", output.resolve())
    if incomplete:
        print("WARNING: {} training jobs incomplete. Do not treat a partial matrix as the final paired comparison.".format(len(incomplete)))


def plot_learning(plan, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output / "learning_curves.pdf") as pdf:
        for parent in plan["parents"]:
            fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
            shown = set()
            colors = plt.get_cmap("tab10")
            routes = sorted(set(j["route"] for j in plan["jobs"]))
            for job in plan["jobs"]:
                if job["parent_id"] != parent["id"] or job["adaptation"] != "full":
                    continue
                root = Path(job["output"])
                color = colors(routes.index(job["route"]) % 10)
                label = job["route"] if job["route"] not in shown else None
                if (root / "training.csv").exists():
                    t = pd.read_csv(root / "training.csv")
                    axs[0, 0].plot(t["update"], t["loss"].rolling(20, min_periods=1).mean(), color=color, alpha=.8, label=label)
                    shown.add(job["route"])
                if (root / "validation.csv").exists():
                    v = pd.read_csv(root / "validation.csv")
                    axs[0, 1].plot(v["update"], v["loss"], color=color)
                    axs[1, 0].plot(v["update"], v["choice_accuracy"], color=color)
                    axs[1, 1].plot(v["update"], v["rule_accuracy"], color=color)
            for ax, title in zip(axs.flat, ["Training loss (20-update mean)", "Fixed ID validation loss",
                                           "Validation choice accuracy", "Validation rule-readout accuracy"]):
                ax.set_title(title)
                ax.set_xlabel("Completed optimizer updates")
                ax.spines[["top", "right"]].set_visible(False)
                if plan.get("continuation"):
                    ax.axvline(plan["continuation"]["source_updates"], color="gray", ls="--", lw=.8,
                               label="Continuation boundary" if ax is axs[0, 0] else None)
            axs[1, 0].axhline(1/3, color="gray", ls=":")
            axs[1, 1].axhline(.5, color="gray", ls=":")
            axs[1, 0].set_ylim(0, 1)
            axs[1, 1].set_ylim(0, 1)
            if axs[0, 0].get_legend_handles_labels()[0]:
                axs[0, 0].legend(fontsize=8)
            fig.suptitle(parent["id"] + ": " + parent["source_name"], fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_absolute(parent, output):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    selected = parent[(parent.adaptation == "full") & (parent.distribution == "id") & (parent.condition == "all_nonfirst")]
    routes = sorted(selected.route.unique())
    for ax, metric, chance in zip(axs, ["choice_accuracy", "rule_accuracy"], [1/3, .5]):
        for x, route in enumerate(routes):
            values = selected.loc[selected.route == route, metric].values
            ax.scatter(np.full(len(values), x), values, s=30, alpha=.6)
            ax.plot([x-.2, x+.2], [np.mean(values)] * 2, color="black")
        ax.axhline(chance, ls=":", color="gray", label="Numerical chance")
        ax.set_xticks(range(len(routes)))
        ax.set_xticklabels(routes, rotation=60, ha="right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(metric.replace("_", " "))
        ax.legend()
    fig.suptitle("ID evaluation: points are parent means across fine-tuning seeds")
    fig.tight_layout()
    fig.savefig(output / "absolute_accuracy.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
