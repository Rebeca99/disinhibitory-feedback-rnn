#!/usr/bin/env python3
"""Create a clean three-panel stochastic-bottleneck summary figure.

This figure uses the compact summary CSVs produced by
`make_stochastic_bottleneck_results_table_figure.py` and redraws the thesis
summary with lowercase panel labels placed independently at the upper-left of
each subplot.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


VARIANT_LABELS = {
    'baseline': 'Baseline',
    'pfc_vip_error': 'PFC-VIP',
    'sr_vip_error': 'SR-VIP',
    'both_vip_error': 'PFC+SR VIP',
}

CONDITION_LABELS = {
    'previous_error': 'Previous error',
    'stable_trials': 'Stable',
    'true_switch_trial': 'Switch trial',
}

WINDOW_LABELS = {
    'feedback_only': 'Feedback',
    'early_post_only': 'Early post',
}

CAUSAL_LABELS = {
    'sr_vip_silence_feedback': 'SR-VIP',
    'sr_sst_silence_feedback': 'SR-SST',
}

POP_LABELS = {
    'sr_vip': 'SR VIP',
    'sr_sst': 'SR SST',
    'sr_edend': 'SR dend',
    'pfc_sst': 'PFC SST',
    'pfc_edend': 'PFC dend',
}

BEHAVIOR_VARIANTS = ['pfc_vip_error', 'sr_vip_error', 'both_vip_error']
BEHAVIOR_CONDITIONS = ['previous_error', 'stable_trials', 'true_switch_trial']
ACTIVITY_VARIANTS = ['baseline', 'pfc_vip_error', 'sr_vip_error', 'both_vip_error']
ACTIVITY_POPS = ['sr_vip', 'sr_sst', 'sr_edend', 'pfc_sst', 'pfc_edend']
CAUSAL_VARIANTS = ['sr_vip_error', 'both_vip_error']
CAUSAL_WINDOWS = ['feedback_only', 'early_post_only']
CAUSAL_TESTS = ['sr_vip_silence_feedback', 'sr_sst_silence_feedback']
CAUSAL_CONDITIONS = ['true_switch_trial', 'previous_error', 'stable_trials']

COLORS = {
    'pfc_vip_error': '#8b8b8b',
    'sr_vip_error': '#4c78a8',
    'both_vip_error': '#c44e52',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--tables-dir',
        default=os.path.expanduser('~/biornn_feedback_stochastic_bottleneck_pfc01_seeds/summaries/thesis_figures'),
        help='Directory containing behavior/activity/causal summary CSVs.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Where to write the figure. Defaults to --tables-dir.',
    )
    parser.add_argument('--prefix', default='stochastic_bottleneck_clean_summary')
    return parser.parse_args()


def add_panel_label(ax, label, x=-0.115, y=1.035):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=14,
        fontweight='bold',
        clip_on=False,
    )


def require_csv(tables_dir, name):
    path = tables_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_behavior(ax, behavior_summary, behavior_seed):
    width = 0.22
    offsets = [-width, 0.0, width]
    x = np.arange(len(BEHAVIOR_CONDITIONS))

    for vidx, variant in enumerate(BEHAVIOR_VARIANTS):
        means = []
        sems = []
        for condition in BEHAVIOR_CONDITIONS:
            row = behavior_summary[
                (behavior_summary['variant'] == variant) &
                (behavior_summary['condition'] == condition)
            ]
            means.append(float(row['mean'].iloc[0]) if not row.empty else np.nan)
            sems.append(float(row['sem'].iloc[0]) if not row.empty else np.nan)

        xpos = x + offsets[vidx]
        ax.bar(
            xpos,
            means,
            width=width * 0.88,
            color=COLORS[variant],
            edgecolor='black',
            linewidth=0.6,
            alpha=0.84,
            label=VARIANT_LABELS[variant],
        )
        ax.errorbar(xpos, means, yerr=sems, fmt='none', ecolor='black', linewidth=0.9, capsize=2.5)

        for cidx, condition in enumerate(BEHAVIOR_CONDITIONS):
            vals = behavior_seed[
                (behavior_seed['variant'] == variant) &
                (behavior_seed['condition'] == condition)
            ]['delta_vs_baseline'].to_numpy()
            if vals.size == 0:
                continue
            jitter = np.linspace(-0.035, 0.035, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(vals.size, xpos[cidx]) + jitter,
                vals,
                s=18,
                color='white',
                edgecolor='#333333',
                linewidth=0.7,
                zorder=4,
            )

    ax.axhline(0, color='#333333', linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in BEHAVIOR_CONDITIONS],
        rotation=25,
        ha='right',
        fontsize=12,
    )
    ax.set_ylabel('Delta rule perf. vs baseline', fontsize=14)
    ax.set_title('Feedback route benefit under stochastic hidden switches', pad=8, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(-0.26, 0.26)
    ax.spines[['top', 'right']].set_visible(False)
    add_panel_label(ax, 'a.', x=-0.12)


def plot_activity_heatmap(ax, fig, activity_summary):
    matrix = []
    for variant in ACTIVITY_VARIANTS:
        row = []
        for pop in ACTIVITY_POPS:
            vals = activity_summary[
                (activity_summary['variant'] == variant) &
                (activity_summary['window'] == 'early_post_feedback') &
                (activity_summary['population'] == pop)
            ]
            row.append(float(vals['mean'].iloc[0]) if not vals.empty else np.nan)
        matrix.append(row)
    matrix = np.asarray(matrix, dtype=float)

    im = ax.imshow(matrix, cmap='coolwarm', vmin=-0.12, vmax=0.12, aspect='auto')
    ax.set_xticks(np.arange(len(ACTIVITY_POPS)))
    ax.set_xticklabels([POP_LABELS[p] for p in ACTIVITY_POPS], rotation=25, ha='right', fontsize=12)
    ax.set_yticks(np.arange(len(ACTIVITY_VARIANTS)))
    ax.set_yticklabels([VARIANT_LABELS[v] for v in ACTIVITY_VARIANTS], fontsize=12)
    ax.set_title(
        'Activity following true errors\nversus correct choices',
        loc='left',
        pad=8,
        fontsize=14,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.035)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('After true error - after correct choice', fontsize=13)
    add_panel_label(ax, 'b.', x=-0.16)


def causal_group_label(window, causal_test, condition):
    return '{}\n{}\n{}'.format(
        WINDOW_LABELS[window],
        CAUSAL_LABELS[causal_test],
        CONDITION_LABELS[condition],
    )


def plot_causal(ax, causal_summary, causal_seed):
    groups = [
        (window, causal_test, condition)
        for window in CAUSAL_WINDOWS
        for causal_test in CAUSAL_TESTS
        for condition in CAUSAL_CONDITIONS
    ]
    width = 0.28
    offsets = [-width / 2, width / 2]
    x = np.arange(len(groups))

    for vidx, variant in enumerate(CAUSAL_VARIANTS):
        means = []
        sems = []
        for window, causal_test, condition in groups:
            row = causal_summary[
                (causal_summary['window_run'] == window) &
                (causal_summary['variant'] == variant) &
                (causal_summary['causal_test'] == causal_test) &
                (causal_summary['condition'] == condition)
            ]
            means.append(float(row['mean'].iloc[0]) if not row.empty else np.nan)
            sems.append(float(row['sem'].iloc[0]) if not row.empty else np.nan)

        xpos = x + offsets[vidx]
        ax.bar(
            xpos,
            means,
            width=width * 0.82,
            color=COLORS[variant],
            edgecolor='black',
            linewidth=0.6,
            alpha=0.82,
            label=VARIANT_LABELS[variant],
        )
        ax.errorbar(xpos, means, yerr=sems, fmt='none', ecolor='black', linewidth=0.9, capsize=2.4)

        for gidx, (window, causal_test, condition) in enumerate(groups):
            vals = causal_seed[
                (causal_seed['window_run'] == window) &
                (causal_seed['variant'] == variant) &
                (causal_seed['causal_test'] == causal_test) &
                (causal_seed['condition'] == condition)
            ]['delta_vs_intact'].to_numpy()
            if vals.size == 0:
                continue
            jitter = np.linspace(-0.035, 0.035, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(vals.size, xpos[gidx]) + jitter,
                vals,
                s=18,
                color='white',
                edgecolor='#333333',
                linewidth=0.7,
                zorder=4,
            )

    ax.axhline(0, color='#333333', linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [causal_group_label(*group) for group in groups],
        rotation=45,
        ha='right',
        fontsize=11.5,
    )
    ax.set_ylabel('Delta rule perf., silenced - intact', fontsize=14)
    ax.set_title('Seed-level causal stability-plasticity effects', pad=8, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(frameon=False, loc='upper left')
    ax.set_ylim(-0.32, 0.58)
    ax.spines[['top', 'right']].set_visible(False)
    add_panel_label(ax, 'c.', x=-0.055)


def main():
    args = parse_args()
    global np, pd
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    tables_dir = Path(args.tables_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else tables_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    behavior_summary = require_csv(tables_dir, 'behavior_delta_summary.csv')
    behavior_seed = require_csv(tables_dir, 'behavior_delta_seed_level.csv')
    activity_summary = require_csv(tables_dir, 'activity_motif_summary.csv')
    causal_summary = require_csv(tables_dir, 'causal_delta_summary.csv')
    causal_seed = require_csv(tables_dir, 'causal_delta_seed_level.csv')
    causal_seed = causal_seed[causal_seed['metric'] == 'mean_rule_perf'].copy()

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 140,
        'savefig.dpi': 300,
    })

    fig = plt.figure(figsize=(13.2, 9.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.22, 1.0], hspace=0.46, wspace=0.30)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    fig.subplots_adjust(top=0.97)

    plot_behavior(ax_a, behavior_summary, behavior_seed)
    plot_activity_heatmap(ax_b, fig, activity_summary)
    plot_causal(ax_c, causal_summary, causal_seed)

    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'{args.prefix}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('wrote clean summary figure to {}'.format(output_dir / f'{args.prefix}.png'))


if __name__ == '__main__':
    main()
