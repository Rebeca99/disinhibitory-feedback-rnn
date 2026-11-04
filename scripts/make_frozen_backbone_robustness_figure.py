#!/usr/bin/env python3
"""Create a compact frozen-backbone robustness figure and tables.

This script is meant for the thesis/manuscript control analysis in which the
recurrent/sensory backbone is frozen and only the feedback/readout parameters
are allowed to adapt. The expected output is a small control figure showing
that VIP feedback routes produce only small changes under this restriction.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


VARIANT_LABELS = {
    'baseline': 'Baseline',
    'both_vip_error': 'PFC+SR VIP',
    'pfc_vip_error': 'PFC VIP',
    'sr_vip_error': 'SR VIP',
}

VARIANT_ORDER = ['baseline', 'both_vip_error', 'pfc_vip_error', 'sr_vip_error']
ROUTE_VARIANTS = ['both_vip_error', 'pfc_vip_error', 'sr_vip_error']

COLORS = {
    'baseline': '#9ca3af',
    'both_vip_error': '#c44e52',
    'pfc_vip_error': '#7f7f7f',
    'sr_vip_error': '#4c78a8',
}

METRIC_LABELS = {
    'mean_last5_test_perf': 'Test performance',
    'mean_last5_test_rule_perf': 'Test rule performance',
    'mean_last5_train_perf': 'Train performance',
    'mean_last5_train_rule_perf': 'Train rule performance',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--frozen-root',
        default=os.path.expanduser('~/biornn_feedback_stochastic_frozen_backbone_pfc01_seeds'),
        help='Root directory for the frozen-backbone run.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory. Defaults to <frozen-root>/summaries/thesis_figures.',
    )
    parser.add_argument('--prefix', default='figure_frozen_backbone_robustness')
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_sem(std, count):
    if count is None or count <= 1 or pd.isna(std):
        return float('nan')
    return float(std) / math.sqrt(float(count))


def make_markdown_table(df):
    if df.empty:
        return ''
    rows = df.reset_index()
    columns = [str(c) for c in rows.columns]
    lines = []
    lines.append('| ' + ' | '.join(columns) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for _, row in rows.iterrows():
        values = []
        for col in rows.columns:
            value = row[col]
            if isinstance(value, float):
                values.append('{:.4f}'.format(value))
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def read_inputs(frozen_root):
    summary_path = frozen_root / 'pfc_0p1' / 'summaries' / 'training_by_variant.csv'
    delta_path = frozen_root / 'summaries' / 'pfc_scale_paired_delta_summary.csv'
    runs_path = frozen_root / 'pfc_0p1' / 'summaries' / 'training_runs.csv'

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not delta_path.exists():
        raise FileNotFoundError(delta_path)

    absolute = pd.read_csv(summary_path)
    deltas = pd.read_csv(delta_path)
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    return absolute, deltas, runs


def extract_freeze_info(runs):
    info = {
        'mode': 'feedback_readout',
        'trainable_params': 3360,
        'frozen_params': 729601,
        'trainable_names': 'rnn.w_out; rnn.w_out_rule; rnn.w_rew',
    }
    if runs.empty:
        return info
    row = runs.iloc[0]
    for key, column in [
        ('mode', 'freeze_backbone_mode'),
        ('trainable_params', 'freeze_backbone_trainable_param_count'),
        ('frozen_params', 'freeze_backbone_frozen_param_count'),
        ('trainable_names', 'freeze_backbone_trainable_names'),
    ]:
        if column in runs.columns and not pd.isna(row.get(column)):
            info[key] = row[column]
    return info


def write_table(absolute, deltas, freeze_info, output_dir):
    abs_cols = [
        'variant',
        'status',
        'n_runs',
        'mean_last5_test_perf',
        'mean_last5_test_rule_perf',
        'mean_last5_train_perf',
        'mean_last5_train_rule_perf',
    ]
    absolute_table = absolute[[c for c in abs_cols if c in absolute.columns]].copy()
    absolute_table['variant_label'] = absolute_table['variant'].map(VARIANT_LABELS).fillna(absolute_table['variant'])
    absolute_table.to_csv(output_dir / 'frozen_backbone_absolute_table.csv', index=False)

    delta_table = deltas[
        deltas['metric'].isin([
            'mean_last5_test_perf',
            'mean_last5_test_rule_perf',
            'mean_last5_train_perf',
            'mean_last5_train_rule_perf',
        ])
    ].copy()
    delta_table['variant_label'] = delta_table['variant'].map(VARIANT_LABELS).fillna(delta_table['variant'])
    delta_table['metric_label'] = delta_table['metric'].map(METRIC_LABELS).fillna(delta_table['metric'])
    delta_table.to_csv(output_dir / 'frozen_backbone_delta_table.csv', index=False)

    pivot_abs = absolute_table.set_index('variant_label')[
        ['mean_last5_test_perf', 'mean_last5_test_rule_perf', 'mean_last5_train_perf', 'mean_last5_train_rule_perf']
    ].round(4)
    pivot_delta = delta_table.pivot_table(index='variant_label', columns='metric_label', values='mean').round(4)

    md = []
    md.append('# Frozen-backbone robustness\n')
    md.append(
        'Backbone-freeze mode: `{}`. Trainable parameters: `{}`. Frozen parameters: `{}`. '
        'Trainable tensors: `{}`.\n'.format(
            freeze_info['mode'],
            freeze_info['trainable_params'],
            freeze_info['frozen_params'],
            freeze_info['trainable_names'],
        )
    )
    md.append('## Absolute performance\n')
    md.append(make_markdown_table(pivot_abs))
    md.append('\n\n## Delta versus baseline\n')
    md.append(make_markdown_table(pivot_delta))
    md.append('\n')
    (output_dir / 'frozen_backbone_robustness_table.md').write_text('\n'.join(md))


def get_value(df, variant, metric, column='mean'):
    row = df[(df['variant'] == variant) & (df['metric'] == metric)]
    if row.empty:
        return float('nan')
    return float(row[column].iloc[0])


def add_panel_label(ax, label, x=-0.18, y=1.03):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        clip_on=False,
    )


def make_figure(absolute, deltas, freeze_info, output_dir, prefix):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 140,
        'savefig.dpi': 300,
    })

    fig = plt.figure(figsize=(11.6, 4.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 1.0], wspace=0.42)
    ax_abs = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[0, 2])

    # Panel A: absolute test performance.
    metrics = ['mean_last5_test_perf', 'mean_last5_test_rule_perf']
    metric_labels = ['Overall', 'Rule']
    x = np.arange(len(VARIANT_ORDER))
    width = 0.34
    offsets = [-width / 2, width / 2]
    for midx, metric in enumerate(metrics):
        vals = []
        for variant in VARIANT_ORDER:
            row = absolute[absolute['variant'] == variant]
            vals.append(float(row[metric].iloc[0]) if not row.empty and metric in row.columns else np.nan)
        ax_abs.bar(
            x + offsets[midx],
            vals,
            width=width,
            color=['#d1d5db' if midx == 0 else COLORS[v] for v in VARIANT_ORDER],
            edgecolor='black',
            linewidth=0.5,
            label=metric_labels[midx],
            alpha=0.88,
        )
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], rotation=25, ha='right')
    ax_abs.set_ylim(0.42, 0.62)
    ax_abs.set_ylabel('Mean last-5 test performance', fontsize=12)
    ax_abs.tick_params(axis='both', labelsize=11)
    add_panel_label(ax_abs, 'a.', x=-0.25, y=1.08)
    ax_abs.set_title('Frozen-backbone performance')
    ax_abs.legend(frameon=False, loc='upper left')
    ax_abs.spines[['top', 'right']].set_visible(False)

    # Panel B: paired deltas versus baseline.
    route_x = np.arange(len(ROUTE_VARIANTS))
    delta_metrics = ['mean_last5_test_perf', 'mean_last5_test_rule_perf']
    for midx, metric in enumerate(delta_metrics):
        means = [get_value(deltas, variant, metric, 'mean') for variant in ROUTE_VARIANTS]
        sems = []
        for variant in ROUTE_VARIANTS:
            row = deltas[(deltas['variant'] == variant) & (deltas['metric'] == metric)]
            if row.empty:
                sems.append(np.nan)
            else:
                sems.append(float(row['sem'].iloc[0]) if 'sem' in row and not pd.isna(row['sem'].iloc[0]) else safe_sem(row['std'].iloc[0], row['count'].iloc[0]))
        xpos = route_x + offsets[midx]
        ax_delta.bar(
            xpos,
            means,
            width=width,
            color=['#d1d5db' if midx == 0 else COLORS[v] for v in ROUTE_VARIANTS],
            edgecolor='black',
            linewidth=0.5,
            label=metric_labels[midx],
            alpha=0.9,
        )
        ax_delta.errorbar(xpos, means, yerr=sems, fmt='none', ecolor='black', linewidth=0.8, capsize=2)
    ax_delta.axhline(0, color='black', linewidth=0.8)
    ax_delta.set_xticks(route_x)
    ax_delta.set_xticklabels([VARIANT_LABELS[v] for v in ROUTE_VARIANTS], rotation=25, ha='right')
    ax_delta.set_ylim(-0.008, 0.016)
    ax_delta.set_ylabel('Delta vs baseline', fontsize=12)
    ax_delta.tick_params(axis='both', labelsize=11)
    add_panel_label(ax_delta, 'b.', x=-0.25, y=1.08)
    ax_delta.set_title('Small route effects with frozen backbone')
    ax_delta.spines[['top', 'right']].set_visible(False)

    # Panel C: interpretation/control details.
    ax_text.set_axis_off()
    add_panel_label(ax_text, 'c.', x=-0.04)
    ax_text.set_title('Control interpretation', loc='left')
    text = (
        'Frozen recurrent/sensory backbone\n'
        '\n'
        'Trainable:\n'
        '  - readout weights\n'
        '  - rule readout weights\n'
        '  - feedback weights\n'
        '\n'
        'Frozen:\n'
        '  - recurrent circuit\n'
        '  - sensory input weights\n'
        '  - history-input weights\n'
        '  - cell-type masks\n'
        '\n'
        'Result: route-dependent gains are near zero,\n'
        'suggesting the main stochastic-task effects\n'
        'depend on circuit-level adaptation rather than\n'
        'readout-only relabelling.'
    )
    ax_text.text(
        0.02,
        0.95,
        text,
        va='top',
        ha='left',
        transform=ax_text.transAxes,
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#f8fafc', edgecolor='#cbd5e1'),
    )
    ax_text.text(
        0.02,
        0.03,
        'Mode: {}\nTrainable params: {}\nFrozen params: {}'.format(
            freeze_info['mode'],
            freeze_info['trainable_params'],
            freeze_info['frozen_params'],
        ),
        va='bottom',
        ha='left',
        fontsize=8,
        transform=ax_text.transAxes,
    )

    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'{prefix}.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    global pd
    import pandas as pd

    frozen_root = Path(args.frozen_root).expanduser().resolve()
    output_dir = ensure_dir(
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else frozen_root / 'summaries' / 'thesis_figures'
    )

    absolute, deltas, runs = read_inputs(frozen_root)
    freeze_info = extract_freeze_info(runs)
    write_table(absolute, deltas, freeze_info, output_dir)
    make_figure(absolute, deltas, freeze_info, output_dir, args.prefix)

    print('wrote frozen-backbone table to {}'.format(output_dir / 'frozen_backbone_robustness_table.md'))
    print('wrote frozen-backbone figure to {}'.format(output_dir / f'{args.prefix}.png'))


if __name__ == '__main__':
    main()
