#!/usr/bin/env python3
"""Make a compact thesis figure for hidden-rule decoding validation.

This script is plot-only. It reads outputs from analyze_rule_decoding.py and
summarizes whether a linear decoder can recover the latent WCST rule from
population activity in the stochastic hidden-switch task.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


VARIANT_LABELS = {
    'baseline': 'Baseline',
    'pfc_vip_error': 'PFC-VIP',
    'sr_vip_error': 'SR-VIP',
    'both_vip_error': 'PFC+SR VIP',
}

VARIANT_ORDER = ['baseline', 'pfc_vip_error', 'sr_vip_error', 'both_vip_error']
DELTA_VARIANT_ORDER = ['pfc_vip_error', 'sr_vip_error', 'both_vip_error']

WINDOW_LABELS = {
    'feedback': 'Feedback',
    'early_post_feedback': 'Early post',
    'pre_stim_delay': 'Pre-stim',
    'response_epoch': 'Response',
}

WINDOW_SHORT_LABELS = {
    'feedback': 'feedback',
    'early_post_feedback': 'early',
    'pre_stim_delay': 'pre-stim',
    'response_epoch': 'response',
}

FEATURE_SET_LABELS = {
    'pfc': 'PFC',
    'sr': 'SR',
    'pfc_sr': 'PFC+SR',
    'pfc_motif': 'PFC motif',
    'sr_motif': 'SR motif',
}

CONDITION_LABELS = {
    'all_nonfirst': 'All trials',
    'previous_error': 'Previous error',
    'stable_trials': 'Stable',
    'true_switch_trial': 'Switch trial',
}

COLORS = {
    'pfc_vip_error': '#7f7f7f',
    'sr_vip_error': '#4c78a8',
    'both_vip_error': '#c44e52',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--analysis-dir',
        default=os.path.expanduser('~/biornn_feedback_stochastic_bottleneck_pfc01_seeds/summaries/rule_decoding_small'),
        help='Directory containing rule_decoding_summary.csv and rule_decoding_delta_summary.csv.',
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.expanduser('~/biornn_feedback_stochastic_bottleneck_pfc01_seeds/summaries/thesis_figures'),
        help='Where to save the figure and selected table.',
    )
    parser.add_argument('--prefix', default='figure_rule_decoding_validation')
    parser.add_argument('--condition', default='all_nonfirst')
    parser.add_argument(
        '--feature-sets',
        nargs='*',
        default=['pfc', 'sr', 'pfc_sr', 'sr_motif'],
        help='Feature sets to show in the absolute/delta heatmaps.',
    )
    parser.add_argument(
        '--windows',
        nargs='*',
        default=['feedback', 'early_post_feedback', 'pre_stim_delay', 'response_epoch'],
        help='Task windows to show in the absolute/delta heatmaps.',
    )
    parser.add_argument('--selected-feature-set', default='sr_motif')
    parser.add_argument('--selected-window', default='early_post_feedback')
    parser.add_argument(
        '--restyle-from-svg',
        default=None,
        help=(
            'Optional existing SVG whose embedded heatmap rasters should be '
            'preserved while rebuilding the surrounding typography. This is '
            'useful when the full decoding CSVs are unavailable locally.'
        ),
    )
    return parser.parse_args()


def read_required(path):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError('{} is empty'.format(path))
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError('{} has no rows'.format(path))
    return df


def make_column_keys(feature_sets, windows):
    keys = []
    labels = []
    for feature_set in feature_sets:
        for window in windows:
            keys.append((feature_set, window))
            labels.append('{} {}'.format(
                FEATURE_SET_LABELS.get(feature_set, feature_set),
                WINDOW_SHORT_LABELS.get(window, window),
            ))
    return keys, labels


def pivot_matrix(df, row_values, column_keys, value_column):
    matrix = np.full((len(row_values), len(column_keys)), np.nan, dtype=float)
    for ridx, row_value in enumerate(row_values):
        for cidx, (feature_set, window) in enumerate(column_keys):
            rows = df[
                (df['variant'] == row_value)
                & (df['feature_set'] == feature_set)
                & (df['window'] == window)
            ]
            if not rows.empty:
                matrix[ridx, cidx] = float(rows[value_column].iloc[0])
    return matrix


def add_panel_label(ax, label, x=-0.08, y=1.03):
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


def annotate_heatmap(ax, matrix, fmt='{:.2f}', fontsize=9):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isnan(value):
                ax.text(j, i, fmt.format(value), ha='center', va='center', fontsize=fontsize, color='black')


def draw_heatmap(ax, matrix, row_labels, col_labels, title, cmap, vmin, vmax, cbar_label, fmt='{:.2f}', annotate=False):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha='right', fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotate:
        annotate_heatmap(ax, matrix, fmt=fmt)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(cbar_label, fontsize=11)
    return im


def selected_condition_table(delta_summary, feature_set, window):
    keep = delta_summary[
        (delta_summary['feature_set'] == feature_set)
        & (delta_summary['window'] == window)
        & (delta_summary['variant'].isin(DELTA_VARIANT_ORDER))
    ].copy()
    if keep.empty:
        return keep
    condition_order = ['previous_error', 'stable_trials', 'true_switch_trial', 'all_nonfirst']
    keep['variant_label'] = keep['variant'].map(VARIANT_LABELS).fillna(keep['variant'])
    keep['condition_label'] = keep['condition'].map(CONDITION_LABELS).fillna(keep['condition'])
    keep['variant_order'] = keep['variant'].map({v: i for i, v in enumerate(DELTA_VARIANT_ORDER)})
    keep['condition_order'] = keep['condition'].map({c: i for i, c in enumerate(condition_order)})
    keep = keep.sort_values(['variant_order', 'condition_order'])
    return keep


def plot_selected_conditions(ax, selected):
    if selected.empty:
        ax.text(0.5, 0.5, 'No selected condition rows found', ha='center', va='center')
        ax.set_axis_off()
        return

    conditions = ['previous_error', 'stable_trials', 'true_switch_trial']
    variants = [v for v in DELTA_VARIANT_ORDER if v in set(selected['variant'])]
    x = np.arange(len(conditions))
    width = 0.22
    offsets = np.linspace(-width, width, max(len(variants), 1))
    for vidx, variant in enumerate(variants):
        means = []
        sems = []
        for condition in conditions:
            rows = selected[(selected['variant'] == variant) & (selected['condition'] == condition)]
            means.append(float(rows['mean'].iloc[0]) if not rows.empty else np.nan)
            sems.append(float(rows['sem'].iloc[0]) if not rows.empty and 'sem' in rows else np.nan)
        ax.bar(
            x + offsets[vidx],
            means,
            width=width,
            yerr=sems,
            color=COLORS.get(variant, '#9ca3af'),
            edgecolor='black',
            linewidth=0.5,
            capsize=2,
            label=VARIANT_LABELS.get(variant, variant),
            alpha=0.9,
        )
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Delta decoding vs baseline', fontsize=11)
    ax.set_title('Route change in SR-motif rule decoding', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(frameon=False, fontsize=9.5, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax.spines[['top', 'right']].set_visible(False)


def embedded_heatmap_rasters(svg_path):
    import base64
    import io
    import re

    from PIL import Image

    svg = Path(svg_path).read_text()
    encoded_images = re.findall(
        r'<image xlink:href="data:image/png;base64,\s*(.*?)"[^>]*>',
        svg,
        flags=re.DOTALL,
    )
    rasters = []
    for encoded in encoded_images:
        payload = base64.b64decode(re.sub(r'\s+', '', encoded))
        raster = Image.open(io.BytesIO(payload)).convert('RGB')
        if raster.width > 100:
            # SVG embeds raster rows bottom-up because the image element is
            # displayed with a vertical transform.
            rasters.append(np.flipud(np.asarray(raster)))
    if len(rasters) < 2:
        raise ValueError('Could not locate both heatmap rasters in {}'.format(svg_path))
    return rasters[:2]


def draw_raster_heatmap(
    ax,
    raster,
    row_labels,
    col_labels,
    title,
    cmap,
    vmin,
    vmax,
    cbar_ticks,
    cbar_label,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ax.imshow(raster, aspect='auto', interpolation='nearest')
    height, width = raster.shape[:2]
    xcenters = (np.arange(len(col_labels)) + 0.5) * width / len(col_labels) - 0.5
    ycenters = (np.arange(len(row_labels)) + 0.5) * height / len(row_labels) - 0.5
    ax.set_xticks(xcenters)
    ax.set_xticklabels(col_labels, rotation=35, ha='right', fontsize=10)
    ax.set_yticks(ycenters)
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    scalar_map = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    scalar_map.set_array([])
    cbar = ax.figure.colorbar(
        scalar_map,
        ax=ax,
        fraction=0.035,
        pad=0.02,
        ticks=cbar_ticks,
    )
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(cbar_label, fontsize=11)


def restyle_existing_svg(args, output_dir):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source_svg = Path(args.restyle_from_svg).expanduser().resolve()
    if not source_svg.exists():
        raise FileNotFoundError(source_svg)
    absolute_raster, delta_raster = embedded_heatmap_rasters(source_svg)

    selected_path = source_svg.parent / 'rule_decoding_validation_selected_table.csv'
    selected = read_required(selected_path)
    column_keys, column_labels = make_column_keys(args.feature_sets, args.windows)
    del column_keys

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'figure.dpi': 140,
        'savefig.dpi': 300,
    })

    fig = plt.figure(figsize=(15.6, 7.3))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 0.95],
        width_ratios=[1.55, 1.0],
        hspace=0.50,
        wspace=0.34,
    )
    ax_abs = fig.add_subplot(grid[0, :])
    ax_delta = fig.add_subplot(grid[1, 0])
    ax_selected = fig.add_subplot(grid[1, 1])

    draw_raster_heatmap(
        ax_abs,
        absolute_raster,
        [VARIANT_LABELS[v] for v in VARIANT_ORDER],
        column_labels,
        'Hidden-rule decoding accuracy',
        cmap='YlGnBu',
        vmin=0.50,
        vmax=0.935,
        cbar_ticks=np.arange(0.50, 0.91, 0.05),
        cbar_label='Balanced accuracy',
    )
    add_panel_label(ax_abs, 'a.', x=-0.035, y=1.08)

    draw_raster_heatmap(
        ax_delta,
        delta_raster,
        [VARIANT_LABELS[v] for v in DELTA_VARIANT_ORDER],
        column_labels,
        'Route deltas relative to baseline',
        cmap='coolwarm',
        vmin=-0.275,
        vmax=0.275,
        cbar_ticks=np.arange(-0.2, 0.21, 0.1),
        cbar_label='Delta balanced accuracy',
    )
    add_panel_label(ax_delta, 'b.', x=-0.08, y=1.08)

    plot_selected_conditions(ax_selected, selected)
    add_panel_label(ax_selected, 'c.', x=-0.10, y=1.08)

    fig.suptitle('Linear validation of latent-rule representations', fontsize=15, y=0.98)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'{args.prefix}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('restyled rule-decoding figure from {}'.format(source_svg))


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.restyle_from_svg:
        restyle_existing_svg(args, output_dir)
        return

    summary = read_required(analysis_dir / 'rule_decoding_summary.csv')
    delta_summary = read_required(analysis_dir / 'rule_decoding_delta_summary.csv')

    summary_keep = summary[
        (summary['condition'] == args.condition)
        & (summary['feature_set'].isin(args.feature_sets))
        & (summary['window'].isin(args.windows))
    ].copy()
    delta_keep = delta_summary[
        (delta_summary['condition'] == args.condition)
        & (delta_summary['feature_set'].isin(args.feature_sets))
        & (delta_summary['window'].isin(args.windows))
        & (delta_summary['variant'].isin(DELTA_VARIANT_ORDER))
    ].copy()
    if summary_keep.empty:
        raise ValueError('No summary rows matched condition/features/windows.')

    column_keys, column_labels = make_column_keys(args.feature_sets, args.windows)
    row_variants = [v for v in VARIANT_ORDER if v in set(summary_keep['variant'])]
    delta_variants = [v for v in DELTA_VARIANT_ORDER if v in set(delta_keep['variant'])]

    absolute_matrix = pivot_matrix(summary_keep, row_variants, column_keys, 'mean')
    delta_matrix = pivot_matrix(delta_keep, delta_variants, column_keys, 'mean') if delta_variants else np.empty((0, len(column_keys)))

    selected = selected_condition_table(delta_summary, args.selected_feature_set, args.selected_window)
    selected.to_csv(output_dir / 'rule_decoding_validation_selected_table.csv', index=False)

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'figure.dpi': 140,
        'savefig.dpi': 300,
    })

    fig = plt.figure(figsize=(15.6, 7.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], width_ratios=[1.55, 1.0], hspace=0.50, wspace=0.34)
    ax_abs = fig.add_subplot(grid[0, :])
    ax_delta = fig.add_subplot(grid[1, 0])
    ax_selected = fig.add_subplot(grid[1, 1])

    draw_heatmap(
        ax_abs,
        absolute_matrix,
        [VARIANT_LABELS.get(v, v) for v in row_variants],
        column_labels,
        'Hidden-rule decoding accuracy',
        cmap='YlGnBu',
        vmin=0.5,
        vmax=max(0.9, np.nanmax(absolute_matrix)),
        cbar_label='Balanced accuracy',
    )
    add_panel_label(ax_abs, 'a.', x=-0.035, y=1.08)

    if delta_variants:
        max_abs = float(np.nanmax(np.abs(delta_matrix))) if np.isfinite(delta_matrix).any() else 0.05
        max_abs = max(max_abs, 0.05)
        draw_heatmap(
            ax_delta,
            delta_matrix,
            [VARIANT_LABELS.get(v, v) for v in delta_variants],
            column_labels,
            'Route deltas relative to baseline',
            cmap='coolwarm',
            vmin=-max_abs,
            vmax=max_abs,
            cbar_label='Delta balanced accuracy',
            fmt='{:+.2f}',
        )
    else:
        ax_delta.text(0.5, 0.5, 'No delta rows found', ha='center', va='center')
        ax_delta.set_axis_off()
    add_panel_label(ax_delta, 'b.', x=-0.08, y=1.08)

    plot_selected_conditions(ax_selected, selected)
    add_panel_label(ax_selected, 'c.', x=-0.10, y=1.08)

    fig.suptitle('Linear validation of latent-rule representations', fontsize=15, y=0.98)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'{args.prefix}.{ext}', bbox_inches='tight')
    plt.close(fig)

    print('wrote rule-decoding figure to {}'.format(output_dir / f'{args.prefix}.png'))
    print('wrote selected table to {}'.format(output_dir / 'rule_decoding_validation_selected_table.csv'))


if __name__ == '__main__':
    main()
