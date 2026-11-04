#!/usr/bin/env python3
"""Create manuscript-style summary tables and figures for stochastic bottleneck results."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


VARIANT_LABELS = {
    'pfc_vip_error': 'PFC-VIP',
    'sr_vip_error': 'SR-VIP',
    'both_vip_error': 'PFC+SR VIP',
    'baseline': 'Baseline',
}
CONDITION_LABELS = {
    'all_nonfirst': 'All non-first',
    'previous_error': 'Previous error',
    'stable_trials': 'Stable',
    'true_switch_trial': 'Switch trial',
}
POP_LABELS = {
    'sr_vip': 'SR VIP',
    'sr_sst': 'SR SST',
    'sr_edend': 'SR dend',
    'pfc_vip': 'PFC VIP',
    'pfc_sst': 'PFC SST',
    'pfc_edend': 'PFC dend',
}
CAUSAL_LABELS = {
    'sr_vip_silence_feedback': 'silence SR VIP',
    'sr_sst_silence_feedback': 'silence SR SST',
    'pfc_vip_silence_feedback': 'silence PFC VIP',
    'pfc_sst_silence_feedback': 'silence PFC SST',
}
WINDOW_LABELS = {
    'feedback_only': 'Feedback',
    'early_post_only': 'Early post',
    'feedback_plus_early': 'Feedback + early',
}
MANUSCRIPT_VARIANTS = ['pfc_vip_error', 'sr_vip_error', 'both_vip_error']
ACTIVITY_VARIANTS = ['baseline', 'pfc_vip_error', 'sr_vip_error', 'both_vip_error']
BEHAVIOR_CONDITIONS = ['previous_error', 'stable_trials', 'true_switch_trial']
CAUSAL_CONDITIONS = ['true_switch_trial', 'previous_error', 'stable_trials']
CAUSAL_WINDOWS = ['feedback_only', 'early_post_only', 'feedback_plus_early']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--summaries-root',
        default=os.path.expanduser('~/biornn_feedback_stochastic_bottleneck_pfc01_seeds/summaries'),
        help='Root summaries directory from the stochastic hidden-switch run.',
    )
    parser.add_argument(
        '--behavior-dir',
        default='allvariants_paired_r5_n300_2gpu',
        help='Subdirectory containing paired_behavior_deltas_vs_baseline.csv and activity_error_minus_correct.csv.',
    )
    parser.add_argument(
        '--causal-dir',
        default='seed_level_causal_summary',
        help='Subdirectory containing seed_level_causal_deltas.csv.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for tables and figures. Defaults to <summaries-root>/paper_tables_figures.',
    )
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def sem(series):
    count = series.count()
    if count <= 1:
        return np.nan
    return float(series.std(ddof=1) / math.sqrt(count))


def summarize_seed_level(df, group_cols, value_col):
    seed_cols = group_cols + ['seed']
    seed_level = df.groupby(seed_cols, as_index=False)[value_col].mean()
    grouped = seed_level.groupby(group_cols)[value_col]
    summary = grouped.agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = grouped.apply(sem).to_numpy()
    return seed_level, summary


def read_behavior_summary(summaries_root, behavior_dir, output_dir):
    path = summaries_root / behavior_dir / 'paired_behavior_deltas_vs_baseline.csv'
    seed_path = output_dir / 'behavior_delta_seed_level.csv'
    summary_path = output_dir / 'behavior_delta_summary.csv'
    if not path.exists():
        if seed_path.exists() and summary_path.exists():
            return pd.read_csv(seed_path), pd.read_csv(summary_path)
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[
        (df['metric'] == 'mean_rule_perf') &
        (df['condition'].isin(['all_nonfirst', 'previous_error', 'stable_trials', 'true_switch_trial'])) &
        (df['variant'].isin(['pfc_vip_error', 'sr_vip_error', 'both_vip_error']))
    ].copy()
    seed_level, summary = summarize_seed_level(
        df,
        ['variant', 'condition'],
        'delta_vs_baseline',
    )
    seed_level.to_csv(output_dir / 'behavior_delta_seed_level.csv', index=False)
    summary.to_csv(output_dir / 'behavior_delta_summary.csv', index=False)
    return seed_level, summary


def read_activity_summary(summaries_root, behavior_dir, output_dir):
    path = summaries_root / behavior_dir / 'activity_error_minus_correct.csv'
    seed_path = output_dir / 'activity_motif_seed_level.csv'
    summary_path = output_dir / 'activity_motif_summary.csv'
    if not path.exists():
        if seed_path.exists() and summary_path.exists():
            return pd.read_csv(seed_path), pd.read_csv(summary_path)
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    keep_pops = ['sr_vip', 'sr_sst', 'sr_edend', 'pfc_vip', 'pfc_sst', 'pfc_edend']
    df = df[
        (df['causal_test'] == 'none') &
        (df['weight_ablation'] == 'none') &
        (df['variant'].isin(['baseline', 'pfc_vip_error', 'sr_vip_error', 'both_vip_error'])) &
        (df['population'].isin(keep_pops)) &
        (df['window'].isin(['feedback', 'early_post_feedback', 'pre_stim_delay', 'response_epoch']))
    ].copy()
    seed_level, summary = summarize_seed_level(
        df,
        ['variant', 'window', 'population'],
        'error_minus_correct_activity',
    )
    seed_level.to_csv(output_dir / 'activity_motif_seed_level.csv', index=False)
    summary.to_csv(output_dir / 'activity_motif_summary.csv', index=False)
    return seed_level, summary


def read_causal_summary(summaries_root, causal_dir, output_dir):
    path = summaries_root / causal_dir / 'seed_level_causal_deltas.csv'
    seed_path = output_dir / 'causal_delta_seed_level.csv'
    summary_path = output_dir / 'causal_delta_summary.csv'
    if not path.exists():
        if seed_path.exists() and summary_path.exists():
            return pd.read_csv(seed_path), pd.read_csv(summary_path)
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[
        (df['metric'] == 'mean_rule_perf') &
        (df['condition'].isin(['previous_error', 'stable_trials', 'true_switch_trial'])) &
        (df['variant'].isin(['pfc_vip_error', 'sr_vip_error', 'both_vip_error'])) &
        (df['causal_test'].isin(['sr_vip_silence_feedback', 'sr_sst_silence_feedback', 'pfc_vip_silence_feedback', 'pfc_sst_silence_feedback'])) &
        (df['window_run'].isin(['feedback_only', 'early_post_only', 'feedback_plus_early']))
    ].copy()
    grouped = df.groupby(['window_run', 'variant', 'causal_test', 'condition'])['delta_vs_intact']
    summary = grouped.agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = grouped.apply(sem).to_numpy()
    df.to_csv(output_dir / 'causal_delta_seed_level.csv', index=False)
    summary.to_csv(output_dir / 'causal_delta_summary.csv', index=False)
    return df, summary


def format_mean_sem(mean, sem_value, digits=3):
    if pd.isna(mean):
        return ''
    if pd.isna(sem_value):
        return f'{mean:.{digits}f}'
    return f'{mean:.{digits}f} +/- {sem_value:.{digits}f}'


def make_markdown_table(rows, columns):
    lines = []
    lines.append('| ' + ' | '.join(columns) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(col, '')) for col in columns) + ' |')
    return '\n'.join(lines) + '\n'


def write_compact_table(behavior_summary, activity_summary, causal_summary, output_dir):
    rows = []

    for variant in ['pfc_vip_error', 'sr_vip_error', 'both_vip_error']:
        for condition in ['previous_error', 'stable_trials', 'true_switch_trial']:
            row = behavior_summary[
                (behavior_summary['variant'] == variant) &
                (behavior_summary['condition'] == condition)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append({
                'Result': 'Behavior: rule perf vs baseline',
                'Variant': VARIANT_LABELS.get(variant, variant),
                'Measure': CONDITION_LABELS.get(condition, condition),
                'Effect': format_mean_sem(r['mean'], r['sem']),
                'n': int(r['count']),
            })

    for variant in ['baseline', 'pfc_vip_error', 'sr_vip_error', 'both_vip_error']:
        for population in ['sr_vip', 'sr_sst', 'sr_edend', 'pfc_sst', 'pfc_edend']:
            row = activity_summary[
                (activity_summary['variant'] == variant) &
                (activity_summary['window'] == 'early_post_feedback') &
                (activity_summary['population'] == population)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append({
                'Result': 'Activity: error minus correct',
                'Variant': VARIANT_LABELS.get(variant, variant),
                'Measure': '{} early post'.format(POP_LABELS.get(population, population)),
                'Effect': format_mean_sem(r['mean'], r['sem']),
                'n': int(r['count']),
            })

    for variant in ['sr_vip_error', 'both_vip_error']:
        for causal_test in ['sr_vip_silence_feedback', 'sr_sst_silence_feedback']:
            for window_run in ['feedback_only', 'early_post_only', 'feedback_plus_early']:
                for condition in ['true_switch_trial', 'previous_error', 'stable_trials']:
                    row = causal_summary[
                        (causal_summary['variant'] == variant) &
                        (causal_summary['causal_test'] == causal_test) &
                        (causal_summary['window_run'] == window_run) &
                        (causal_summary['condition'] == condition)
                    ]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    rows.append({
                        'Result': 'Causal: silenced minus intact',
                        'Variant': VARIANT_LABELS.get(variant, variant),
                        'Measure': '{}; {}; {}'.format(
                            WINDOW_LABELS.get(window_run, window_run),
                            CAUSAL_LABELS.get(causal_test, causal_test),
                            CONDITION_LABELS.get(condition, condition),
                        ),
                        'Effect': format_mean_sem(r['mean'], r['sem']),
                        'n': int(r['count']),
                    })

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / 'core_results_table.csv', index=False)
    md = make_markdown_table(table.to_dict('records'), ['Result', 'Variant', 'Measure', 'Effect', 'n'])
    with open(output_dir / 'core_results_table.md', 'w') as handle:
        handle.write(md)
    return table


def write_manuscript_tables(behavior_summary, activity_summary, causal_summary, output_dir):
    """Write compact tables that can be pasted into notes or a manuscript draft."""
    sections = []

    behavior_rows = []
    for variant in MANUSCRIPT_VARIANTS:
        row = {'Feedback route': VARIANT_LABELS[variant]}
        for condition in BEHAVIOR_CONDITIONS:
            vals = behavior_summary[
                (behavior_summary['variant'] == variant) &
                (behavior_summary['condition'] == condition)
            ]
            if vals.empty:
                row[CONDITION_LABELS[condition]] = ''
                continue
            r = vals.iloc[0]
            row[CONDITION_LABELS[condition]] = format_mean_sem(r['mean'], r['sem'])
        behavior_rows.append(row)
    behavior_table = pd.DataFrame(behavior_rows)
    behavior_table.to_csv(output_dir / 'manuscript_behavior_table.csv', index=False)
    sections.append('## Behavior: rule-performance delta vs baseline\n')
    sections.append(make_markdown_table(
        behavior_table.to_dict('records'),
        ['Feedback route'] + [CONDITION_LABELS[c] for c in BEHAVIOR_CONDITIONS],
    ))

    activity_rows = []
    for variant in ACTIVITY_VARIANTS:
        row = {'Feedback route': VARIANT_LABELS[variant]}
        for population in ['sr_vip', 'sr_sst', 'sr_edend', 'pfc_sst', 'pfc_edend']:
            vals = activity_summary[
                (activity_summary['variant'] == variant) &
                (activity_summary['window'] == 'early_post_feedback') &
                (activity_summary['population'] == population)
            ]
            if vals.empty:
                row[POP_LABELS[population]] = ''
                continue
            r = vals.iloc[0]
            row[POP_LABELS[population]] = format_mean_sem(r['mean'], r['sem'])
        activity_rows.append(row)
    activity_table = pd.DataFrame(activity_rows)
    activity_table.to_csv(output_dir / 'manuscript_activity_table.csv', index=False)
    sections.append('\n## Activity: error-minus-correct activity, early post-feedback\n')
    sections.append(make_markdown_table(
        activity_table.to_dict('records'),
        ['Feedback route'] + [POP_LABELS[p] for p in ['sr_vip', 'sr_sst', 'sr_edend', 'pfc_sst', 'pfc_edend']],
    ))

    causal_rows = []
    for variant in ['sr_vip_error', 'both_vip_error']:
        for causal_test in ['sr_vip_silence_feedback', 'sr_sst_silence_feedback']:
            for window in CAUSAL_WINDOWS:
                row = {
                    'Feedback route': VARIANT_LABELS[variant],
                    'Silencing': CAUSAL_LABELS[causal_test],
                    'Window': WINDOW_LABELS[window],
                }
                for condition in CAUSAL_CONDITIONS:
                    vals = causal_summary[
                        (causal_summary['variant'] == variant) &
                        (causal_summary['causal_test'] == causal_test) &
                        (causal_summary['window_run'] == window) &
                        (causal_summary['condition'] == condition)
                    ]
                    if vals.empty:
                        row[CONDITION_LABELS[condition]] = ''
                        continue
                    r = vals.iloc[0]
                    row[CONDITION_LABELS[condition]] = format_mean_sem(r['mean'], r['sem'])
                causal_rows.append(row)
    causal_table = pd.DataFrame(causal_rows)
    causal_table.to_csv(output_dir / 'manuscript_causal_table.csv', index=False)
    sections.append('\n## Causal perturbation: rule-performance delta, silenced minus intact\n')
    sections.append(make_markdown_table(
        causal_table.to_dict('records'),
        ['Feedback route', 'Silencing', 'Window'] + [CONDITION_LABELS[c] for c in CAUSAL_CONDITIONS],
    ))

    sections.append(
        '\nNotes:\n'
        '- Values are seed-level mean +/- SEM.\n'
        '- Behavior values are route minus baseline; positive means the route improved rule performance.\n'
        '- Causal values are silenced minus intact; negative means silencing hurt rule performance.\n'
        '- Dendritic causal silencing is intentionally omitted because the current hook returned zero effects and should be validated before interpretation.\n'
    )

    manuscript_md = output_dir / 'manuscript_core_results_table.md'
    with open(manuscript_md, 'w') as handle:
        handle.write('\n'.join(sections))
    return behavior_table, activity_table, causal_table


def plot_bar_with_seed_points(ax, seed_df, summary_df, x_order, hue_order, x_col, hue_col, value_col, colors, y_label):
    width = 0.22
    offsets = np.linspace(-width, width, len(hue_order))
    x_positions = np.arange(len(x_order))
    for hidx, hue in enumerate(hue_order):
        means = []
        sems = []
        for x in x_order:
            row = summary_df[(summary_df[x_col] == x) & (summary_df[hue_col] == hue)]
            means.append(float(row['mean'].iloc[0]) if not row.empty else np.nan)
            sems.append(float(row['sem'].iloc[0]) if not row.empty else np.nan)
        xpos = x_positions + offsets[hidx]
        ax.bar(
            xpos,
            means,
            width=width * 0.9,
            color=colors[hidx],
            alpha=0.82,
            edgecolor='black',
            linewidth=0.5,
            label=VARIANT_LABELS.get(hue, hue),
        )
        ax.errorbar(xpos, means, yerr=sems, fmt='none', ecolor='black', linewidth=0.8, capsize=2)
        for xidx, x in enumerate(x_order):
            vals = seed_df[(seed_df[x_col] == x) & (seed_df[hue_col] == hue)][value_col].to_numpy()
            if vals.size == 0:
                continue
            jitter = np.linspace(-width * 0.22, width * 0.22, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(np.full(vals.size, xpos[xidx]) + jitter, vals, s=18, color='white', edgecolor='black', linewidth=0.5, zorder=3)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([CONDITION_LABELS.get(x, x) for x in x_order], rotation=25, ha='right')
    ax.tick_params(axis='both', labelsize=13)
    ax.set_ylabel(y_label, fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)


def plot_heatmap(ax, matrix, row_labels, col_labels, title, cmap='coolwarm', vlim=None):
    values = np.asarray(matrix, dtype=float)
    if vlim is None:
        finite = values[np.isfinite(values)]
        vlim = max(abs(finite.min()), abs(finite.max())) if finite.size else 1.0
    im = ax.imshow(values, cmap=cmap, vmin=-vlim, vmax=vlim, aspect='auto')
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_title(title, loc='left', pad=12, fontsize=15)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', fontsize=12)
    return im


def add_panel_label(ax, label, x=-0.16, y=1.045):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=16,
        fontweight='bold',
        clip_on=False,
    )


def causal_matrix(causal_summary, causal_test):
    rows = []
    row_labels = []
    for variant in ['sr_vip_error', 'both_vip_error']:
        for window in CAUSAL_WINDOWS:
            row = []
            for condition in CAUSAL_CONDITIONS:
                vals = causal_summary[
                    (causal_summary['variant'] == variant) &
                    (causal_summary['causal_test'] == causal_test) &
                    (causal_summary['window_run'] == window) &
                    (causal_summary['condition'] == condition)
                ]
                row.append(float(vals['mean'].iloc[0]) if not vals.empty else np.nan)
            rows.append(row)
            row_labels.append('{} - {}'.format(VARIANT_LABELS[variant], WINDOW_LABELS[window]))
    return rows, row_labels


def draw_text(draw, xy, text, font, fill=(20, 20, 20), anchor=None):
    kwargs = {'fill': fill, 'font': font}
    if anchor is not None:
        kwargs['anchor'] = anchor
    draw.text(xy, str(text), **kwargs)


def get_font(size, bold=False):
    from PIL import ImageFont

    candidates = [
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        '/Library/Fonts/Arial.ttf',
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def interp_color(value, vlim):
    value = max(-vlim, min(vlim, float(value)))
    if value >= 0:
        t = value / vlim if vlim else 0
        lo = np.array([247, 247, 247])
        hi = np.array([180, 54, 60])
    else:
        t = -value / vlim if vlim else 0
        lo = np.array([247, 247, 247])
        hi = np.array([67, 113, 168])
    rgb = (lo * (1 - t) + hi * t).astype(int)
    return tuple(int(x) for x in rgb)


def write_embedded_svg(png_path, svg_path, width, height):
    import base64

    encoded = base64.b64encode(Path(png_path).read_bytes()).decode('ascii')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>\n'
        '</svg>\n'
    )
    Path(svg_path).write_text(svg)


def make_figure_pil(behavior_seed, behavior_summary, activity_summary, causal_summary, output_dir):
    from PIL import Image, ImageDraw

    width, height = 2400, 1700
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    font_title = get_font(40, bold=True)
    font_panel = get_font(30, bold=True)
    font_label = get_font(26)
    font_small = get_font(22)
    font_tiny = get_font(20)

    colors = {
        'pfc_vip_error': (135, 135, 135),
        'sr_vip_error': (76, 120, 168),
        'both_vip_error': (196, 78, 82),
    }

    # Panel A: behavior bars with seed points.
    ax = (100, 145, 1050, 640)
    x0, y0, x1, y1 = ax
    draw_text(draw, (x0, y0 - 45), 'a.', font_panel)
    y_min, y_max = -0.23, 0.23
    plot_left, plot_top, plot_right, plot_bottom = x0 + 100, y0 + 40, x1 - 30, y1 - 100

    def map_y(v):
        return plot_bottom - (float(v) - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(40, 40, 40), width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(40, 40, 40), width=2)
    draw.line((plot_left, map_y(0), plot_right, map_y(0)), fill=(80, 80, 80), width=2)
    for tick in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        y = map_y(tick)
        draw.line((plot_left - 6, y, plot_left, y), fill=(40, 40, 40), width=2)
        draw_text(draw, (plot_left - 12, y), f'{tick:.1f}', font_tiny, anchor='rm')
    draw_text(draw, (plot_left, y0 - 45), 'Route-dependent rule updating', font_panel)
    draw_text(draw, (plot_left, plot_top - 28), 'Delta rule perf. vs baseline', font_small)

    group_centers = np.linspace(plot_left + 130, plot_right - 130, len(BEHAVIOR_CONDITIONS))
    bar_w = 42
    variant_offsets = [-52, 0, 52]
    for cidx, condition in enumerate(BEHAVIOR_CONDITIONS):
        cx = group_centers[cidx]
        draw_text(draw, (cx, plot_bottom + 28), CONDITION_LABELS[condition], font_small, anchor='ma')
        for vidx, variant in enumerate(MANUSCRIPT_VARIANTS):
            vals = behavior_summary[(behavior_summary['condition'] == condition) & (behavior_summary['variant'] == variant)]
            if vals.empty:
                continue
            mean = float(vals['mean'].iloc[0])
            sem_value = float(vals['sem'].iloc[0])
            bx = cx + variant_offsets[vidx]
            y_mean = map_y(mean)
            y_zero = map_y(0)
            draw.rectangle((bx - bar_w / 2, min(y_mean, y_zero), bx + bar_w / 2, max(y_mean, y_zero)), fill=colors[variant], outline=(20, 20, 20))
            draw.line((bx, map_y(mean - sem_value), bx, map_y(mean + sem_value)), fill=(20, 20, 20), width=2)
            draw.line((bx - 8, map_y(mean - sem_value), bx + 8, map_y(mean - sem_value)), fill=(20, 20, 20), width=2)
            draw.line((bx - 8, map_y(mean + sem_value), bx + 8, map_y(mean + sem_value)), fill=(20, 20, 20), width=2)
            seed_vals = behavior_seed[(behavior_seed['condition'] == condition) & (behavior_seed['variant'] == variant)]['delta_vs_baseline'].to_numpy()
            for sidx, value in enumerate(seed_vals):
                jitter = (sidx - (len(seed_vals) - 1) / 2) * 7
                draw.ellipse((bx + jitter - 5, map_y(value) - 5, bx + jitter + 5, map_y(value) + 5), fill='white', outline=(20, 20, 20), width=1)
    legend_x = plot_left + 10
    legend_y = y1 - 16
    for idx, variant in enumerate(MANUSCRIPT_VARIANTS):
        lx = legend_x + idx * 210
        draw.rectangle((lx, legend_y, lx + 26, legend_y + 18), fill=colors[variant], outline=(20, 20, 20))
        draw_text(draw, (lx + 34, legend_y - 1), VARIANT_LABELS[variant], font_tiny)

    # Panel B: activity heatmap.
    ax = (1200, 145, 2300, 640)
    x0, y0, x1, y1 = ax
    heat_pops = ['sr_vip', 'sr_sst', 'sr_edend', 'pfc_sst', 'pfc_edend']
    cell_w, cell_h = 140, 72
    hx, hy = x0 + 250, y0 + 80
    draw_text(draw, (x0, y0 - 45), 'b.', font_panel)
    draw_text(draw, (hx, y0 - 45), 'Activity following true errors versus correct choices', font_panel)
    for j, pop in enumerate(heat_pops):
        draw_text(draw, (hx + j * cell_w + cell_w / 2, hy - 18), POP_LABELS[pop], font_tiny, anchor='mm')
    for i, variant in enumerate(ACTIVITY_VARIANTS):
        draw_text(draw, (hx - 20, hy + i * cell_h + cell_h / 2), VARIANT_LABELS[variant], font_small, anchor='rm')
        for j, pop in enumerate(heat_pops):
            vals = activity_summary[
                (activity_summary['variant'] == variant) &
                (activity_summary['window'] == 'early_post_feedback') &
                (activity_summary['population'] == pop)
            ]
            value = float(vals['mean'].iloc[0]) if not vals.empty else np.nan
            x = hx + j * cell_w
            y = hy + i * cell_h
            fill = (235, 235, 235) if np.isnan(value) else interp_color(value, 0.12)
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=fill, outline='white', width=3)
            text_fill = (20, 20, 20)
            draw_text(draw, (x + cell_w / 2, y + cell_h / 2), f'{value:.2f}', font_small, fill=text_fill, anchor='mm')
    draw_text(
        draw,
        (hx, hy + len(ACTIVITY_VARIANTS) * cell_h + 34),
        'After true error - after correct choice',
        font_tiny,
    )

    def draw_causal_heatmap(panel, causal_test, label, title):
        x0, y0, x1, y1 = panel
        mat, rows = causal_matrix(causal_summary, causal_test)
        cell_w, cell_h = 165, 70
        hx, hy = x0 + 360, y0 + 70
        draw_text(draw, (x0, y0 - 45), label, font_panel)
        draw_text(draw, (hx, y0 - 45), title, font_panel)
        for j, condition in enumerate(CAUSAL_CONDITIONS):
            draw_text(draw, (hx + j * cell_w + cell_w / 2, hy - 22), CONDITION_LABELS[condition], font_tiny, anchor='mm')
        for i, row_label in enumerate(rows):
            draw_text(draw, (hx - 20, hy + i * cell_h + cell_h / 2), row_label, font_tiny, anchor='rm')
            for j, value in enumerate(mat[i]):
                x = hx + j * cell_w
                y = hy + i * cell_h
                fill = (235, 235, 235) if np.isnan(value) else interp_color(value, 0.24)
                draw.rectangle((x, y, x + cell_w, y + cell_h), fill=fill, outline='white', width=3)
                draw_text(draw, (x + cell_w / 2, y + cell_h / 2), f'{value:.2f}', font_small, anchor='mm')
        draw_text(draw, (hx, hy + len(rows) * cell_h + 34), 'Silenced - intact rule performance', font_tiny)

    draw_causal_heatmap((100, 745, 1110, 1500), 'sr_vip_silence_feedback', 'c.', 'SR VIP silencing')
    draw_causal_heatmap((1160, 745, 2170, 1500), 'sr_sst_silence_feedback', 'd.', 'SR SST silencing')

    png_path = output_dir / 'stochastic_bottleneck_core_results.png'
    pdf_path = output_dir / 'stochastic_bottleneck_core_results.pdf'
    svg_path = output_dir / 'stochastic_bottleneck_core_results.svg'
    image.save(png_path)
    image.save(pdf_path, 'PDF', resolution=300.0)
    write_embedded_svg(png_path, svg_path, width, height)


def make_figure(behavior_seed, behavior_summary, activity_summary, causal_seed, causal_summary, output_dir):
    try:
        import matplotlib
    except ModuleNotFoundError:
        make_figure_pil(behavior_seed, behavior_summary, activity_summary, causal_summary, output_dir)
        return

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 11,
        'figure.dpi': 140,
        'savefig.dpi': 300,
    })

    fig = plt.figure(figsize=(15.4, 10.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], width_ratios=[1.2, 1.0], hspace=0.42, wspace=0.74)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(top=0.96)

    plot_bar_with_seed_points(
        ax_a,
        behavior_seed,
        behavior_summary,
        x_order=['previous_error', 'stable_trials', 'true_switch_trial'],
        hue_order=MANUSCRIPT_VARIANTS,
        x_col='condition',
        hue_col='variant',
        value_col='delta_vs_baseline',
        colors=['#8b8b8b', '#4c78a8', '#c44e52'],
        y_label='Delta rule perf. vs baseline',
    )
    ax_a.set_title('Route-dependent rule updating', loc='left', pad=12, fontsize=15)
    add_panel_label(ax_a, 'a.', x=-0.16)
    ax_a.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, borderaxespad=0.0)

    heat_variants = ACTIVITY_VARIANTS
    heat_pops = ['sr_vip', 'sr_sst', 'sr_edend', 'pfc_sst', 'pfc_edend']
    mat = []
    for variant in heat_variants:
        row = []
        for pop in heat_pops:
            vals = activity_summary[
                (activity_summary['variant'] == variant) &
                (activity_summary['window'] == 'early_post_feedback') &
                (activity_summary['population'] == pop)
            ]
            row.append(float(vals['mean'].iloc[0]) if not vals.empty else np.nan)
        mat.append(row)
    im = plot_heatmap(
        ax_b,
        mat,
        row_labels=[VARIANT_LABELS[v] for v in heat_variants],
        col_labels=[POP_LABELS[p] for p in heat_pops],
        title='Activity following true errors\nversus correct choices',
        vlim=0.12,
    )
    add_panel_label(ax_b, 'b.', x=-0.16)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.018)
    cbar.set_label('After true error - after correct choice', labelpad=6, fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    causal_vlim = 0.24
    sr_vip_mat, sr_vip_rows = causal_matrix(causal_summary, 'sr_vip_silence_feedback')
    im_c = plot_heatmap(
        ax_c,
        sr_vip_mat,
        row_labels=sr_vip_rows,
        col_labels=[CONDITION_LABELS[c] for c in CAUSAL_CONDITIONS],
        title='Causal effect of SR VIP silencing',
        vlim=causal_vlim,
    )
    add_panel_label(ax_c, 'c.', x=-0.20)
    sr_sst_mat, sr_sst_rows = causal_matrix(causal_summary, 'sr_sst_silence_feedback')
    im_d = plot_heatmap(
        ax_d,
        sr_sst_mat,
        row_labels=sr_sst_rows,
        col_labels=[CONDITION_LABELS[c] for c in CAUSAL_CONDITIONS],
        title='Causal effect of SR SST silencing',
        vlim=causal_vlim,
    )
    add_panel_label(ax_d, 'd.', x=-0.20)
    cbar = fig.colorbar(im_d, ax=[ax_c, ax_d], fraction=0.032, pad=0.015)
    cbar.set_label('Silenced - intact rule performance', labelpad=6, fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'stochastic_bottleneck_core_results.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    global np, pd
    import numpy as np
    import pandas as pd

    summaries_root = Path(args.summaries_root).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve() if args.output_dir else summaries_root / 'paper_tables_figures')

    behavior_seed, behavior_summary = read_behavior_summary(summaries_root, args.behavior_dir, output_dir)
    _, activity_summary = read_activity_summary(summaries_root, args.behavior_dir, output_dir)
    causal_seed, causal_summary = read_causal_summary(summaries_root, args.causal_dir, output_dir)

    write_compact_table(behavior_summary, activity_summary, causal_summary, output_dir)
    write_manuscript_tables(behavior_summary, activity_summary, causal_summary, output_dir)
    make_figure(behavior_seed, behavior_summary, activity_summary, causal_seed, causal_summary, output_dir)

    print('wrote results to {}'.format(output_dir))
    print('main table: {}'.format(output_dir / 'core_results_table.md'))
    print('main figure: {}'.format(output_dir / 'stochastic_bottleneck_core_results.png'))


if __name__ == '__main__':
    main()
