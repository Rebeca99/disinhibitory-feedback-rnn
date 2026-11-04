#!/usr/bin/env python3
"""Summarize causal silencing effects using trained seed as the unit of replication."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - scipy is optional for quick environments
    stats = None


DEFAULT_CONDITIONS = ['all_nonfirst', 'stable_trials', 'previous_error', 'true_switch_trial']
DEFAULT_FOCUSED_TESTS = [
    'sr_vip_silence_feedback',
    'sr_sst_silence_feedback',
    'pfc_vip_silence_feedback',
    'pfc_sst_silence_feedback',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--analysis-dirs',
        nargs='+',
        required=True,
        help='One or more analysis directories containing causal_behavior_deltas.csv.',
    )
    parser.add_argument(
        '--labels',
        nargs='*',
        default=None,
        help='Optional labels for the analysis dirs, e.g. feedback early_post combined.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Where to write seed-level summary CSVs. Defaults to <parent>/seed_level_causal_summary.',
    )
    parser.add_argument(
        '--conditions',
        nargs='*',
        default=DEFAULT_CONDITIONS,
        help='Behavior conditions to include.',
    )
    parser.add_argument(
        '--focused-tests',
        nargs='*',
        default=DEFAULT_FOCUSED_TESTS,
        help='Causal tests to include in the focused printed summary.',
    )
    return parser.parse_args()


def infer_label(path):
    name = path.name
    if 'feedback_only' in name:
        return 'feedback_only'
    if 'early_post_only' in name:
        return 'early_post_only'
    if 'feedback_plus_early' in name:
        return 'feedback_plus_early'
    label = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    return label or 'analysis'


def read_inputs(analysis_dirs, labels):
    frames = []
    for path, label in zip(analysis_dirs, labels):
        csv_path = path / 'causal_behavior_deltas.csv'
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        if df.empty:
            print('skipping empty {}'.format(csv_path))
            continue
        df['window_run'] = label
        df['analysis_dir'] = str(path)
        frames.append(df)
    if not frames:
        raise RuntimeError('No non-empty causal_behavior_deltas.csv files were found.')
    return pd.concat(frames, ignore_index=True)


def add_stats(summary, grouped):
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['n_negative'] = grouped.apply(lambda x: int((x < 0).sum()))
    summary['n_positive'] = grouped.apply(lambda x: int((x > 0).sum()))
    summary['median'] = grouped.median()
    if stats is not None:
        summary['ttest_p'] = grouped.apply(
            lambda x: float(stats.ttest_1samp(x.dropna(), 0.0).pvalue) if len(x.dropna()) > 1 else np.nan
        )
        summary['wilcoxon_p'] = grouped.apply(
            lambda x: float(stats.wilcoxon(x.dropna()).pvalue)
            if len(x.dropna()) > 0 and np.any(np.asarray(x.dropna()) != 0)
            else np.nan
        )
    return summary


def summarize(df, group_cols):
    grouped = df.groupby(group_cols)['delta_vs_intact']
    summary = grouped.agg(['mean', 'std', 'count'])
    return add_stats(summary, grouped)


def write_summary(title, summary, path, max_rows=120):
    print('\n' + title)
    if summary.empty:
        print('(no rows)')
        return
    summary.round(4).to_csv(path)
    with pd.option_context('display.max_rows', max_rows, 'display.width', 220):
        print(summary.round(4))
    print('wrote {}'.format(path))


def main():
    args = parse_args()
    analysis_dirs = [Path(path).expanduser().resolve() for path in args.analysis_dirs]
    if args.labels is None or len(args.labels) == 0:
        labels = [infer_label(path) for path in analysis_dirs]
    else:
        if len(args.labels) != len(analysis_dirs):
            raise ValueError('--labels must have the same length as --analysis-dirs')
        labels = args.labels

    if args.output_dir is None:
        output_dir = analysis_dirs[0].parent / 'seed_level_causal_summary'
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_inputs(analysis_dirs, labels)
    df = df[df['condition'].isin(args.conditions)].copy()

    seed_keys = ['window_run', 'variant', 'causal_test', 'metric', 'condition', 'seed']
    seed_level = (
        df.groupby(seed_keys, as_index=False)['delta_vs_intact']
        .mean()
        .sort_values(seed_keys)
    )
    seed_level.to_csv(output_dir / 'seed_level_causal_deltas.csv', index=False)
    print('wrote {}'.format(output_dir / 'seed_level_causal_deltas.csv'))

    summary = summarize(
        seed_level,
        ['window_run', 'variant', 'causal_test', 'metric', 'condition'],
    )
    write_summary(
        'Seed-level causal deltas',
        summary,
        output_dir / 'seed_level_causal_delta_summary.csv',
    )

    focused = seed_level[
        (seed_level['metric'] == 'mean_rule_perf') &
        (seed_level['causal_test'].isin(args.focused_tests))
    ].copy()
    focused_summary = summarize(
        focused,
        ['window_run', 'variant', 'causal_test', 'condition'],
    )
    write_summary(
        'Focused seed-level rule deltas',
        focused_summary,
        output_dir / 'focused_seed_level_rule_delta_summary.csv',
    )

    init_keys = ['window_run', 'variant', 'causal_test', 'metric', 'init_rule', 'condition', 'seed']
    if 'init_rule' in df.columns:
        seed_by_rule = (
            df.groupby(init_keys, as_index=False)['delta_vs_intact']
            .mean()
            .sort_values(init_keys)
        )
        seed_by_rule.to_csv(output_dir / 'seed_level_causal_deltas_by_init_rule.csv', index=False)
        split_summary = summarize(
            seed_by_rule[
                (seed_by_rule['metric'] == 'mean_rule_perf') &
                (seed_by_rule['causal_test'].isin(args.focused_tests))
            ],
            ['window_run', 'variant', 'causal_test', 'init_rule', 'condition'],
        )
        write_summary(
            'Focused seed-level rule deltas split by initial rule',
            split_summary,
            output_dir / 'focused_seed_level_rule_delta_by_init_rule.csv',
        )

    pivot = focused.pivot_table(
        index=['variant', 'causal_test', 'condition', 'seed'],
        columns='window_run',
        values='delta_vs_intact',
        aggfunc='mean',
    ).reset_index()
    pivot.to_csv(output_dir / 'seed_level_window_pivot.csv', index=False)
    print('wrote {}'.format(output_dir / 'seed_level_window_pivot.csv'))


if __name__ == '__main__':
    main()
