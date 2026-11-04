#!/usr/bin/env python3
"""Summarize causal silencing deltas from analyze_feedback_mechanism.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional on small local installs
    stats = None


CONDITIONS = ['all_nonfirst', 'stable_trials', 'previous_error', 'true_switch_trial']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--analysis-dir',
        required=True,
        help='Directory containing causal_behavior_deltas.csv.',
    )
    parser.add_argument(
        '--conditions',
        nargs='*',
        default=CONDITIONS,
        help='Behavior conditions to include in the summary.',
    )
    return parser.parse_args()


def grouped_summary(df, group_cols):
    summary = df.groupby(group_cols)['delta_vs_intact'].agg(['mean', 'std', 'count'])
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['n_negative'] = df.groupby(group_cols)['delta_vs_intact'].apply(lambda x: int((x < 0).sum()))
    if stats is not None:
        summary['ttest_p'] = df.groupby(group_cols)['delta_vs_intact'].apply(
            lambda x: float(stats.ttest_1samp(x.dropna(), 0.0).pvalue) if len(x.dropna()) > 1 else np.nan
        )
        summary['wilcoxon_p'] = df.groupby(group_cols)['delta_vs_intact'].apply(
            lambda x: float(stats.wilcoxon(x.dropna()).pvalue)
            if len(x.dropna()) > 0 and np.any(np.asarray(x.dropna()) != 0)
            else np.nan
        )
    return summary


def print_and_write(title, summary, path):
    print('\n' + title)
    if summary.empty:
        print('(no rows)')
        return
    summary.round(4).to_csv(path)
    print(summary.round(4))
    print('wrote {}'.format(path))


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    path = analysis_dir / 'causal_behavior_deltas.csv'
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        print('No causal rows found in {}'.format(path))
        return

    df = df[df['condition'].isin(args.conditions)]
    print('Read {} causal-delta rows from {}'.format(len(df), path))
    print('Negative values mean silencing hurt performance relative to the intact model.')

    overall = grouped_summary(df, ['variant', 'causal_test', 'metric', 'condition'])
    split_rule = grouped_summary(df, ['variant', 'causal_test', 'metric', 'init_rule', 'condition'])

    print_and_write(
        'Overall causal deltas',
        overall,
        analysis_dir / 'causal_delta_summary.csv',
    )
    print_and_write(
        'Causal deltas split by initial rule',
        split_rule,
        analysis_dir / 'causal_delta_by_init_rule.csv',
    )

    focused = df[
        (df['metric'] == 'mean_rule_perf') &
        (df['condition'].isin(['all_nonfirst', 'previous_error', 'stable_trials', 'true_switch_trial']))
    ]
    focused_summary = grouped_summary(focused, ['variant', 'causal_test', 'condition'])
    print_and_write(
        'Focused rule-performance causal deltas',
        focused_summary,
        analysis_dir / 'focused_rule_causal_delta_summary.csv',
    )


if __name__ == '__main__':
    main()
