#!/usr/bin/env python3
"""Summarize frozen feedback-route ablation results."""

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
        help='Directory containing weight_ablation_behavior_deltas.csv.',
    )
    parser.add_argument(
        '--conditions',
        nargs='*',
        default=CONDITIONS,
        help='Behavior conditions to include in the summary.',
    )
    return parser.parse_args()


def print_grouped(title, df, group_cols):
    print('\n' + title)
    if df.empty:
        print('(no rows)')
        return
    summary = df.groupby(group_cols)['delta_vs_intact'].agg(['mean', 'std', 'count'])
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['n_negative'] = df.groupby(group_cols)['delta_vs_intact'].apply(lambda x: int((x < 0).sum()))
    if stats is not None:
        summary['ttest_p'] = df.groupby(group_cols)['delta_vs_intact'].apply(
            lambda x: float(stats.ttest_1samp(x, 0.0, nan_policy='omit').pvalue) if len(x.dropna()) > 1 else np.nan
        )
        summary['wilcoxon_p'] = df.groupby(group_cols)['delta_vs_intact'].apply(
            lambda x: float(stats.wilcoxon(x.dropna()).pvalue)
            if len(x.dropna()) > 0 and np.any(np.asarray(x.dropna()) != 0)
            else np.nan
        )
    summary = summary.round(4)
    print(summary)


def main():
    args = parse_args()
    path = Path(args.analysis_dir) / 'weight_ablation_behavior_deltas.csv'
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        print('No ablation rows found in {}'.format(path))
        return

    df = df[df['condition'].isin(args.conditions)]
    print('Read {} ablation-delta rows from {}'.format(len(df), path))
    print('Negative values mean the ablation hurt performance relative to the intact model.')

    print_grouped(
        'Overall ablation deltas',
        df,
        ['variant', 'weight_ablation', 'metric', 'condition'],
    )
    print_grouped(
        'Deltas split by initial rule',
        df,
        ['variant', 'weight_ablation', 'metric', 'init_rule', 'condition'],
    )


if __name__ == '__main__':
    main()
