#!/usr/bin/env python3
"""Print compact summaries from analyze_rule_decoding.py outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', required=True)
    parser.add_argument('--condition', default='all_nonfirst')
    parser.add_argument('--feature-sets', nargs='*', default=['pfc', 'sr', 'pfc_sr', 'pfc_motif', 'sr_motif'])
    parser.add_argument('--windows', nargs='*', default=['feedback', 'early_post_feedback', 'pre_stim_delay', 'response_epoch'])
    return parser.parse_args()


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    summary_path = analysis_dir / 'rule_decoding_summary.csv'
    delta_path = analysis_dir / 'rule_decoding_delta_summary.csv'

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    summary = summary[
        (summary['condition'] == args.condition) &
        (summary['feature_set'].isin(args.feature_sets)) &
        (summary['window'].isin(args.windows))
    ].copy()

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print('\n=== Rule-decoding balanced accuracy ({}) ==='.format(args.condition))
    if summary.empty:
        print('No rows found.')
    else:
        table = summary.pivot_table(
            index=['feature_set', 'window'],
            columns='variant',
            values='mean',
            aggfunc='mean',
        )
        print(table)

    if not delta_path.exists() or delta_path.stat().st_size == 0:
        return
    deltas = pd.read_csv(delta_path)
    deltas = deltas[
        (deltas['condition'] == args.condition) &
        (deltas['feature_set'].isin(args.feature_sets)) &
        (deltas['window'].isin(args.windows))
    ].copy()
    print('\n=== Rule-decoding deltas vs baseline ({}) ==='.format(args.condition))
    if deltas.empty:
        print('No delta rows found.')
    else:
        table = deltas.pivot_table(
            index=['feature_set', 'window'],
            columns='variant',
            values='mean',
            aggfunc='mean',
        )
        print(table)

    print('\nFiles:')
    for name in [
        'rule_decoding_scores.csv',
        'rule_decoding_summary.csv',
        'rule_decoding_deltas_vs_baseline.csv',
        'rule_decoding_delta_summary.csv',
        'rule_decoding_summary.png',
        'rule_decoding_summary.pdf',
    ]:
        path = analysis_dir / name
        if path.exists():
            print(path)


if __name__ == '__main__':
    main()
