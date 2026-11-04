#!/usr/bin/env python3
"""Summarize PFC-feedback-scale diagnostic training sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    'mean_last5_test_perf',
    'mean_last5_test_rule_perf',
    'mean_last5_train_perf',
    'mean_last5_train_rule_perf',
]
HISTORY_COLS = [
    'give_prev_stim',
    'give_prev_choice',
    'give_prev_rew',
    'prev_stim_mag',
    'prev_choice_mag',
    'prev_rew_mag',
]
SWITCH_COLS = [
    'block_len',
    'train_switch_mode',
    'test_switch_mode',
    'stochastic_min_block_len',
    'stochastic_max_block_len',
    'stochastic_switch_prob',
    'test_stochastic_min_block_len',
    'test_stochastic_max_block_len',
    'test_stochastic_switch_prob',
]
FREEZE_COLS = [
    'freeze_backbone_mode',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, help='Root containing pfc_* output directories.')
    parser.add_argument('--summary-dir', default=None, help='Directory for sweep-level summary CSVs.')
    return parser.parse_args()


def read_runs(output_root):
    frames = []
    for path in sorted(Path(output_root).glob('pfc_*/summaries/training_runs.csv')):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df['scale_output_dir'] = str(path.parents[1])
        for col in HISTORY_COLS:
            if col not in df.columns:
                if col.startswith('give_'):
                    df[col] = True
                else:
                    df[col] = 1.0
        for col in SWITCH_COLS:
            if col not in df.columns:
                if col.endswith('_mode'):
                    df[col] = 'fixed_interval' if col == 'test_switch_mode' else 'fixed_n'
                else:
                    df[col] = ''
        for col in FREEZE_COLS:
            if col not in df.columns:
                df[col] = 'none'
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def numeric_summary(df):
    rows = []
    group_cols = HISTORY_COLS + SWITCH_COLS + FREEZE_COLS + ['prev_rew_to_pfc_scale', 'variant', 'status']
    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row['n_runs'] = len(group)
        for metric in METRICS:
            vals = pd.to_numeric(group[metric], errors='coerce').dropna()
            row[metric] = vals.mean() if len(vals) else ''
            row[metric + '_std'] = vals.std() if len(vals) > 1 else ''
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(df):
    rows = []
    index_cols = HISTORY_COLS + SWITCH_COLS + FREEZE_COLS + ['prev_rew_to_pfc_scale', 'seed']
    for metric in METRICS:
        wide = df.pivot_table(index=index_cols, columns='variant', values=metric, aggfunc='mean')
        if 'baseline' not in wide.columns:
            continue
        for variant in wide.columns:
            if variant == 'baseline':
                continue
            delta = wide[variant] - wide['baseline']
            for idx, value in delta.dropna().items():
                row = dict(zip(index_cols, idx))
                row.update({'metric': metric, 'variant': variant, 'delta_vs_baseline': float(value)})
                rows.append(row)
    return pd.DataFrame(rows)


def paired_delta_summary(deltas):
    if deltas.empty:
        return pd.DataFrame()
    summary = (
        deltas.groupby(HISTORY_COLS + FREEZE_COLS + ['prev_rew_to_pfc_scale', 'variant', 'metric'])['delta_vs_baseline']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    summary['sem'] = summary['std'] / summary['count'].pow(0.5)
    return summary


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    summary_dir = Path(args.summary_dir).resolve() if args.summary_dir else output_root / 'summaries'
    summary_dir.mkdir(parents=True, exist_ok=True)

    runs = read_runs(output_root)
    if runs.empty:
        raise RuntimeError('No training_runs.csv files found under {}'.format(output_root))

    by_variant = numeric_summary(runs)
    deltas = paired_deltas(runs)
    delta_summary = paired_delta_summary(deltas)

    runs.to_csv(summary_dir / 'pfc_scale_training_runs.csv', index=False)
    by_variant.to_csv(summary_dir / 'pfc_scale_by_variant.csv', index=False)
    deltas.to_csv(summary_dir / 'pfc_scale_paired_deltas_vs_baseline.csv', index=False)
    delta_summary.to_csv(summary_dir / 'pfc_scale_paired_delta_summary.csv', index=False)

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 180)
    print('\nTraining summary by PFC scale and variant')
    print(by_variant.round(4))
    print('\nPaired deltas versus baseline')
    print(delta_summary.round(4))
    print('\nWrote PFC-scale diagnostic summaries to {}'.format(summary_dir))


if __name__ == '__main__':
    main()
