#!/usr/bin/env python3
"""Pool stochastic hidden-switch robustness results across pretrained checkpoints."""

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
FREEZE_COLS = ['freeze_backbone_mode']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, help='Root containing ckpt_* subdirectories.')
    parser.add_argument('--summary-dir', default=None, help='Destination for pooled summary CSVs.')
    return parser.parse_args()


def read_label(checkpoint_dir: Path) -> str:
    label_path = checkpoint_dir / 'pretrained_checkpoint_label.txt'
    if label_path.exists():
        label = label_path.read_text().strip()
        if label:
            return label
    name = checkpoint_dir.name
    if name.startswith('ckpt_'):
        parts = name.split('_', 2)
        if len(parts) == 3:
            return parts[2]
    return name


def read_checkpoint_path(checkpoint_dir: Path) -> str:
    path = checkpoint_dir / 'pretrained_checkpoint_path.txt'
    return path.read_text().strip() if path.exists() else ''


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in HISTORY_COLS:
        if col not in df.columns:
            df[col] = True if col.startswith('give_') else 1.0
    for col in SWITCH_COLS:
        if col not in df.columns:
            if col == 'train_switch_mode':
                df[col] = 'fixed_n'
            elif col == 'test_switch_mode':
                df[col] = 'fixed_interval'
            else:
                df[col] = ''
    for col in FREEZE_COLS:
        if col not in df.columns:
            df[col] = 'none'
    if 'prev_rew_to_pfc_scale' not in df.columns:
        df['prev_rew_to_pfc_scale'] = 1.0
    return df


def read_runs(output_root: Path) -> pd.DataFrame:
    frames = []
    for checkpoint_dir in sorted(output_root.glob('ckpt_*')):
        if not checkpoint_dir.is_dir():
            continue
        pfc_summary = checkpoint_dir / 'summaries' / 'pfc_scale_training_runs.csv'
        if pfc_summary.exists():
            df = pd.read_csv(pfc_summary)
        else:
            parts = []
            for path in sorted(checkpoint_dir.glob('pfc_*/summaries/training_runs.csv')):
                parts.append(pd.read_csv(path))
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)
        if df.empty:
            continue
        df = ensure_columns(df)
        df['pretrained_label'] = read_label(checkpoint_dir)
        df['pretrained_path'] = read_checkpoint_path(checkpoint_dir)
        df['checkpoint_output_dir'] = str(checkpoint_dir)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def numeric_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = (
        ['pretrained_label']
        + HISTORY_COLS
        + SWITCH_COLS
        + FREEZE_COLS
        + ['prev_rew_to_pfc_scale', 'variant', 'status']
    )
    for keys, group in runs.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row['n_runs'] = len(group)
        for metric in METRICS:
            vals = pd.to_numeric(group[metric], errors='coerce').dropna()
            row[metric] = vals.mean() if len(vals) else ''
            row[metric + '_std'] = vals.std() if len(vals) > 1 else ''
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index_cols = (
        ['pretrained_label', 'seed']
        + HISTORY_COLS
        + SWITCH_COLS
        + FREEZE_COLS
        + ['prev_rew_to_pfc_scale']
    )
    for metric in METRICS:
        wide = runs.pivot_table(index=index_cols, columns='variant', values=metric, aggfunc='mean')
        if 'baseline' not in wide.columns:
            continue
        for variant in wide.columns:
            if variant == 'baseline':
                continue
            delta = wide[variant] - wide['baseline']
            for idx, value in delta.dropna().items():
                row = dict(zip(index_cols, idx))
                row.update({'variant': variant, 'metric': metric, 'delta_vs_baseline': float(value)})
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_deltas(deltas: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    summary = (
        deltas.groupby(group_cols, dropna=False)['delta_vs_baseline']
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
        raise RuntimeError('No checkpoint robustness training summaries found under {}'.format(output_root))

    by_checkpoint_variant = numeric_summary(runs)
    deltas = paired_deltas(runs)

    setting_cols = HISTORY_COLS + SWITCH_COLS + FREEZE_COLS + ['prev_rew_to_pfc_scale']
    pooled_summary = summarize_deltas(deltas, setting_cols + ['variant', 'metric'])

    if deltas.empty:
        checkpoint_mean_deltas = pd.DataFrame()
        checkpoint_level_summary = pd.DataFrame()
    else:
        checkpoint_mean_deltas = (
            deltas.groupby(['pretrained_label'] + setting_cols + ['variant', 'metric'], dropna=False)['delta_vs_baseline']
            .mean()
            .reset_index()
        )
        checkpoint_level_summary = summarize_deltas(
            checkpoint_mean_deltas,
            setting_cols + ['variant', 'metric'],
        )

    runs.to_csv(summary_dir / 'pretrained_checkpoint_training_runs.csv', index=False)
    by_checkpoint_variant.to_csv(summary_dir / 'pretrained_checkpoint_by_variant.csv', index=False)
    deltas.to_csv(summary_dir / 'pretrained_checkpoint_paired_deltas_vs_baseline.csv', index=False)
    pooled_summary.to_csv(summary_dir / 'pretrained_checkpoint_pooled_delta_summary.csv', index=False)
    checkpoint_mean_deltas.to_csv(summary_dir / 'pretrained_checkpoint_mean_deltas_by_checkpoint.csv', index=False)
    checkpoint_level_summary.to_csv(summary_dir / 'pretrained_checkpoint_level_delta_summary.csv', index=False)

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 180)

    print('\nAbsolute performance by checkpoint and variant')
    print(by_checkpoint_variant.round(4))
    print('\nPooled paired deltas across checkpoint-seed pairs')
    print(pooled_summary.round(4))
    print('\nCheckpoint-level deltas; each pretrained checkpoint weighted equally')
    print(checkpoint_level_summary.round(4))
    print('\nWrote pretrained-checkpoint robustness summaries to {}'.format(summary_dir))


if __name__ == '__main__':
    main()
