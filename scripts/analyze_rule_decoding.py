#!/usr/bin/env python3
"""Small rule-decoding validation for stochastic hidden-switch BioRNN models.

This analysis reuses the same saved models and frozen stochastic test task used
by the feedback-mechanism analyses. It does not retrain models. For each saved
model, it reruns paired test sessions, averages activity within task windows,
and asks whether a linear decoder can recover the current hidden rule.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE_ROOT / '.mplconfig'))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import analyze_feedback_mechanism as mech  # noqa: E402


RULE_TO_INT = {'color': 0, 'shape': 1}
VARIANT_LABELS = {
    'baseline': 'Baseline',
    'pfc_vip_error': 'PFC-VIP',
    'sr_vip_error': 'SR-VIP',
    'both_vip_error': 'PFC+SR VIP',
}
WINDOW_LABELS = {
    'feedback': 'Feedback',
    'early_post_feedback': 'Early post',
    'pre_stim_delay': 'Pre-stim delay',
    'response_epoch': 'Response',
}
FEATURE_SET_LABELS = {
    'pfc': 'PFC',
    'sr': 'SR',
    'pfc_sr': 'PFC+SR',
    'pfc_motif': 'PFC VIP/SST/dend',
    'sr_motif': 'SR VIP/SST/dend',
}
CONDITION_LABELS = {
    'all_nonfirst': 'All non-first',
    'previous_error': 'Previous error',
    'stable_trials': 'Stable',
    'true_switch_trial': 'Switch trial',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='Training output root containing saved_models/.')
    parser.add_argument('--saved-models-dir', default=None, help='Override saved model directory.')
    parser.add_argument('--analysis-dir', default=None, help='Directory for decoder outputs.')
    parser.add_argument('--model-glob', default='*feedback_stochastic_hidden_switch*')
    parser.add_argument('--variants', nargs='*', default=['baseline', 'pfc_vip_error', 'sr_vip_error', 'both_vip_error'])
    parser.add_argument('--seeds', nargs='*', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--max-models', type=int, default=None)
    parser.add_argument('--n-trials', type=int, default=300)
    parser.add_argument('--test-repeats', type=int, default=3)
    parser.add_argument('--test-seed-base', type=int, default=14000)
    parser.add_argument('--switch-every', type=int, default=20)
    parser.add_argument('--init-rules', nargs='*', default=['color', 'shape'], choices=['color', 'shape'])
    parser.add_argument(
        '--windows',
        nargs='*',
        default=['feedback', 'early_post_feedback', 'pre_stim_delay', 'response_epoch'],
        choices=['feedback', 'early_post_feedback', 'pre_stim_delay', 'response_epoch'],
    )
    parser.add_argument(
        '--feature-sets',
        nargs='*',
        default=['pfc', 'sr', 'pfc_sr', 'pfc_motif', 'sr_motif'],
        choices=['pfc', 'sr', 'pfc_sr', 'pfc_motif', 'sr_motif'],
    )
    parser.add_argument(
        '--conditions',
        nargs='*',
        default=['all_nonfirst', 'previous_error', 'stable_trials', 'true_switch_trial'],
        choices=['all_nonfirst', 'previous_error', 'stable_trials', 'true_switch_trial'],
    )
    parser.add_argument('--max-pca-components', type=int, default=20)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def set_test_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def sem(series):
    count = series.count()
    if count <= 1:
        return np.nan
    return float(series.std(ddof=1) / math.sqrt(count))


def feature_indices(model, feature_set):
    cg = model.rnn.cg_idx
    groups = {
        'pfc': ['pfc_esoma', 'pfc_edend', 'pfc_pv', 'pfc_sst', 'pfc_vip'],
        'sr': ['sr_esoma', 'sr_edend', 'sr_pv', 'sr_sst', 'sr_vip'],
        'pfc_sr': [
            'pfc_esoma', 'pfc_edend', 'pfc_pv', 'pfc_sst', 'pfc_vip',
            'sr_esoma', 'sr_edend', 'sr_pv', 'sr_sst', 'sr_vip',
        ],
        'pfc_motif': ['pfc_edend', 'pfc_sst', 'pfc_vip'],
        'sr_motif': ['sr_edend', 'sr_sst', 'sr_vip'],
    }[feature_set]
    idx = []
    for group in groups:
        idx.extend(cg[group])
    return idx


def stack_activity(data):
    trials = [trial.numpy() if hasattr(trial, 'numpy') else np.asarray(trial) for trial in data['rnn_activity']]
    min_steps = min(trial.shape[0] for trial in trials)
    return np.stack([trial[:min_steps] for trial in trials], axis=0)


def condition_masks_trial_level(data, switch_every):
    perfs = np.asarray(data['perfs'], dtype=float)
    masks_sample = mech.condition_masks(perfs, switch_every, switch_trials=data.get('switch_trials'))

    trial_masks = {}
    for condition, mask in masks_sample.items():
        if condition == 'previous_error':
            trial_masks[condition] = mask.mean(axis=1) >= 0.5
        elif condition == 'previous_correct':
            trial_masks[condition] = mask.mean(axis=1) >= 0.5
        elif condition == 'all_nonfirst':
            trial_masks[condition] = mask.any(axis=1)
        else:
            trial_masks[condition] = mask.any(axis=1)
    return trial_masks


def decode_scores(X, y, cv_folds, max_pca_components, random_state):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    valid = np.isfinite(X).all(axis=1)
    X = X[valid]
    y = y[valid]
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return {
            'n_samples': int(len(y)),
            'n_color': int((y == 0).sum()),
            'n_shape': int((y == 1).sum()),
            'cv_folds': 0,
            'accuracy_mean': np.nan,
            'accuracy_std': np.nan,
            'balanced_accuracy_mean': np.nan,
            'balanced_accuracy_std': np.nan,
    }

    n_splits = min(int(cv_folds), int(counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(cv.split(X, y))
    min_train_samples = min(len(train_idx) for train_idx, _ in splits)

    steps = [('scale', StandardScaler())]
    if max_pca_components > 0 and X.shape[1] > max_pca_components:
        # PCA is fit inside each cross-validation fold, so n_components must
        # also respect the smallest training fold, not just the full dataset.
        n_components = min(int(max_pca_components), int(X.shape[1]), int(min_train_samples))
        if n_components >= 1 and n_components < X.shape[1]:
            steps.append(('pca', PCA(n_components=n_components, random_state=random_state)))
    steps.append(('clf', LogisticRegression(max_iter=1000, solver='liblinear', random_state=random_state)))
    clf = Pipeline(steps)

    acc = []
    bal = []
    for train_idx, test_idx in splits:
        clf_fold = copy.deepcopy(clf)
        clf_fold.fit(X[train_idx], y[train_idx])
        pred = clf_fold.predict(X[test_idx])
        acc.append(accuracy_score(y[test_idx], pred))
        bal.append(balanced_accuracy_score(y[test_idx], pred))

    return {
        'n_samples': int(len(y)),
        'n_color': int((y == 0).sum()),
        'n_shape': int((y == 1).sum()),
        'cv_folds': int(n_splits),
        'accuracy_mean': float(np.mean(acc)),
        'accuracy_std': float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
        'balanced_accuracy_mean': float(np.mean(bal)),
        'balanced_accuracy_std': float(np.std(bal, ddof=1)) if len(bal) > 1 else 0.0,
    }


def decoding_rows_for_test(data, model, hp, hp_task, args, meta):
    activity = stack_activity(data)
    n_trials, n_steps, batch_size, _ = activity.shape
    windows = mech.window_slices(hp, hp_task, n_steps)
    trial_masks = condition_masks_trial_level(data, args.switch_every)
    labels = np.asarray([RULE_TO_INT[rule] for rule in data['rules']], dtype=int)
    rows = []

    for feature_set in args.feature_sets:
        idx = feature_indices(model, feature_set)
        if not idx:
            continue
        pop_activity = activity[:, :, :, idx]
        for window in args.windows:
            start, stop = windows[window]
            if stop <= start:
                continue
            # One sample per trial: average across time within the window and
            # across the independent card-sort batch for that hidden rule.
            X_all = pop_activity[:, start:stop, :, :].mean(axis=(1, 2))
            for condition in args.conditions:
                mask = trial_masks.get(condition)
                if mask is None or not mask.any():
                    continue
                random_state = int(meta['test_seed'] + len(feature_set) * 100 + len(window) * 10 + len(condition))
                scores = decode_scores(
                    X=X_all[mask],
                    y=labels[mask],
                    cv_folds=args.cv_folds,
                    max_pca_components=args.max_pca_components,
                    random_state=random_state,
                )
                row = dict(meta)
                row.update({
                    'feature_set': feature_set,
                    'window': window,
                    'condition': condition,
                    'n_trials_total': int(n_trials),
                    'batch_size': int(batch_size),
                    'n_features': int(X_all.shape[1]),
                    **scores,
                })
                rows.append(row)
    return rows


def write_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def write_summaries(analysis_dir, rows):
    raw = pd.DataFrame(rows)
    if raw.empty:
        pd.DataFrame().to_csv(analysis_dir / 'rule_decoding_summary.csv', index=False)
        pd.DataFrame().to_csv(analysis_dir / 'rule_decoding_deltas_vs_baseline.csv', index=False)
        pd.DataFrame().to_csv(analysis_dir / 'rule_decoding_delta_summary.csv', index=False)
        return

    group_cols = ['variant', 'feature_set', 'window', 'condition']
    seed_cols = group_cols + ['seed']
    seed_level = raw.groupby(seed_cols, as_index=False)['balanced_accuracy_mean'].mean()
    grouped = seed_level.groupby(group_cols)['balanced_accuracy_mean']
    summary = grouped.agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = grouped.apply(sem).to_numpy()
    summary.to_csv(analysis_dir / 'rule_decoding_summary.csv', index=False)
    seed_level.to_csv(analysis_dir / 'rule_decoding_seed_level.csv', index=False)

    keys = ['seed', 'init_rule', 'test_repeat', 'test_seed', 'feature_set', 'window', 'condition']
    wide = raw.pivot_table(index=keys, columns='variant', values='balanced_accuracy_mean', aggfunc='mean')
    delta_rows = []
    if 'baseline' in wide.columns:
        for variant in wide.columns:
            if variant == 'baseline':
                continue
            deltas = wide[variant] - wide['baseline']
            for idx, value in deltas.dropna().items():
                row = dict(zip(keys, idx))
                row.update({'variant': variant, 'delta_vs_baseline': float(value)})
                delta_rows.append(row)
    write_csv(analysis_dir / 'rule_decoding_deltas_vs_baseline.csv', delta_rows)

    deltas = pd.DataFrame(delta_rows)
    if not deltas.empty:
        seed_delta = deltas.groupby(['variant', 'feature_set', 'window', 'condition', 'seed'], as_index=False)['delta_vs_baseline'].mean()
        delta_grouped = seed_delta.groupby(['variant', 'feature_set', 'window', 'condition'])['delta_vs_baseline']
        delta_summary = delta_grouped.agg(['mean', 'std', 'count']).reset_index()
        delta_summary['sem'] = delta_grouped.apply(sem).to_numpy()
        delta_summary.to_csv(analysis_dir / 'rule_decoding_delta_summary.csv', index=False)
        seed_delta.to_csv(analysis_dir / 'rule_decoding_delta_seed_level.csv', index=False)
    else:
        pd.DataFrame().to_csv(analysis_dir / 'rule_decoding_delta_summary.csv', index=False)
        pd.DataFrame().to_csv(analysis_dir / 'rule_decoding_delta_seed_level.csv', index=False)


def write_quick_figure(analysis_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print('could not import matplotlib for figure: {}'.format(exc), flush=True)
        return

    summary_path = analysis_dir / 'rule_decoding_summary.csv'
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return

    keep = summary[
        (summary['condition'] == 'all_nonfirst') &
        (summary['feature_set'].isin(['pfc', 'sr', 'pfc_sr'])) &
        (summary['window'].isin(['feedback', 'early_post_feedback', 'pre_stim_delay']))
    ].copy()
    if keep.empty:
        return

    variants = ['baseline', 'pfc_vip_error', 'sr_vip_error', 'both_vip_error']
    windows = ['feedback', 'early_post_feedback', 'pre_stim_delay']
    feature_sets = ['pfc', 'sr', 'pfc_sr']
    colors = {
        'baseline': '#4a5568',
        'pfc_vip_error': '#8f5db7',
        'sr_vip_error': '#2f8f83',
        'both_vip_error': '#d9822b',
    }

    fig, axes = plt.subplots(1, len(feature_sets), figsize=(11, 3.2), sharey=True)
    for ax, feature_set in zip(axes, feature_sets):
        x = np.arange(len(windows))
        width = 0.18
        for i, variant in enumerate(variants):
            vals = []
            errs = []
            for window in windows:
                row = keep[
                    (keep['feature_set'] == feature_set) &
                    (keep['window'] == window) &
                    (keep['variant'] == variant)
                ]
                vals.append(float(row['mean'].iloc[0]) if not row.empty else np.nan)
                errs.append(float(row['sem'].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, yerr=errs, color=colors[variant], label=VARIANT_LABELS.get(variant, variant), alpha=0.9)
        ax.axhline(0.5, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(FEATURE_SET_LABELS[feature_set], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([WINDOW_LABELS[w] for w in windows], rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0.45, 1.0)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Rule decoding balanced accuracy')
    axes[-1].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.suptitle('Linear decoding of hidden rule from population activity', fontsize=11)
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(analysis_dir / f'rule_decoding_summary.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(args.output_dir) / 'summaries' / 'rule_decoding'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    device = mech.choose_device(args.device)
    model_paths = mech.find_models(args)

    print('rule-decoding analysis_dir={}'.format(analysis_dir), flush=True)
    print('found {} models'.format(len(model_paths)), flush=True)
    print('variants={}'.format(args.variants), flush=True)
    print('seeds={}'.format(args.seeds), flush=True)
    print('n_trials={} repeats={} init_rules={}'.format(args.n_trials, args.test_repeats, args.init_rules), flush=True)
    print('windows={}'.format(args.windows), flush=True)
    print('feature_sets={}'.format(args.feature_sets), flush=True)
    print('conditions={}'.format(args.conditions), flush=True)
    print('device={}'.format(device), flush=True)

    if args.dry_run:
        print('dry run complete; no model forward passes launched', flush=True)
        return

    rows = []
    model_rows = []
    for path in model_paths:
        print('loading {}'.format(path.name), flush=True)
        saved, model, hp, hp_task = mech.load_saved(path, device)
        variant = mech.infer_variant(hp, path)
        seed = mech.infer_seed(hp, path)
        model_rows.append(mech.model_summary(path, saved, hp, variant, seed))

        for init_rule in args.init_rules:
            for test_repeat in range(args.test_repeats):
                test_seed = mech.paired_test_seed(
                    args.test_seed_base,
                    seed,
                    init_rule,
                    'none',
                    test_repeat,
                    pair_causal_seeds=True,
                )
                set_test_seed(test_seed, device)
                print(
                    '  test variant={} seed={} init_rule={} repeat={} test_seed={}'.format(
                        variant, seed, init_rule, test_repeat, test_seed
                    ),
                    flush=True,
                )
                perf, perf_rule, loss, data = mech.run_test(
                    model=model,
                    hp=hp,
                    hp_task=hp_task,
                    init_rule=init_rule,
                    n_trials=args.n_trials,
                    switch_every=args.switch_every,
                    causal_test='none',
                    opto_window='feedback',
                )
                meta = {
                    'file': path.name,
                    'variant': variant,
                    'seed': seed,
                    'init_rule': init_rule,
                    'test_repeat': test_repeat,
                    'test_seed': test_seed,
                    'n_trials': args.n_trials,
                    'switch_every': args.switch_every,
                    'test_switch_mode': data.get('switch_mode', hp.get('test_switch_mode', 'fixed_interval')),
                    'n_switches': len(data.get('switch_trials', [])),
                    'mean_perf': float(perf),
                    'mean_rule_perf': float(perf_rule),
                    'mean_loss': float(loss),
                }
                rows.extend(decoding_rows_for_test(data, model, hp, hp_task, args, meta))

    write_csv(analysis_dir / 'model_summary.csv', model_rows)
    write_csv(analysis_dir / 'rule_decoding_scores.csv', rows)
    write_summaries(analysis_dir, rows)
    write_quick_figure(analysis_dir)
    print('wrote rule-decoding outputs to {}'.format(analysis_dir), flush=True)


if __name__ == '__main__':
    main()
