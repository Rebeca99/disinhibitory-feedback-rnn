#!/usr/bin/env python3
"""Run frozen-weight feedback-routing comparisons and summarize activity."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE_ROOT / '.mplconfig'))
sys.path.insert(0, str(REPO_ROOT))

from functions import get_default_hp, test_frozen_weights  # noqa: E402
from model_working import Net_readoutSR_working  # noqa: E402


VARIANTS = {
    'baseline': {},
    'pfc_vip_error': {
        'prev_rew_to_pfc_vip': True,
        'prev_rew_negative_only_to_vip': True,
    },
    'sr_vip_error': {
        'prev_rew_to_sr_vip': True,
        'prev_rew_negative_only_to_vip': True,
    },
    'both_vip_error': {
        'prev_rew_to_pfc_vip': True,
        'prev_rew_to_sr_vip': True,
        'prev_rew_negative_only_to_vip': True,
    },
    'vip_only_error': {
        'prev_rew_to_pfc_default': False,
        'prev_rew_to_pfc_vip': True,
        'prev_rew_to_sr_vip': True,
        'prev_rew_negative_only_to_vip': True,
    },
    'both_vip_all_feedback': {
        'prev_rew_to_pfc_vip': True,
        'prev_rew_to_sr_vip': True,
        'prev_rew_negative_only_to_vip': False,
    },
}

POPULATIONS = [
    'sr_esoma',
    'sr_edend',
    'sr_pv',
    'sr_sst',
    'sr_vip',
    'pfc_esoma',
    'pfc_edend',
    'pfc_pv',
    'pfc_sst',
    'pfc_vip',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--trained-dir',
        default=str(WORKSPACE_ROOT / 'data' / 'liu_wang_2024' / 'trained_networks'),
        help='Directory containing pretrained Liu/Wang checkpoints.',
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Checkpoint filename. Defaults to the first sorted checkpoint in --trained-dir.',
    )
    parser.add_argument('--n-trials', type=int, default=30, help='Number of WCST trials per variant.')
    parser.add_argument('--switch-every', type=int, default=10, help='Rule switch interval for the smoke run.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed reused for each variant.')
    parser.add_argument(
        '--variants',
        nargs='*',
        default=['baseline', 'pfc_vip_error', 'sr_vip_error', 'both_vip_error'],
        choices=sorted(VARIANTS.keys()),
        help='Variants to run.',
    )
    parser.add_argument(
        '--out-dir',
        default=str(WORKSPACE_ROOT / 'analysis' / 'feedback_routing_smoke'),
        help='Output directory for CSV and JSON summaries.',
    )
    return parser.parse_args()


def find_checkpoint(trained_dir, checkpoint_name):
    trained_dir = Path(trained_dir)
    if checkpoint_name is not None:
        checkpoint = trained_dir / checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        return checkpoint

    checkpoints = sorted(path for path in trained_dir.iterdir() if path.is_file())
    if not checkpoints:
        raise FileNotFoundError('no checkpoint files found in {}'.format(trained_dir))
    return checkpoints[0]


def load_model_with_overrides(checkpoint_path, overrides):
    if torch.cuda.is_available():
        saved_data = torch.load(str(checkpoint_path))
    else:
        saved_data = torch.load(str(checkpoint_path), map_location=torch.device('cpu'))

    hp = copy.deepcopy(saved_data['hp'])
    hp_task = copy.deepcopy(saved_data['hp_task'])
    hp_default, _, _ = get_default_hp()
    for key, value in hp_default.items():
        hp.setdefault(key, value)
    hp.update(overrides)

    model = Net_readoutSR_working(hp)
    model.load_state_dict(saved_data['model_state_dict'], strict=False)
    model.rnn.apply_prev_rew_mask()
    loss_fnc = nn.MSELoss()
    return model, hp, hp_task, loss_fnc


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def population_timecourse(data, rnn, hp_task, hp):
    feedback_end = int(hp_task['trial_history_end'] // hp['dt'])
    perfs = np.asarray(data['perfs'], dtype=bool)
    activity_trials = [trial.numpy() for trial in data['rnn_activity']]
    min_steps = min(trial.shape[0] for trial in activity_trials)
    activity = np.stack([trial[:min_steps] for trial in activity_trials], axis=0)
    n_trials, n_steps, batch_size, _ = activity.shape
    rows = []
    summary_rows = []

    conditions = {
        'all_previous_feedback': np.ones((n_trials, batch_size), dtype=bool),
        'previous_correct': np.zeros((n_trials, batch_size), dtype=bool),
        'previous_error': np.zeros((n_trials, batch_size), dtype=bool),
    }
    conditions['all_previous_feedback'][0, :] = False
    if n_trials > 1:
        conditions['previous_correct'][1:, :] = perfs[:-1, :]
        conditions['previous_error'][1:, :] = ~perfs[:-1, :]

    for condition, mask in conditions.items():
        for population in POPULATIONS:
            pop_idx = rnn.cg_idx[population]
            pop_activity = activity[:, :, :, pop_idx].mean(axis=3).transpose(0, 2, 1)
            selected = pop_activity[mask]
            if selected.size == 0:
                continue
            selected = selected.reshape(-1, n_steps)
            time_mean = selected.mean(axis=0)
            for t_step, mean_activity in enumerate(time_mean):
                rows.append({
                    'feedback_condition': condition,
                    'population': population,
                    'time_step': t_step,
                    'time_ms': t_step * hp['dt'],
                    'mean_activity': float(mean_activity),
                })
            summary_rows.append({
                'feedback_condition': condition,
                'population': population,
                'n_samples': int(selected.shape[0]),
                'feedback_window_mean': float(time_mean[:feedback_end].mean()),
                'post_feedback_window_mean': float(time_mean[feedback_end:min(n_steps, feedback_end * 2)].mean()),
            })

    return rows, summary_rows


def feedback_weight_summary(rnn):
    weights = rnn.effective_weight(w=rnn.w_rew, mask=rnn.mask_rew).detach().cpu()
    rows = []
    for row_idx, row_name in [(0, 'correct_channel'), (1, 'error_channel')]:
        for population in POPULATIONS:
            values = weights[row_idx, rnn.cg_idx[population]]
            rows.append({
                'feedback_channel': row_name,
                'population': population,
                'n_masked_targets': int((rnn.mask_rew[row_idx, rnn.cg_idx[population]] != 0).sum().item()),
                'mean_abs_effective_weight': float(values.abs().mean().item()),
                'max_abs_effective_weight': float(values.abs().max().item()),
            })
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = find_checkpoint(args.trained_dir, args.checkpoint)

    metrics = []
    all_timecourse_rows = []
    all_summary_rows = []
    all_weight_rows = []

    for variant_name in args.variants:
        print('running variant: {}'.format(variant_name), flush=True)
        reset_seeds(args.seed)
        model, hp, hp_task, loss_fnc = load_model_with_overrides(checkpoint, VARIANTS[variant_name])
        perf, perf_rule, loss, data = test_frozen_weights(
            model=model,
            n_trials_test=args.n_trials,
            init_rule='color',
            hp=hp,
            hp_task=hp_task,
            loss_fnc=loss_fnc,
            task=hp['task'],
            delay_var=hp_task.get('delay_var', 0),
            random_switch=False,
            switch_every_test=args.switch_every,
            n_switches=max(1, args.n_trials // args.switch_every),
            plot=False,
            toprint=False,
        )

        metrics.append({
            'variant': variant_name,
            'checkpoint': checkpoint.name,
            'n_trials': args.n_trials,
            'switch_every': args.switch_every,
            'mean_perf': float(perf),
            'mean_rule_perf': float(perf_rule),
            'mean_loss': float(loss),
            'note': 'Frozen-weight rerouting is a smoke test; VIP feedback weights were not trained in the baseline checkpoints.',
        })

        timecourse_rows, summary_rows = population_timecourse(data, model.rnn, hp_task, hp)
        for row in timecourse_rows:
            row['variant'] = variant_name
            all_timecourse_rows.append(row)
        for row in summary_rows:
            row['variant'] = variant_name
            all_summary_rows.append(row)
        for row in feedback_weight_summary(model.rnn):
            row['variant'] = variant_name
            all_weight_rows.append(row)

    write_csv(
        out_dir / 'variant_metrics.csv',
        metrics,
        ['variant', 'checkpoint', 'n_trials', 'switch_every', 'mean_perf', 'mean_rule_perf', 'mean_loss', 'note'],
    )
    write_csv(
        out_dir / 'population_timecourses.csv',
        all_timecourse_rows,
        ['variant', 'feedback_condition', 'population', 'time_step', 'time_ms', 'mean_activity'],
    )
    write_csv(
        out_dir / 'population_feedback_window_summary.csv',
        all_summary_rows,
        ['variant', 'feedback_condition', 'population', 'n_samples', 'feedback_window_mean', 'post_feedback_window_mean'],
    )
    write_csv(
        out_dir / 'feedback_weight_summary.csv',
        all_weight_rows,
        ['variant', 'feedback_channel', 'population', 'n_masked_targets', 'mean_abs_effective_weight', 'max_abs_effective_weight'],
    )

    manifest = {
        'checkpoint': str(checkpoint),
        'variants': args.variants,
        'n_trials': args.n_trials,
        'switch_every': args.switch_every,
        'seed': args.seed,
        'outputs': [
            'variant_metrics.csv',
            'population_timecourses.csv',
            'population_feedback_window_summary.csv',
            'feedback_weight_summary.csv',
        ],
        'caveat': 'Frozen-weight rerouting exposes weights that were masked during baseline training; use this as a wiring and analysis smoke test before matched retraining.',
    }
    with open(out_dir / 'manifest.json', 'w') as handle:
        json.dump(manifest, handle, indent=2)

    print('wrote feedback-routing summaries to {}'.format(out_dir), flush=True)


if __name__ == '__main__':
    main()
