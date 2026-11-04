#!/usr/bin/env python3
"""Analyze trained feedback-gating models for VIP/SST mechanism signatures."""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE_ROOT / '.mplconfig'))
sys.path.insert(0, str(REPO_ROOT))

import model_working  # noqa: F401,E402
import train  # noqa: F401,E402
from functions import get_default_hp, test_frozen_weights  # noqa: E402
from model_working import Net_readoutSR_working  # noqa: E402


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
VARIANTS = ['baseline', 'both_vip_error', 'both_vip_all_feedback', 'pfc_vip_error', 'sr_vip_error', 'vip_only_error']
CAUSAL_TESTS = [
    'none',
    'vip_silence_feedback',
    'sst_silence_feedback',
    'sr_vip_silence_feedback',
    'pfc_vip_silence_feedback',
    'sr_sst_silence_feedback',
    'pfc_sst_silence_feedback',
    'sr_edend_silence_feedback',
    'pfc_edend_silence_feedback',
]
WEIGHT_ABLATIONS = ['none', 'vip_feedback_ablate', 'sr_vip_feedback_ablate', 'pfc_vip_feedback_ablate']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='Training output root containing saved_models/.')
    parser.add_argument('--saved-models-dir', default=None, help='Override saved model directory.')
    parser.add_argument('--analysis-dir', default=None, help='Directory for mechanism CSV outputs.')
    parser.add_argument('--model-glob', default='*feedback_noisy_task_finetune*', help='Glob within saved_models/.')
    parser.add_argument('--variants', nargs='*', default=['baseline', 'both_vip_error', 'both_vip_all_feedback'])
    parser.add_argument('--seeds', nargs='*', type=int, default=None)
    parser.add_argument('--max-models', type=int, default=None)
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--test-repeats', type=int, default=1, help='Number of paired frozen-test repeats per model/init rule/condition.')
    parser.add_argument('--test-seed-base', type=int, default=12345, help='Base seed for paired frozen-test repeats.')
    parser.add_argument('--switch-every', type=int, default=20)
    parser.add_argument('--init-rules', nargs='*', default=['color', 'shape'], choices=['color', 'shape'])
    parser.add_argument('--causal-tests', nargs='*', default=['none'], choices=CAUSAL_TESTS)
    parser.add_argument(
        '--weight-ablations',
        nargs='*',
        default=['none'],
        choices=WEIGHT_ABLATIONS,
        help='Analysis-only ablations of previous-reward feedback routes before frozen testing.',
    )
    parser.add_argument(
        '--opto-window',
        default='feedback',
        choices=['feedback', 'early_post_feedback', 'feedback_plus_early_post', 'feedback_delay', 'full_trial'],
        help='Epoch used for causal silencing. feedback=trial-history epoch; early_post_feedback=immediately after feedback; feedback_plus_early_post=both.',
    )
    parser.add_argument(
        '--pair-causal-seeds',
        action='store_true',
        help='Use the same frozen-test seed for intact and causal silencing conditions.',
    )
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    return parser.parse_args()


def infer_variant(hp, path):
    jobname = hp.get('jobname', '') or path.name
    for variant in sorted(VARIANTS, key=len, reverse=True):
        if variant in jobname:
            return variant
    return 'unknown'


def infer_seed(hp, path):
    jobname = hp.get('jobname', '') or path.name
    match = re.search(r'seed(\d+)', jobname)
    if match:
        return int(match.group(1))
    return int(hp.get('torch_seed', -1))


def status_from_name(path):
    return path.name.split('_', 1)[0] if '_' in path.name else 'unknown'


def choose_device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_saved(path, device):
    saved = torch.load(str(path), map_location=device)
    hp = copy.deepcopy(saved.get('hp', {}))
    hp_default, _, _ = get_default_hp()
    for key, value in hp_default.items():
        hp.setdefault(key, value)
    hp_task = copy.deepcopy(saved.get('hp_task', {}))
    model = Net_readoutSR_working(hp)
    model.to(device)
    model.rnn.to(device)
    model.load_state_dict(saved['model_state_dict'], strict=False)
    model.rnn.apply_prev_rew_mask()
    model.eval()
    return saved, model, hp, hp_task


def set_test_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def paired_test_seed(base_seed, train_seed, init_rule, causal_test, test_repeat, pair_causal_seeds=False):
    init_offset = {'color': 0, 'shape': 1}.get(init_rule, 0)
    causal_offset = 0 if pair_causal_seeds else {name: idx for idx, name in enumerate(CAUSAL_TESTS)}.get(causal_test, 0)
    train_seed = max(0, int(train_seed))
    return int(base_seed + train_seed * 100000 + init_offset * 10000 + causal_offset * 1000 + test_repeat)


def find_models(args):
    saved_dir = Path(args.saved_models_dir) if args.saved_models_dir else Path(args.output_dir) / 'saved_models'
    paths = sorted(path for path in saved_dir.glob(args.model_glob) if path.is_file())
    selected = []
    device = torch.device('cpu')
    for path in paths:
        try:
            saved = torch.load(str(path), map_location=device)
        except Exception as exc:
            print('skipping unreadable model {}: {}'.format(path, exc), flush=True)
            continue
        hp = saved.get('hp', {})
        variant = infer_variant(hp, path)
        seed = infer_seed(hp, path)
        if args.variants and variant not in args.variants:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue
        selected.append(path)
    if args.max_models is not None:
        selected = selected[:args.max_models]
    if not selected:
        raise FileNotFoundError('no saved models matched in {}'.format(saved_dir))
    return selected


def tail_mean(values, n=5):
    if len(values) == 0:
        return np.nan
    return float(np.asarray(values[-n:], dtype=float).mean())


def model_summary(path, saved, hp, variant, seed):
    return {
        'file': path.name,
        'status': status_from_name(path),
        'jobname': hp.get('jobname', ''),
        'variant': variant,
        'seed': seed,
        'n_train_points': len(saved.get('perf_list', [])),
        'n_test_points': len(saved.get('test_perf_list', [])),
        'mean_last5_train_perf': tail_mean(saved.get('perf_list', [])),
        'mean_last5_train_rule_perf': tail_mean(saved.get('perf_rule_list', [])),
        'mean_last5_test_perf': tail_mean(saved.get('test_perf_list', [])),
        'mean_last5_test_rule_perf': tail_mean(saved.get('test_perf_rule_list', [])),
        'prev_rew_to_pfc_default': hp.get('prev_rew_to_pfc_default', True),
        'prev_rew_to_pfc_vip': hp.get('prev_rew_to_pfc_vip', False),
        'prev_rew_to_sr_vip': hp.get('prev_rew_to_sr_vip', False),
        'prev_rew_negative_only_to_vip': hp.get('prev_rew_negative_only_to_vip', False),
        'prev_rew_to_pfc_scale': hp.get('prev_rew_to_pfc_scale', 1.0),
        'prev_rew_to_vip_scale': hp.get('prev_rew_to_vip_scale', 1.0),
        'warm_start_vip_feedback_init': hp.get('warm_start_vip_feedback_init', 0.0),
        'feedback_noise_mode': hp.get('feedback_noise_mode', 'none'),
        'feedback_noise_prob': hp.get('feedback_noise_prob', 0.0),
        'feedback_noise_flip_fraction': hp.get('feedback_noise_flip_fraction', 0.5),
    }


def feedback_weight_rows(model, variant, seed, file_name, weight_ablation):
    rnn = model.rnn
    effective = rnn.effective_weight(w=rnn.w_rew, mask=rnn.mask_rew).detach().cpu()
    raw = rnn.w_rew.detach().cpu()
    mask = rnn.mask_rew.detach().cpu()
    rows = []
    for channel_idx, channel in [(0, 'correct'), (1, 'error')]:
        for population in POPULATIONS:
            idx = rnn.cg_idx[population]
            mask_vals = mask[channel_idx, idx]
            raw_vals = raw[channel_idx, idx]
            eff_vals = effective[channel_idx, idx]
            active = mask_vals != 0
            rows.append({
                'file': file_name,
                'variant': variant,
                'seed': seed,
                'weight_ablation': weight_ablation,
                'feedback_channel': channel,
                'population': population,
                'n_targets': int(active.sum().item()),
                'mask_mean': float(mask_vals.mean().item()),
                'raw_abs_mean_all': float(raw_vals.abs().mean().item()),
                'raw_abs_mean_targets': float(raw_vals[active].abs().mean().item()) if active.any() else 0.0,
                'effective_abs_mean_all': float(eff_vals.abs().mean().item()),
                'effective_abs_mean_targets': float(eff_vals[active].abs().mean().item()) if active.any() else 0.0,
                'effective_abs_sum': float(eff_vals.abs().sum().item()),
            })
    return rows


def stack_test_arrays(data):
    perfs = np.asarray(data['perfs'], dtype=float)
    perf_rules = np.asarray(data['perf_rules'], dtype=float)
    activity_trials = [trial.numpy() if hasattr(trial, 'numpy') else np.asarray(trial) for trial in data['rnn_activity']]
    min_steps = min(trial.shape[0] for trial in activity_trials)
    activity = np.stack([trial[:min_steps] for trial in activity_trials], axis=0)
    return perfs, perf_rules, activity


def window_slices(hp, hp_task, n_steps):
    feedback_end = int(hp_task.get('trial_history_end', 100) // hp.get('dt', 10))
    center_on = int(hp_task.get('center_card_on', 1100) // hp.get('dt', 10))
    test_on = int(hp_task.get('test_cards_on', 1600) // hp.get('dt', 10))
    return {
        'feedback': (0, min(n_steps, feedback_end)),
        'early_post_feedback': (min(n_steps, feedback_end), min(n_steps, feedback_end + max(1, feedback_end))),
        'pre_stim_delay': (min(n_steps, feedback_end), min(n_steps, center_on)),
        'response_epoch': (min(n_steps, test_on), n_steps),
    }


def trial_positions_after_switch(n_trials, switch_every, switch_trials=None):
    if switch_trials is None:
        switch_trials = list(range(0, n_trials, switch_every))
    switch_trials = sorted({int(tr) for tr in switch_trials if 0 <= int(tr) < n_trials})
    if 0 not in switch_trials:
        switch_trials = [0] + switch_trials

    positions = np.zeros(n_trials, dtype=int)
    last_switch = 0
    next_idx = 1
    for tr in range(n_trials):
        while next_idx < len(switch_trials) and switch_trials[next_idx] <= tr:
            last_switch = switch_trials[next_idx]
            next_idx += 1
        positions[tr] = tr - last_switch
    return positions


def condition_masks(perfs, switch_every, switch_trials=None):
    n_trials, batch_size = perfs.shape
    masks = {}
    nonfirst = np.ones((n_trials, batch_size), dtype=bool)
    nonfirst[0, :] = False
    masks['all_nonfirst'] = nonfirst
    prev_correct = np.zeros_like(nonfirst)
    prev_error = np.zeros_like(nonfirst)
    prev_correct[1:, :] = perfs[:-1, :] > 0.5
    prev_error[1:, :] = perfs[:-1, :] <= 0.5
    masks['previous_correct'] = prev_correct
    masks['previous_error'] = prev_error
    if switch_trials is None:
        switch_trials = list(range(0, n_trials, switch_every))
    switch_trials = sorted({int(tr) for tr in switch_trials if 0 <= int(tr) < n_trials})
    true_switch = np.zeros_like(nonfirst)
    for tr in switch_trials:
        if tr > 0:
            true_switch[tr, :] = True
    masks['true_switch_trial'] = true_switch
    stable = np.zeros_like(nonfirst)
    positions = trial_positions_after_switch(n_trials, switch_every, switch_trials)
    for tr in range(1, n_trials):
        if positions[tr] >= 5:
            stable[tr, :] = True
    masks['stable_trials'] = stable
    return masks


def activity_rows(
    data,
    model,
    hp,
    hp_task,
    variant,
    seed,
    file_name,
    init_rule,
    causal_test,
    opto_window,
    weight_ablation,
    test_repeat,
    test_seed,
    switch_every,
):
    perfs, perf_rules, activity = stack_test_arrays(data)
    n_trials, n_steps, batch_size, _ = activity.shape
    switch_trials = data.get('switch_trials')
    masks = condition_masks(perfs, switch_every, switch_trials=switch_trials)
    windows = window_slices(hp, hp_task, n_steps)
    rows = []
    contrast_rows = []

    condition_means = {}
    for condition, mask in masks.items():
        if not mask.any():
            continue
        for population in POPULATIONS:
            idx = model.rnn.cg_idx[population]
            pop = activity[:, :, :, idx].mean(axis=3).transpose(0, 2, 1)
            selected = pop[mask]
            if selected.size == 0:
                continue
            selected = selected.reshape(-1, n_steps)
            for window, (start, stop) in windows.items():
                if stop <= start:
                    continue
                mean_activity = float(selected[:, start:stop].mean())
                condition_means[(condition, population, window)] = mean_activity
                rows.append({
                    'file': file_name,
                    'variant': variant,
                    'seed': seed,
                    'init_rule': init_rule,
                    'causal_test': causal_test,
                    'opto_window': opto_window,
                    'weight_ablation': weight_ablation,
                    'test_repeat': test_repeat,
                    'test_seed': test_seed,
                    'condition': condition,
                    'population': population,
                    'window': window,
                    'mean_activity': mean_activity,
                    'n_samples': int(selected.shape[0]),
                })

    for population in POPULATIONS:
        for window in windows:
            error_key = ('previous_error', population, window)
            correct_key = ('previous_correct', population, window)
            if error_key in condition_means and correct_key in condition_means:
                contrast_rows.append({
                    'file': file_name,
                    'variant': variant,
                    'seed': seed,
                    'init_rule': init_rule,
                    'causal_test': causal_test,
                    'opto_window': opto_window,
                    'weight_ablation': weight_ablation,
                    'test_repeat': test_repeat,
                    'test_seed': test_seed,
                    'population': population,
                    'window': window,
                    'error_minus_correct_activity': condition_means[error_key] - condition_means[correct_key],
                })

    behavior_rows = []
    for condition, mask in masks.items():
        if not mask.any():
            continue
        behavior_rows.append({
            'file': file_name,
            'variant': variant,
            'seed': seed,
            'init_rule': init_rule,
            'causal_test': causal_test,
            'opto_window': opto_window,
            'weight_ablation': weight_ablation,
            'test_repeat': test_repeat,
            'test_seed': test_seed,
            'condition': condition,
            'mean_perf': float(perfs[mask].mean()),
            'mean_rule_perf': float(perf_rules[mask].mean()),
            'n_samples': int(mask.sum()),
        })

    position_rows = []
    positions = trial_positions_after_switch(n_trials, switch_every, switch_trials=switch_trials)
    max_position = min(switch_every, int(positions.max()) + 1)
    for pos in range(max_position):
        tr_mask = np.zeros((n_trials, batch_size), dtype=bool)
        tr_mask[positions == pos, :] = True
        if not tr_mask.any():
            continue
        position_rows.append({
            'file': file_name,
            'variant': variant,
            'seed': seed,
            'init_rule': init_rule,
            'causal_test': causal_test,
            'opto_window': opto_window,
            'weight_ablation': weight_ablation,
            'test_repeat': test_repeat,
            'test_seed': test_seed,
            'trial_position_after_switch': pos,
            'mean_perf': float(perfs[tr_mask].mean()),
            'mean_rule_perf': float(perf_rules[tr_mask].mean()),
            'n_samples': int(tr_mask.sum()),
        })

    return rows, contrast_rows, behavior_rows, position_rows


def apply_weight_ablation(model, weight_ablation):
    if weight_ablation == 'none':
        return
    if weight_ablation == 'vip_feedback_ablate':
        groups = ['sr_vip', 'pfc_vip']
    elif weight_ablation == 'sr_vip_feedback_ablate':
        groups = ['sr_vip']
    elif weight_ablation == 'pfc_vip_feedback_ablate':
        groups = ['pfc_vip']
    else:
        raise ValueError('unknown weight_ablation {}'.format(weight_ablation))
    with torch.no_grad():
        for group in groups:
            model.rnn.mask_rew[:, model.rnn.cg_idx[group]] = 0


def opto_time_range(hp_task, hp, opto_window):
    dt = hp.get('dt', 10)
    feedback_end = int(hp_task.get('trial_history_end', 100) // dt)
    center_on = int(hp_task.get('center_card_on', 1100) // dt)
    trial_end = int(hp_task.get('trial_end', 2100) // dt)
    early_post_end = min(trial_end, feedback_end + max(1, feedback_end))
    if opto_window == 'feedback':
        start, stop = 0, feedback_end
    elif opto_window == 'early_post_feedback':
        start, stop = feedback_end, early_post_end
    elif opto_window == 'feedback_plus_early_post':
        start, stop = 0, early_post_end
    elif opto_window == 'feedback_delay':
        start, stop = 0, center_on
    elif opto_window == 'full_trial':
        start, stop = 0, trial_end
    else:
        raise ValueError('unknown opto_window {}'.format(opto_window))
    return set(range(max(0, start), max(0, stop)))


def make_opto(model, hp_task, hp, causal_test, opto_window):
    if causal_test == 'none':
        return None
    causal_groups = {
        'vip_silence_feedback': ['sr_vip', 'pfc_vip'],
        'sst_silence_feedback': ['sr_sst', 'pfc_sst'],
        'sr_vip_silence_feedback': ['sr_vip'],
        'pfc_vip_silence_feedback': ['pfc_vip'],
        'sr_sst_silence_feedback': ['sr_sst'],
        'pfc_sst_silence_feedback': ['pfc_sst'],
        'sr_edend_silence_feedback': ['sr_edend'],
        'pfc_edend_silence_feedback': ['pfc_edend'],
    }
    if causal_test not in causal_groups:
        raise ValueError('unknown causal test {}'.format(causal_test))
    groups = causal_groups[causal_test]
    idx = []
    for group in groups:
        idx.extend(model.rnn.cg_idx[group])
    return {'t': opto_time_range(hp_task, hp, opto_window), 'neuron_idx': idx, 'value': 0}


def run_test(model, hp, hp_task, init_rule, n_trials, switch_every, causal_test, opto_window):
    loss_fnc = nn.MSELoss()
    opto = make_opto(model, hp_task, hp, causal_test, opto_window)
    return test_frozen_weights(
        model=model,
        n_trials_test=n_trials,
        init_rule=init_rule,
        hp=hp,
        hp_task=hp_task,
        loss_fnc=loss_fnc,
        task=hp['task'],
        delay_var=hp_task.get('delay_var', 0),
        reset_network=False,
        give_prev_stim=hp.get('give_prev_stim', True),
        give_prev_choice=hp.get('give_prev_choice', True),
        give_prev_rew=hp.get('give_prev_rew', True),
        plot=False,
        toprint=False,
        random_switch=False,
        switch_every_test=switch_every,
        n_switches=max(1, n_trials // switch_every),
        opto=opto,
    )


def write_csv(path, rows):
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def write_pairwise_variant_deltas(path, behavior_rows):
    df = pd.DataFrame(behavior_rows)
    if df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = df[df['condition'].isin(['all_nonfirst', 'previous_error', 'true_switch_trial', 'stable_trials'])]
    keys = ['seed', 'init_rule', 'causal_test', 'opto_window', 'weight_ablation', 'test_repeat', 'test_seed', 'condition']
    rows = []
    for metric in ['mean_perf', 'mean_rule_perf']:
        wide = df.pivot_table(index=keys, columns='variant', values=metric, aggfunc='mean')
        if 'baseline' not in wide.columns:
            continue
        for variant in wide.columns:
            if variant == 'baseline':
                continue
            delta = wide[variant] - wide['baseline']
            for idx, value in delta.dropna().items():
                row = dict(zip(keys, idx))
                row.update({'metric': metric, 'variant': variant, 'delta_vs_baseline': float(value)})
                rows.append(row)
    write_csv(path, rows)


def write_weight_ablation_deltas(path, behavior_rows):
    df = pd.DataFrame(behavior_rows)
    if df.empty or 'weight_ablation' not in df.columns:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = df[df['condition'].isin(['all_nonfirst', 'previous_error', 'true_switch_trial', 'stable_trials'])]
    keys = ['variant', 'seed', 'init_rule', 'causal_test', 'opto_window', 'test_repeat', 'test_seed', 'condition']
    rows = []
    for metric in ['mean_perf', 'mean_rule_perf']:
        wide = df.pivot_table(index=keys, columns='weight_ablation', values=metric, aggfunc='mean')
        if 'none' not in wide.columns:
            continue
        for ablation in wide.columns:
            if ablation == 'none':
                continue
            delta = wide[ablation] - wide['none']
            for idx, value in delta.dropna().items():
                row = dict(zip(keys, idx))
                row.update({'metric': metric, 'weight_ablation': ablation, 'delta_vs_intact': float(value)})
                rows.append(row)
    write_csv(path, rows)


def write_causal_test_deltas(path, behavior_rows):
    df = pd.DataFrame(behavior_rows)
    if df.empty or 'causal_test' not in df.columns:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = df[df['condition'].isin(['all_nonfirst', 'previous_error', 'true_switch_trial', 'stable_trials'])]
    keys = ['variant', 'seed', 'init_rule', 'opto_window', 'weight_ablation', 'test_repeat', 'test_seed', 'condition']
    rows = []
    for metric in ['mean_perf', 'mean_rule_perf']:
        wide = df.pivot_table(index=keys, columns='causal_test', values=metric, aggfunc='mean')
        if 'none' not in wide.columns:
            continue
        for causal_test in wide.columns:
            if causal_test == 'none':
                continue
            delta = wide[causal_test] - wide['none']
            for idx, value in delta.dropna().items():
                row = dict(zip(keys, idx))
                row.update({'metric': metric, 'causal_test': causal_test, 'delta_vs_intact': float(value)})
                rows.append(row)
    write_csv(path, rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_dir / 'summaries' / 'mechanism'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model_paths = find_models(args)
    print('analyzing {} models on {}'.format(len(model_paths), device), flush=True)

    model_rows = []
    weight_rows = []
    metric_rows = []
    activity_summary_rows = []
    activity_contrast_rows = []
    behavior_rows = []
    position_rows = []

    for path in model_paths:
        print('loading {}'.format(path.name), flush=True)
        saved, model, hp, hp_task = load_saved(path, device)
        variant = infer_variant(hp, path)
        seed = infer_seed(hp, path)
        model_rows.append(model_summary(path, saved, hp, variant, seed))
        original_mask_rew = model.rnn.mask_rew.detach().clone()

        for weight_ablation in args.weight_ablations:
            with torch.no_grad():
                model.rnn.mask_rew.copy_(original_mask_rew)
            apply_weight_ablation(model, weight_ablation)
            weight_rows.extend(feedback_weight_rows(model, variant, seed, path.name, weight_ablation))

            for init_rule in args.init_rules:
                for causal_test in args.causal_tests:
                    for test_repeat in range(args.test_repeats):
                        test_seed = paired_test_seed(
                            args.test_seed_base,
                            seed,
                            init_rule,
                            causal_test,
                            test_repeat,
                            pair_causal_seeds=args.pair_causal_seeds,
                        )
                        set_test_seed(test_seed, device)
                        print(
                            '  test init_rule={} causal={} weight_ablation={} repeat={} test_seed={}'.format(
                                init_rule, causal_test, weight_ablation, test_repeat, test_seed
                            ),
                            flush=True,
                        )
                        perf, perf_rule, loss, data = run_test(
                            model=model,
                            hp=hp,
                            hp_task=hp_task,
                            init_rule=init_rule,
                            n_trials=args.n_trials,
                            switch_every=args.switch_every,
                            causal_test=causal_test,
                            opto_window=args.opto_window,
                        )
                        metric_rows.append({
                            'file': path.name,
                            'variant': variant,
                            'seed': seed,
                            'init_rule': init_rule,
                            'causal_test': causal_test,
                            'opto_window': args.opto_window,
                            'weight_ablation': weight_ablation,
                            'test_repeat': test_repeat,
                            'test_seed': test_seed,
                            'n_trials': args.n_trials,
                            'switch_every': args.switch_every,
                            'test_switch_mode': data.get('switch_mode', hp.get('test_switch_mode', 'fixed_interval')),
                            'n_switches': len(data.get('switch_trials', [])),
                            'mean_perf': float(perf),
                            'mean_rule_perf': float(perf_rule),
                            'mean_loss': float(loss),
                        })
                        rows = activity_rows(
                            data,
                            model,
                            hp,
                            hp_task,
                            variant,
                            seed,
                            path.name,
                            init_rule,
                            causal_test,
                            args.opto_window,
                            weight_ablation,
                            test_repeat,
                            test_seed,
                            args.switch_every,
                        )
                        activity_summary_rows.extend(rows[0])
                        activity_contrast_rows.extend(rows[1])
                        behavior_rows.extend(rows[2])
                        position_rows.extend(rows[3])

    write_csv(analysis_dir / 'model_summary.csv', model_rows)
    write_csv(analysis_dir / 'feedback_weight_summary.csv', weight_rows)
    write_csv(analysis_dir / 'frozen_test_metrics.csv', metric_rows)
    write_csv(analysis_dir / 'activity_by_condition.csv', activity_summary_rows)
    write_csv(analysis_dir / 'activity_error_minus_correct.csv', activity_contrast_rows)
    write_csv(analysis_dir / 'behavior_by_condition.csv', behavior_rows)
    write_csv(analysis_dir / 'switch_position_behavior.csv', position_rows)
    write_pairwise_variant_deltas(analysis_dir / 'paired_behavior_deltas_vs_baseline.csv', behavior_rows)
    write_weight_ablation_deltas(analysis_dir / 'weight_ablation_behavior_deltas.csv', behavior_rows)
    write_causal_test_deltas(analysis_dir / 'causal_behavior_deltas.csv', behavior_rows)

    print('wrote mechanism outputs to {}'.format(analysis_dir), flush=True)


if __name__ == '__main__':
    main()
