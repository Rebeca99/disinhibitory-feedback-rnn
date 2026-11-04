#!/usr/bin/env python3
"""Summarize completed feedback-gating training runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE_ROOT / '.mplconfig'))
sys.path.insert(0, str(REPO_ROOT))

import model_working  # noqa: F401,E402
import train  # noqa: F401,E402
from functions import get_default_hp  # noqa: F401,E402


VARIANT_FLAGS = {
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='Training root containing saved_models/.')
    parser.add_argument('--summary-dir', default=None, help='Directory for CSV/JSON summaries.')
    return parser.parse_args()


def load_saved(path):
    if torch.cuda.is_available():
        return torch.load(str(path), map_location='cpu')
    return torch.load(str(path), map_location=torch.device('cpu'))


def infer_variant(hp):
    jobname = hp.get('jobname', '')
    for variant in sorted(VARIANT_FLAGS.keys(), key=len, reverse=True):
        if variant in jobname:
            return variant
    for variant, flags in VARIANT_FLAGS.items():
        if all(hp.get(key, False) == value for key, value in flags.items()):
            if variant == 'baseline':
                if hp.get('prev_rew_to_pfc_vip', False) or hp.get('prev_rew_to_sr_vip', False):
                    continue
            return variant
    return 'unknown'


def infer_seed(hp):
    jobname = hp.get('jobname', '')
    match = re.search(r'seed(\d+)', jobname)
    if match:
        return int(match.group(1))
    return hp.get('torch_seed', '')


def tail_mean(values, n=5):
    if len(values) == 0:
        return ''
    arr = np.asarray(values[-n:], dtype=float)
    return float(np.nanmean(arr))


def summarize_model(path):
    saved = load_saved(path)
    hp = saved.get('hp', {})
    perf_list = saved.get('perf_list', [])
    perf_rule_list = saved.get('perf_rule_list', [])
    loss_list = saved.get('loss_list', [])
    test_perf_list = saved.get('test_perf_list', [])
    test_perf_rule_list = saved.get('test_perf_rule_list', [])
    test_loss_list = saved.get('test_loss_list', [])

    status = path.name.split('_', 1)[0] if '_' in path.name else path.name

    return {
        'file': path.name,
        'status': status,
        'jobname': hp.get('jobname', ''),
        'variant': infer_variant(hp),
        'seed': infer_seed(hp),
        'n_train_points': len(perf_list),
        'n_test_points': len(test_perf_list),
        'final_train_perf': float(perf_list[-1]) if len(perf_list) else '',
        'final_train_rule_perf': float(perf_rule_list[-1]) if len(perf_rule_list) else '',
        'final_loss': float(loss_list[-1]) if len(loss_list) else '',
        'mean_last5_train_perf': tail_mean(perf_list),
        'mean_last5_train_rule_perf': tail_mean(perf_rule_list),
        'mean_last5_loss': tail_mean(loss_list),
        'final_test_perf': float(test_perf_list[-1]) if len(test_perf_list) else '',
        'final_test_rule_perf': float(test_perf_rule_list[-1]) if len(test_perf_rule_list) else '',
        'final_test_loss': float(test_loss_list[-1]) if len(test_loss_list) else '',
        'mean_last5_test_perf': tail_mean(test_perf_list),
        'mean_last5_test_rule_perf': tail_mean(test_perf_rule_list),
        'mean_last5_test_loss': tail_mean(test_loss_list),
        'prev_rew_to_pfc_default': hp.get('prev_rew_to_pfc_default', True),
        'prev_rew_to_pfc_vip': hp.get('prev_rew_to_pfc_vip', False),
        'prev_rew_to_sr_vip': hp.get('prev_rew_to_sr_vip', False),
        'prev_rew_negative_only_to_vip': hp.get('prev_rew_negative_only_to_vip', False),
        'prev_rew_to_pfc_scale': hp.get('prev_rew_to_pfc_scale', 1.0),
        'prev_rew_to_vip_scale': hp.get('prev_rew_to_vip_scale', 1.0),
        'warm_start_vip_feedback_init': hp.get('warm_start_vip_feedback_init', 0.0),
        'give_prev_stim': hp.get('give_prev_stim', True),
        'give_prev_choice': hp.get('give_prev_choice', True),
        'give_prev_rew': hp.get('give_prev_rew', True),
        'prev_stim_mag': hp.get('prev_stim_mag', 1.0),
        'prev_choice_mag': hp.get('prev_choice_mag', 1.0),
        'prev_rew_mag': hp.get('prev_rew_mag', 1.0),
        'feedback_noise_mode': hp.get('feedback_noise_mode', 'none'),
        'feedback_noise_prob': hp.get('feedback_noise_prob', 0.0),
        'feedback_noise_flip_fraction': hp.get('feedback_noise_flip_fraction', 0.5),
        'block_len': hp.get('block_len', ''),
        'train_switch_mode': hp.get('train_switch_mode', 'fixed_n'),
        'test_switch_mode': hp.get('test_switch_mode', 'fixed_interval'),
        'stochastic_min_block_len': hp.get('stochastic_min_block_len', ''),
        'stochastic_max_block_len': hp.get('stochastic_max_block_len', ''),
        'stochastic_switch_prob': hp.get('stochastic_switch_prob', ''),
        'test_stochastic_min_block_len': hp.get('test_stochastic_min_block_len', ''),
        'test_stochastic_max_block_len': hp.get('test_stochastic_max_block_len', ''),
        'test_stochastic_switch_prob': hp.get('test_stochastic_switch_prob', ''),
        'freeze_backbone_mode': hp.get('freeze_backbone_mode', 'none'),
        'freeze_backbone_trainable_param_count': hp.get('freeze_backbone_trainable_param_count', ''),
        'freeze_backbone_frozen_param_count': hp.get('freeze_backbone_frozen_param_count', ''),
        'freeze_backbone_trainable_names': ';'.join(hp.get('freeze_backbone_trainable_names', [])),
    }


def write_csv(path, rows):
    fieldnames = [
        'file',
        'status',
        'jobname',
        'variant',
        'seed',
        'n_train_points',
        'n_test_points',
        'final_train_perf',
        'final_train_rule_perf',
        'final_loss',
        'mean_last5_train_perf',
        'mean_last5_train_rule_perf',
        'mean_last5_loss',
        'final_test_perf',
        'final_test_rule_perf',
        'final_test_loss',
        'mean_last5_test_perf',
        'mean_last5_test_rule_perf',
        'mean_last5_test_loss',
        'prev_rew_to_pfc_default',
        'prev_rew_to_pfc_vip',
        'prev_rew_to_sr_vip',
        'prev_rew_negative_only_to_vip',
        'prev_rew_to_pfc_scale',
        'prev_rew_to_vip_scale',
        'warm_start_vip_feedback_init',
        'give_prev_stim',
        'give_prev_choice',
        'give_prev_rew',
        'prev_stim_mag',
        'prev_choice_mag',
        'prev_rew_mag',
        'feedback_noise_mode',
        'feedback_noise_prob',
        'feedback_noise_flip_fraction',
        'block_len',
        'train_switch_mode',
        'test_switch_mode',
        'stochastic_min_block_len',
        'stochastic_max_block_len',
        'stochastic_switch_prob',
        'test_stochastic_min_block_len',
        'test_stochastic_max_block_len',
        'test_stochastic_switch_prob',
        'freeze_backbone_mode',
        'freeze_backbone_trainable_param_count',
        'freeze_backbone_frozen_param_count',
        'freeze_backbone_trainable_names',
    ]
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    aggregate = []
    for variant in sorted({row['variant'] for row in rows}):
        variant_rows = [row for row in rows if row['variant'] == variant]
        for status in sorted({row['status'] for row in variant_rows}):
            status_rows = [row for row in variant_rows if row['status'] == status]
            numeric = {}
            for key in ['mean_last5_test_perf', 'mean_last5_test_rule_perf', 'mean_last5_train_perf', 'mean_last5_train_rule_perf']:
                vals = [row[key] for row in status_rows if row[key] != '']
                numeric[key] = float(np.mean(vals)) if vals else ''
            aggregate.append({
                'variant': variant,
                'status': status,
                'n_runs': len(status_rows),
                **numeric,
            })
    return aggregate


def write_aggregate_csv(path, rows):
    fieldnames = [
        'variant',
        'status',
        'n_runs',
        'mean_last5_test_perf',
        'mean_last5_test_rule_perf',
        'mean_last5_train_perf',
        'mean_last5_train_rule_perf',
    ]
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    saved_models_dir = output_dir / 'saved_models'
    summary_dir = Path(args.summary_dir).resolve() if args.summary_dir else output_dir / 'summaries'
    summary_dir.mkdir(parents=True, exist_ok=True)

    if not saved_models_dir.exists():
        raise FileNotFoundError('saved_models directory not found: {}'.format(saved_models_dir))

    model_files = sorted(path for path in saved_models_dir.iterdir() if path.is_file())
    rows = [summarize_model(path) for path in model_files]
    aggregate = aggregate_rows(rows)

    write_csv(summary_dir / 'training_runs.csv', rows)
    write_aggregate_csv(summary_dir / 'training_by_variant.csv', aggregate)

    manifest = {
        'output_dir': str(output_dir),
        'saved_models_dir': str(saved_models_dir),
        'summary_dir': str(summary_dir),
        'n_models': len(rows),
        'outputs': ['training_runs.csv', 'training_by_variant.csv'],
    }
    with open(summary_dir / 'manifest.json', 'w') as handle:
        json.dump(manifest, handle, indent=2)

    print('summarized {} trained models into {}'.format(len(rows), summary_dir), flush=True)


if __name__ == '__main__':
    main()
