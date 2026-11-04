#!/usr/bin/env python3
"""Train one feedback-routing variant for the disinhibitory-motif project."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE_ROOT / '.mplconfig'))
sys.path.insert(0, str(REPO_ROOT))

from functions import get_default_hp  # noqa: E402
from train import train_bpxtrials_v2_working  # noqa: E402


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=sorted(VARIANTS.keys()))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', required=True, help='Training root on the GPU filesystem.')
    parser.add_argument('--job-prefix', default='feedback_gating')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--block-len', type=int, default=20)
    parser.add_argument('--n-switches', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--max-train-batches', type=int, default=100000)
    parser.add_argument('--max-runtime-hours', type=float, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--test-n-trials', type=int, default=200)
    parser.add_argument('--test-switch-every', type=int, default=20)
    parser.add_argument('--dry-run', action='store_true', help='Write config and exit without training.')
    return parser.parse_args()


def build_hp(args):
    hp, _, _ = get_default_hp()
    hp['task'] = 'wcst'
    hp['jobname'] = '{}_{}_seed{}'.format(args.job_prefix, args.variant, args.seed)
    hp['torch_seed'] = args.seed
    hp['batch_size'] = args.batch_size
    hp['block_len'] = args.block_len
    hp['n_switches'] = args.n_switches
    hp['dend_nonlinearity'] = 'subtractive'
    hp['training_root'] = str(Path(args.output_dir).resolve())
    hp['checkpoint_interval'] = args.checkpoint_interval
    hp['max_train_batches'] = args.max_train_batches
    hp['test_n_trials'] = args.test_n_trials
    hp['test_switch_every'] = args.test_switch_every
    hp['plot_during_training'] = False
    hp['save_model'] = True
    hp['save_figures'] = False
    hp.update(VARIANTS[args.variant])

    if args.learning_rate is not None:
        hp['learning_rate'] = args.learning_rate
    if args.max_runtime_hours is not None:
        hp['max_runtime_seconds'] = int(args.max_runtime_hours * 3600)

    return hp


def write_config(hp):
    config_dir = Path(hp['training_root']) / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / '{}.json'.format(hp['jobname'])
    with open(config_path, 'w') as handle:
        json.dump(hp, handle, indent=2, sort_keys=True)
    return config_path


def main():
    args = parse_args()
    hp = build_hp(args)
    config_path = write_config(hp)

    print('jobname: {}'.format(hp['jobname']), flush=True)
    print('variant: {}'.format(args.variant), flush=True)
    print('seed: {}'.format(args.seed), flush=True)
    print('training_root: {}'.format(hp['training_root']), flush=True)
    print('config: {}'.format(config_path), flush=True)
    print('cuda_available: {}'.format(torch.cuda.is_available()), flush=True)
    if torch.cuda.is_available():
        print('cuda_device: {}'.format(torch.cuda.get_device_name(0)), flush=True)

    if args.dry_run:
        print('dry run complete; no training launched', flush=True)
        return

    train_bpxtrials_v2_working(hp)


if __name__ == '__main__':
    main()
