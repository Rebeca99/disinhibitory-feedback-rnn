#!/usr/bin/env python3
"""Fine-tune one feedback-routing variant on the noisy-feedback WCST task."""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument('--pretrained-path', required=True, help='Path to one pretrained Liu/Wang checkpoint file.')
    parser.add_argument('--output-dir', required=True, help='Training root on the GPU filesystem.')
    parser.add_argument('--job-prefix', default='feedback_noisy_task_finetune')
    parser.add_argument(
        '--short-job-name',
        action='store_true',
        help=(
            'Use a compact job name containing only prefix, variant and seed. '
            'Useful on filesystems that truncate very long checkpoint/model names.'
        ),
    )
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-train-batches', type=int, default=2000)
    parser.add_argument('--max-runtime-hours', type=float, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--test-n-trials', type=int, default=200)
    parser.add_argument('--test-switch-every', type=int, default=20)
    parser.add_argument('--block-len', type=int, default=None, help='Training block length; defaults to checkpoint/default hp.')
    parser.add_argument('--train-switch-mode', default='fixed_n', choices=['fixed_n', 'stochastic_hazard'])
    parser.add_argument('--test-switch-mode', default='fixed_interval', choices=['fixed_interval', 'stochastic_hazard'])
    parser.add_argument('--stochastic-min-block-len', type=int, default=10)
    parser.add_argument('--stochastic-max-block-len', type=int, default=35)
    parser.add_argument('--stochastic-switch-prob', type=float, default=0.08)
    parser.add_argument('--test-stochastic-min-block-len', type=int, default=None)
    parser.add_argument('--test-stochastic-max-block-len', type=int, default=None)
    parser.add_argument('--test-stochastic-switch-prob', type=float, default=None)
    parser.add_argument('--feedback-noise-mode', default='mixed', choices=['none', 'flip', 'dropout', 'mixed'])
    parser.add_argument('--feedback-noise-prob', type=float, default=0.2)
    parser.add_argument('--feedback-noise-flip-fraction', type=float, default=0.5)
    parser.add_argument('--pfc-feedback-scale', type=float, default=1.0, help='Multiplier on the original direct PFC previous-reward route.')
    parser.add_argument('--vip-feedback-scale', type=float, default=1.0, help='Multiplier on previous-reward routes targeting VIP cells.')
    parser.add_argument('--vip-feedback-init', type=float, default=0.0, help='Initial value for newly opened VIP feedback weights after warm start.')
    parser.add_argument('--prev-stim-mag', type=float, default=1.0, help='Magnitude multiplier for previous stimulus history input.')
    parser.add_argument('--prev-choice-mag', type=float, default=1.0, help='Magnitude multiplier for previous choice history input.')
    parser.add_argument('--prev-rew-mag', type=float, default=1.0, help='Magnitude multiplier for previous feedback/reward history input.')
    parser.add_argument(
        '--freeze-backbone-mode',
        default='none',
        choices=['none', 'feedback_only', 'readout_only', 'feedback_readout', 'feedback_history_readout'],
        help=(
            'Frozen-backbone robustness mode. "feedback_readout" freezes recurrent/sensory backbone '
            'and trains only previous-reward feedback weights plus readouts.'
        ),
    )
    parser.add_argument('--no-prev-stim', action='store_true', help='Remove previous stimulus history input.')
    parser.add_argument('--no-prev-choice', action='store_true', help='Remove previous choice history input.')
    parser.add_argument('--no-prev-rew', action='store_true', help='Remove previous feedback/reward history input.')
    parser.add_argument('--curriculum', action='store_true', help='Use the original trial-history curriculum during fine-tuning.')
    parser.add_argument('--dry-run', action='store_true', help='Write config and exit without training.')
    return parser.parse_args()


def load_pretrained_hp(pretrained_path):
    saved = torch.load(str(pretrained_path), map_location='cpu')
    hp = copy.deepcopy(saved['hp'])
    hp_default, _, _ = get_default_hp()
    for key, value in hp_default.items():
        hp.setdefault(key, value)
    return hp


def build_hp(args):
    pretrained_path = Path(args.pretrained_path).resolve()
    hp = load_pretrained_hp(pretrained_path)
    noise_tag = '{}_p{:03d}'.format(args.feedback_noise_mode, int(round(args.feedback_noise_prob * 1000)))
    route_tag = 'pfc{:03d}_vipg{:03d}_vipi{:03d}'.format(
        int(round(args.pfc_feedback_scale * 100)),
        int(round(args.vip_feedback_scale * 100)),
        int(round(args.vip_feedback_init * 1000)),
    )
    history_tag = 'hist_s{:03d}_c{:03d}_r{:03d}'.format(
        int(round(args.prev_stim_mag * 100)),
        int(round(args.prev_choice_mag * 100)),
        int(round(args.prev_rew_mag * 100)),
    )
    if args.train_switch_mode == 'stochastic_hazard':
        switch_tag = 'stoch_min{:02d}_max{:02d}_p{:03d}'.format(
            args.stochastic_min_block_len,
            args.stochastic_max_block_len,
            int(round(args.stochastic_switch_prob * 1000)),
        )
    else:
        switch_tag = 'fixed'
    hp['task'] = 'wcst'
    freeze_tag = ''
    if args.freeze_backbone_mode != 'none':
        freeze_tag = '_frozen{}'.format(args.freeze_backbone_mode)
    if args.short_job_name:
        hp['jobname'] = '{}_{}_seed{}'.format(args.job_prefix, args.variant, args.seed)
    else:
        hp['jobname'] = '{}{}_{}_{}_{}_{}_{}_seed{}'.format(
            args.job_prefix,
            freeze_tag,
            args.variant,
            noise_tag,
            route_tag,
            history_tag,
            switch_tag,
            args.seed,
        )
    hp['torch_seed'] = args.seed
    hp['warm_start_model_path'] = str(pretrained_path)
    hp['training_root'] = str(Path(args.output_dir).resolve())
    hp['learning_rate'] = args.learning_rate
    hp['checkpoint_interval'] = args.checkpoint_interval
    hp['max_train_batches'] = args.max_train_batches
    hp['test_n_trials'] = args.test_n_trials
    hp['test_switch_every'] = args.test_switch_every
    if args.block_len is not None:
        hp['block_len'] = args.block_len
    hp['train_switch_mode'] = args.train_switch_mode
    hp['test_switch_mode'] = args.test_switch_mode
    hp['stochastic_min_block_len'] = args.stochastic_min_block_len
    hp['stochastic_max_block_len'] = args.stochastic_max_block_len
    hp['stochastic_switch_prob'] = args.stochastic_switch_prob
    hp['test_stochastic_min_block_len'] = (
        args.test_stochastic_min_block_len
        if args.test_stochastic_min_block_len is not None
        else args.stochastic_min_block_len
    )
    hp['test_stochastic_max_block_len'] = (
        args.test_stochastic_max_block_len
        if args.test_stochastic_max_block_len is not None
        else args.stochastic_max_block_len
    )
    hp['test_stochastic_switch_prob'] = (
        args.test_stochastic_switch_prob
        if args.test_stochastic_switch_prob is not None
        else args.stochastic_switch_prob
    )
    hp['plot_during_training'] = False
    hp['save_model'] = True
    hp['save_figures'] = False
    hp['curriculum'] = args.curriculum
    hp['give_prev_stim'] = not args.no_prev_stim
    hp['give_prev_choice'] = not args.no_prev_choice
    hp['give_prev_rew'] = not args.no_prev_rew
    hp['prev_stim_mag'] = args.prev_stim_mag
    hp['prev_choice_mag'] = args.prev_choice_mag
    hp['prev_rew_mag'] = args.prev_rew_mag
    hp['freeze_backbone_mode'] = args.freeze_backbone_mode
    hp['zero_warm_start_vip_feedback'] = True
    hp['warm_start_vip_feedback_init'] = args.vip_feedback_init
    hp['prev_rew_to_pfc_scale'] = args.pfc_feedback_scale
    hp['prev_rew_to_vip_scale'] = args.vip_feedback_scale
    hp['feedback_noise_mode'] = args.feedback_noise_mode
    hp['feedback_noise_prob'] = args.feedback_noise_prob
    hp['feedback_noise_flip_fraction'] = args.feedback_noise_flip_fraction
    hp.update(VARIANTS[args.variant])
    hp['allow_zero_prev_rew_targets'] = (
        hp.get('prev_rew_to_pfc_default', True)
        and float(hp.get('prev_rew_to_pfc_scale', 1.0)) == 0.0
        and not hp.get('prev_rew_to_pfc_vip', False)
        and not hp.get('prev_rew_to_sr_vip', False)
    )

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
    print('pretrained_path: {}'.format(hp['warm_start_model_path']), flush=True)
    print('training_root: {}'.format(hp['training_root']), flush=True)
    print('learning_rate: {}'.format(hp['learning_rate']), flush=True)
    print('max_train_batches: {}'.format(hp['max_train_batches']), flush=True)
    print('block_len: {}'.format(hp['block_len']), flush=True)
    print('train_switch_mode: {}'.format(hp['train_switch_mode']), flush=True)
    print('test_switch_mode: {}'.format(hp['test_switch_mode']), flush=True)
    print('stochastic_switch: min={}, max={}, p={}'.format(
        hp['stochastic_min_block_len'],
        hp['stochastic_max_block_len'],
        hp['stochastic_switch_prob'],
    ), flush=True)
    print('test_stochastic_switch: min={}, max={}, p={}'.format(
        hp['test_stochastic_min_block_len'],
        hp['test_stochastic_max_block_len'],
        hp['test_stochastic_switch_prob'],
    ), flush=True)
    print('feedback_noise_mode: {}'.format(hp['feedback_noise_mode']), flush=True)
    print('feedback_noise_prob: {}'.format(hp['feedback_noise_prob']), flush=True)
    print('feedback_noise_flip_fraction: {}'.format(hp['feedback_noise_flip_fraction']), flush=True)
    print('pfc_feedback_scale: {}'.format(hp['prev_rew_to_pfc_scale']), flush=True)
    print('vip_feedback_scale: {}'.format(hp['prev_rew_to_vip_scale']), flush=True)
    print('vip_feedback_init: {}'.format(hp['warm_start_vip_feedback_init']), flush=True)
    print('history: give_prev_stim={} mag={}, give_prev_choice={} mag={}, give_prev_rew={} mag={}'.format(
        hp['give_prev_stim'],
        hp['prev_stim_mag'],
        hp['give_prev_choice'],
        hp['prev_choice_mag'],
        hp['give_prev_rew'],
        hp['prev_rew_mag'],
    ), flush=True)
    print('freeze_backbone_mode: {}'.format(hp['freeze_backbone_mode']), flush=True)
    print('allow_zero_prev_rew_targets: {}'.format(hp['allow_zero_prev_rew_targets']), flush=True)
    print('config: {}'.format(config_path), flush=True)
    print('cuda_available: {}'.format(torch.cuda.is_available()), flush=True)
    if torch.cuda.is_available():
        print('cuda_device: {}'.format(torch.cuda.get_device_name(0)), flush=True)

    if args.dry_run:
        print('dry run complete; no noisy-task fine-tuning launched', flush=True)
        return

    train_bpxtrials_v2_working(hp)


if __name__ == '__main__':
    main()
