#!/usr/bin/env python3
"""Smoke tests for noisy-feedback trial-history encoding."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE_ROOT / '.mplconfig'))
sys.path.insert(0, str(REPO_ROOT))

from functions import apply_feedback_noise, compute_trial_history  # noqa: E402


def reward_current(last_rew):
    current, _, _ = compute_trial_history(
        last_rew=last_rew,
        prev_stim=None,
        prev_choice=None,
        input_period=np.arange(0, 2),
        batch_size=2,
        n_steps=3,
        input_dim=16,
        stim_start=0,
        stim_end=10,
        dt=10,
        choice_dim=3,
    )
    return current[0]


def main():
    last_rew = torch.tensor([True, False])

    observed = apply_feedback_noise(last_rew, {'feedback_noise_mode': 'flip', 'feedback_noise_prob': 1.0})
    assert torch.equal(observed.cpu(), torch.tensor([False, True]))
    assert torch.equal(reward_current(observed), torch.tensor([[0., 1.], [1., 0.]]))

    observed = apply_feedback_noise(last_rew, {'feedback_noise_mode': 'dropout', 'feedback_noise_prob': 1.0})
    assert torch.equal(observed.cpu(), torch.zeros(2, 2))
    assert torch.equal(reward_current(observed), torch.zeros(2, 2))

    observed = apply_feedback_noise(last_rew, {
        'feedback_noise_mode': 'mixed',
        'feedback_noise_prob': 1.0,
        'feedback_noise_flip_fraction': 1.0,
    })
    assert torch.equal(observed.cpu(), torch.tensor([[0., 1.], [1., 0.]]))
    assert torch.equal(reward_current(observed), torch.tensor([[0., 1.], [1., 0.]]))

    observed = apply_feedback_noise(last_rew, {
        'feedback_noise_mode': 'mixed',
        'feedback_noise_prob': 1.0,
        'feedback_noise_flip_fraction': 0.0,
    })
    assert torch.equal(observed.cpu(), torch.zeros(2, 2))
    assert torch.equal(reward_current(observed), torch.zeros(2, 2))

    print('all noisy-feedback checks passed')


if __name__ == '__main__':
    main()
