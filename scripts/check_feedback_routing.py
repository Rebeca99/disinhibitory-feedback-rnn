#!/usr/bin/env python3
"""Smoke tests for previous-reward feedback routing masks."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE_ROOT / '.mplconfig'))
sys.path.insert(0, str(REPO_ROOT))

from functions import get_default_hp, load_model_v2  # noqa: E402
from model_working import Net_readoutSR_working  # noqa: E402


GROUPS = ['sr_edend', 'sr_pv', 'sr_sst', 'sr_vip', 'pfc_edend', 'pfc_pv', 'pfc_sst', 'pfc_vip']


def make_hp(**overrides):
    hp, _, _ = get_default_hp()
    hp['task'] = 'wcst'
    hp['save_model'] = False
    hp['save_figures'] = False
    hp['plot_during_training'] = False
    hp.update(overrides)
    return hp


def make_model(**overrides):
    hp = make_hp(**overrides)
    return Net_readoutSR_working(hp)


def counts_by_group(rnn, mask_row):
    return {
        group: int(mask_row[rnn.cg_idx[group]].sum().item())
        for group in GROUPS
    }


def assert_counts(label, actual, expected):
    for key, value in expected.items():
        if actual[key] != value:
            raise AssertionError('{} expected {}={} but found {}'.format(label, key, value, actual[key]))


def assert_history_unchanged(rnn):
    stim_counts = counts_by_group(rnn, rnn.mask_prev_stim[0])
    choice_counts = counts_by_group(rnn, rnn.mask_prev_choice[0])
    expected = {'pfc_edend': 140, 'pfc_pv': 10, 'pfc_vip': 0, 'sr_vip': 0}
    assert_counts('previous stimulus routing', stim_counts, expected)
    assert_counts('previous choice routing', choice_counts, expected)


def check_default_routing():
    model = make_model()
    rnn = model.rnn
    expected = {'pfc_edend': 140, 'pfc_pv': 10, 'pfc_vip': 0, 'sr_vip': 0}
    for row in [0, 1]:
        actual = counts_by_group(rnn, rnn.mask_rew[row])
        assert_counts('default reward row {}'.format(row), actual, expected)
        print('default reward row {}: {}'.format(row, actual))
    assert_history_unchanged(rnn)


def check_pfc_vip_routing():
    model = make_model(prev_rew_to_pfc_vip=True)
    rnn = model.rnn
    expected = {'pfc_edend': 140, 'pfc_pv': 10, 'pfc_vip': 10, 'sr_vip': 0}
    for row in [0, 1]:
        actual = counts_by_group(rnn, rnn.mask_rew[row])
        assert_counts('PFC VIP reward row {}'.format(row), actual, expected)
        print('PFC VIP reward row {}: {}'.format(row, actual))
    assert_history_unchanged(rnn)


def check_negative_only_sr_vip_routing():
    model = make_model(prev_rew_to_sr_vip=True, prev_rew_negative_only_to_vip=True)
    rnn = model.rnn
    correct_counts = counts_by_group(rnn, rnn.mask_rew[0])
    error_counts = counts_by_group(rnn, rnn.mask_rew[1])
    assert_counts(
        'negative-only SR VIP correct row',
        correct_counts,
        {'pfc_edend': 140, 'pfc_pv': 10, 'pfc_vip': 0, 'sr_vip': 0},
    )
    assert_counts(
        'negative-only SR VIP error row',
        error_counts,
        {'pfc_edend': 140, 'pfc_pv': 10, 'pfc_vip': 0, 'sr_vip': 10},
    )
    print('negative-only SR VIP correct row: {}'.format(correct_counts))
    print('negative-only SR VIP error row: {}'.format(error_counts))
    assert_history_unchanged(rnn)


def check_vip_only_error_routing():
    model = make_model(
        prev_rew_to_pfc_default=False,
        prev_rew_to_pfc_vip=True,
        prev_rew_to_sr_vip=True,
        prev_rew_negative_only_to_vip=True,
    )
    rnn = model.rnn
    correct_counts = counts_by_group(rnn, rnn.mask_rew[0])
    error_counts = counts_by_group(rnn, rnn.mask_rew[1])
    assert_counts(
        'VIP-only correct row',
        correct_counts,
        {'pfc_edend': 0, 'pfc_pv': 0, 'pfc_vip': 0, 'sr_vip': 0},
    )
    assert_counts(
        'VIP-only error row',
        error_counts,
        {'pfc_edend': 0, 'pfc_pv': 0, 'pfc_vip': 10, 'sr_vip': 10},
    )
    print('VIP-only correct row: {}'.format(correct_counts))
    print('VIP-only error row: {}'.format(error_counts))
    assert_history_unchanged(rnn)


def check_feedback_scale_routing():
    model = make_model(
        prev_rew_to_pfc_vip=True,
        prev_rew_to_sr_vip=True,
        prev_rew_negative_only_to_vip=True,
        prev_rew_to_pfc_scale=0.5,
        prev_rew_to_vip_scale=2.0,
    )
    rnn = model.rnn
    correct_counts = counts_by_group(rnn, rnn.mask_rew[0])
    error_counts = counts_by_group(rnn, rnn.mask_rew[1])
    assert_counts(
        'scaled correct row',
        correct_counts,
        {'pfc_edend': 70, 'pfc_pv': 5, 'pfc_vip': 0, 'sr_vip': 0},
    )
    assert_counts(
        'scaled error row',
        error_counts,
        {'pfc_edend': 70, 'pfc_pv': 5, 'pfc_vip': 20, 'sr_vip': 20},
    )
    print('scaled correct row: {}'.format(correct_counts))
    print('scaled error row: {}'.format(error_counts))
    assert_history_unchanged(rnn)


def check_legacy_fdbk_to_vip():
    model = make_model(fdbk_to_vip=True)
    rnn = model.rnn
    expected = {'pfc_edend': 0, 'pfc_pv': 0, 'pfc_vip': 10, 'sr_vip': 0}
    for row in [0, 1]:
        actual = counts_by_group(rnn, rnn.mask_rew[row])
        assert_counts('legacy fdbk_to_vip reward row {}'.format(row), actual, expected)
        print('legacy fdbk_to_vip reward row {}: {}'.format(row, actual))


def check_first_downloaded_checkpoint():
    trained_dir = WORKSPACE_ROOT / 'data' / 'liu_wang_2024' / 'trained_networks'
    if not trained_dir.exists():
        print('checkpoint smoke test skipped: {} not found'.format(trained_dir))
        return
    checkpoint = sorted(path for path in trained_dir.iterdir() if path.is_file())[0]
    model, hp, _, _, _ = load_model_v2(str(checkpoint), checkpoint.name, simple=False, plot=False, toprint=False)
    if hp['fdbk_to_vip'] is not False:
        raise AssertionError('expected downloaded checkpoint to preserve fdbk_to_vip=False')
    rnn = model.rnn
    for row in [0, 1]:
        actual = counts_by_group(rnn, rnn.mask_rew[row])
        assert_counts(
            'downloaded checkpoint reward row {}'.format(row),
            actual,
            {'pfc_edend': 140, 'pfc_pv': 10, 'pfc_vip': 0, 'sr_vip': 0},
        )
    print('downloaded checkpoint loaded: {}'.format(checkpoint.name))


def main():
    check_default_routing()
    check_pfc_vip_routing()
    check_negative_only_sr_vip_routing()
    check_legacy_fdbk_to_vip()
    check_vip_only_error_routing()
    check_feedback_scale_routing()
    check_first_downloaded_checkpoint()
    print('all feedback-routing checks passed')


if __name__ == '__main__':
    main()
