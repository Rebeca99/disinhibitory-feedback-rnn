#!/usr/bin/env python3
"""Create a thesis schematic for the stochastic hidden-switch feedback task.

This figure is intentionally data-free: it explains the model extension and the
task logic before the quantitative results. It is designed to sit before the
core results figure in the thesis chapter.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        default=os.path.expanduser('~/biornn_feedback_stochastic_bottleneck_pfc01_seeds/summaries/thesis_figures'),
        help='Where to write schematic figure files.',
    )
    parser.add_argument('--prefix', default='figure_model_task_schematic')
    return parser.parse_args()


def draw_arrow(ax, start, end, color='#222222', lw=1.8, style='-|>', alpha=1.0, connectionstyle='arc3,rad=0.0'):
    ax.annotate(
        '',
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            alpha=alpha,
            shrinkA=4,
            shrinkB=4,
            mutation_scale=12,
            connectionstyle=connectionstyle,
        ),
    )


def draw_inhibitory(ax, start, end, color='#333333', lw=1.8, alpha=1.0):
    ax.annotate(
        '',
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle='-[,widthB=4.5,lengthB=0.8',
            color=color,
            lw=lw,
            alpha=alpha,
            shrinkA=4,
            shrinkB=0,
        ),
    )


def add_box(ax, xy, width, height, label, face, edge='#222222', fontsize=9, weight='regular', rounding=0.06):
    import matplotlib.patches as patches

    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f'round,pad=0.012,rounding_size={rounding}',
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, label, ha='center', va='center', fontsize=fontsize, weight=weight)
    return box


def add_population(ax, x, y, label, color):
    return add_box(ax, (x, y), 0.16, 0.075, label, color, fontsize=8, rounding=0.025)


def panel_title(ax, letter, title):
    ax.text(-0.03, 1.02, f'{letter}.', fontsize=12, weight='bold', va='bottom', ha='left')
    ax.text(0.04, 1.02, title, fontsize=12, weight='bold', va='bottom', ha='left')


def centered_title(ax, title):
    ax.text(0.5, 1.02, title, fontsize=12, weight='bold', va='bottom', ha='center', color='black')


def draw_module(ax, x, y, title, show_feedback=False):
    import matplotlib.patches as patches

    module = patches.FancyBboxPatch(
        (x, y),
        0.42,
        0.43,
        boxstyle='round,pad=0.02,rounding_size=0.035',
        facecolor='#f8fafc',
        edgecolor='#334155',
        linewidth=1.4,
    )
    ax.add_patch(module)
    ax.text(x + 0.21, y + 0.39, title, ha='center', va='center', fontsize=10, weight='bold')
    e = add_population(ax, x + 0.04, y + 0.25, 'E soma', '#dbeafe')
    dend = add_population(ax, x + 0.23, y + 0.25, 'E dend', '#bfdbfe')
    pv = add_population(ax, x + 0.04, y + 0.12, 'PV', '#fee2e2')
    sst = add_population(ax, x + 0.23, y + 0.12, 'SST', '#fecaca')
    vip = add_population(ax, x + 0.135, y + 0.015, 'VIP', '#fde68a')
    draw_arrow(ax, (x + 0.31, y + 0.25), (x + 0.18, y + 0.25), color='#2563eb', lw=1.5)
    draw_inhibitory(ax, (x + 0.31, y + 0.16), (x + 0.31, y + 0.25), color='#b91c1c', lw=1.5)
    draw_inhibitory(ax, (x + 0.21, y + 0.055), (x + 0.31, y + 0.12), color='#b45309', lw=1.5)
    draw_inhibitory(ax, (x + 0.12, y + 0.16), (x + 0.12, y + 0.25), color='#b91c1c', lw=1.2, alpha=0.7)
    if show_feedback:
        draw_arrow(ax, (x + 0.21, y + 0.47), (x + 0.21, y + 0.09), color='#b45309', lw=2.3, connectionstyle='arc3,rad=-0.15')
    return {'e': e, 'dend': dend, 'pv': pv, 'sst': sst, 'vip': vip}


def make_schematic(output_dir, prefix):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.8, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1.14, 1.0], hspace=0.42, wspace=0.30)
    ax_arch = fig.add_subplot(grid[0, 0])
    ax_task = fig.add_subplot(grid[0, 1])
    ax_variants = fig.add_subplot(grid[1, 0])
    ax_timeline = fig.add_subplot(grid[1, 1])
    for ax in [ax_arch, ax_task, ax_variants, ax_timeline]:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Architecture.
    centered_title(ax_arch, 'Biologically constrained feedback-gating model')
    draw_module(ax_arch, 0.04, 0.17, 'PFC / rule module')
    draw_module(ax_arch, 0.53, 0.17, 'Sensorimotor module')
    draw_arrow(ax_arch, (0.43, 0.42), (0.55, 0.42), color='#2563eb', lw=2.2)
    ax_arch.text(0.49, 0.48, 'top-down\nrule state', ha='center', va='bottom', fontsize=8, color='#1d4ed8')
    add_box(ax_arch, (0.36, 0.82), 0.28, 0.08, 'negative feedback', '#fee2e2', edge='#b91c1c', fontsize=9, weight='bold')
    draw_arrow(ax_arch, (0.43, 0.82), (0.19, 0.58), color='#991b1b', lw=1.7, connectionstyle='arc3,rad=0.12')
    draw_arrow(ax_arch, (0.50, 0.82), (0.66, 0.58), color='#991b1b', lw=1.7, connectionstyle='arc3,rad=-0.12')
    draw_arrow(ax_arch, (0.57, 0.82), (0.78, 0.22), color='#b45309', lw=2.2, connectionstyle='arc3,rad=-0.34')
    ax_arch.text(
        0.72,
        0.82,
        'added VIP-targeted\nerror route',
        ha='left',
        va='center',
        fontsize=8,
        color='#92400e',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.9),
    )
    ax_arch.text(
        0.50,
        0.08,
        'Error input activates VIP \u2192 suppresses SST \u2192 releases dendritic integration',
        fontsize=8,
        ha='center',
        color='black',
    )

    # Task logic.
    panel_title(ax_task, 'b', 'Stochastic hidden-switch WCST')
    block_y = 0.75
    blocks = [(0.04, 0.24, 'Color'), (0.28, 0.19, 'Shape'), (0.50, 0.26, 'Color'), (0.80, 0.16, 'Shape')]
    colors = {'Color': '#bfdbfe', 'Shape': '#ddd6fe'}
    for x, w, label in blocks:
        rect = patches.FancyBboxPatch((x, block_y), w, 0.12, boxstyle='round,pad=0.008,rounding_size=0.015', facecolor=colors[label], edgecolor='#334155')
        ax_task.add_patch(rect)
        ax_task.text(x + w / 2, block_y + 0.06, label, ha='center', va='center', fontsize=9, weight='bold')
    for x in [0.28, 0.50, 0.80]:
        ax_task.plot([x, x], [0.69, 0.91], color='#64748b', lw=1.2, ls='--')
        ax_task.text(x, 0.64, 'hidden\nswitch', ha='center', va='top', fontsize=7, color='#475569')
    ax_task.text(
        0.04,
        0.53,
        'Stochastic block lengths make switch timing uncertain; longer stochastic blocks are used during testing.',
        fontsize=8,
        wrap=True,
    )
    add_box(ax_task, (0.05, 0.29), 0.18, 0.10, 'previous\nchoice', '#f1f5f9', fontsize=8)
    add_box(ax_task, (0.27, 0.29), 0.18, 0.10, 'previous\nstimulus', '#f1f5f9', fontsize=8)
    add_box(ax_task, (0.49, 0.29), 0.18, 0.10, 'feedback\n(noisy)', '#fee2e2', fontsize=8)
    add_box(ax_task, (0.74, 0.29), 0.20, 0.10, 'infer current\nrule', '#dcfce7', fontsize=8, weight='bold')
    draw_arrow(ax_task, (0.23, 0.34), (0.27, 0.34), color='#64748b')
    draw_arrow(ax_task, (0.45, 0.34), (0.49, 0.34), color='#64748b')
    draw_arrow(ax_task, (0.67, 0.34), (0.74, 0.34), color='#991b1b')
    ax_task.text(
        0.05,
        0.10,
        'After an unannounced switch, negative feedback is the first reliable cue that the latent rule changed.',
        fontsize=8,
        wrap=True,
    )

    # Variants.
    panel_title(ax_variants, 'c', 'Feedback-route variants')
    headers = ['Variant', 'Direct PFC route', 'PFC VIP', 'SR VIP', 'Interpretation']
    xcols = [0.02, 0.22, 0.43, 0.58, 0.73]
    widths = [0.18, 0.18, 0.12, 0.12, 0.24]
    for x, w, h in zip(xcols, widths, headers):
        add_box(ax_variants, (x, 0.80), w, 0.09, h, '#e2e8f0', fontsize=7, weight='bold', rounding=0.012)
    rows = [
        ('Baseline', 'weak', '-', '-', 'abstract error input'),
        ('PFC-VIP', 'weak', 'error', '-', 'local PFC disinhibition'),
        ('SR-VIP', 'weak', '-', 'error', 'sensory dendritic gating'),
        ('PFC+SR VIP', 'weak', 'error', 'error', 'coordinated feedback gate'),
    ]
    for ridx, row in enumerate(rows):
        y = 0.66 - ridx * 0.15
        for x, w, value in zip(xcols, widths, row):
            face = '#fff7ed' if value == 'error' else '#f8fafc'
            add_box(ax_variants, (x, y), w, 0.09, value, face, fontsize=7, rounding=0.012)
    ax_variants.text(0.02, 0.05, 'All final stochastic-bottleneck variants retained a weakened direct PFC feedback route (scale 0.1) and used VIP feedback gain 5.0.', fontsize=8)

    # Trial timeline.
    panel_title(ax_timeline, 'd', 'Trial timing used for analysis windows')
    segments = [
        (0, 100, 'history\nfeedback', '#fee2e2'),
        (100, 1100, 'delay', '#f8fafc'),
        (1100, 1600, 'reference\ncard', '#dbeafe'),
        (1600, 2100, 'test cards\nresponse', '#dcfce7'),
    ]
    total = 2100
    x_start, x_end, y = 0.08, 0.96, 0.55
    for start, stop, label, color in segments:
        x = x_start + (start / total) * (x_end - x_start)
        w = ((stop - start) / total) * (x_end - x_start)
        rect = patches.Rectangle((x, y), w, 0.16, facecolor=color, edgecolor='#334155', linewidth=1.0)
        ax_timeline.add_patch(rect)
        if start == 0:
            ax_timeline.text(x + w / 2, y + 0.24, label, ha='center', va='bottom', fontsize=7)
            draw_arrow(ax_timeline, (x + w / 2, y + 0.22), (x + w / 2, y + 0.16), color='#991b1b', lw=1.1)
        else:
            ax_timeline.text(x + w / 2, y + 0.08, label, ha='center', va='center', fontsize=8)
        ax_timeline.text(x, y - 0.06, str(start), ha='center', va='top', fontsize=7)
    ax_timeline.text(x_end, y - 0.06, '2100 ms', ha='center', va='top', fontsize=7)
    ax_timeline.plot([x_start, x_end], [y - 0.02, y - 0.02], color='#334155', lw=1.2)
    for xpos, label in [(0.12, 'feedback'), (0.24, 'early post-feedback'), (0.46, 'pre-stim delay'), (0.82, 'response')]:
        ax_timeline.text(xpos, 0.24, label, ha='center', fontsize=7)
        ax_timeline.plot([xpos, xpos], [0.34, 0.50], color='#64748b', lw=1, ls=':')

    fig.suptitle('Error-driven disinhibitory feedback gating in a stochastic rule-updating task', fontsize=14, y=0.995, weight='bold')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'{prefix}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    make_schematic(Path(args.output_dir).expanduser(), args.prefix)
    print('wrote schematic to {}'.format(Path(args.output_dir).expanduser()))


if __name__ == '__main__':
    main()
