#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The paper's figure style, and one builder per figure in the paper.

The rcParams block below, together with `figsize`, is unchanged from the
plotting helper published with the PINNs code of Raissi et al.:

    Created on Mon Oct  9 20:11:57 2017
    @author: mraissi

That helper is what the paper's figures were rendered with, so it is kept
verbatim -- it is the source of the serif Computer Modern typography, the
10 pt base size and the 8 pt tick labels seen in every figure. Everything
from `use_paper_style` on is new: the palette, the grid/label helpers and the
figure builders that were previously copy-pasted into each experiment
notebook. (The helper's `newfig`/`savefig` are gone; no notebook ever called
them, and its `savefig` wrote PDF + EPS rather than the PNG the figures use.)

Importing this module applies the paper style globally, including
`text.usetex = True`, exactly as the original notebooks did by importing the
helper. **A working LaTeX installation is therefore required** (any TeX
distribution providing `latex` and `dvipng`: TeX Live, MiKTeX, MacTeX).
`mambampc/__init__.py` deliberately does not import this module, so
`import mambampc` stays free of the LaTeX dependency; only plotting needs it.

Colour roles are fixed across every figure: Mamba-MPC is purple, LSTM-MPC is
green, Transformer-MPC is blue, references are black. Some of the original
notebook cells were edited after the paper figures were rendered and disagree
with the published figures; where they do, the published figure is what is
reproduced here.
"""

import pathlib
import warnings

import numpy as np
import matplotlib as mpl

# mpl.use('pgf')


def figsize(scale, nplots=1):
    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = nplots * fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),  # default fig size of 0.9 textwidth
    "pgf.preamble": "\n".join(
        ["\\usepackage[utf8x]{inputenc}", "\\usepackage[T1]{fontenc}"]
    ),
}


def use_paper_style():
    """Apply the paper's matplotlib style. Idempotent; called once at import.

    Call it again if something else has since overwritten the rcParams (a
    seaborn import, `%matplotlib inline`, or another notebook cell).
    """
    mpl.rcParams.update(pgf_with_latex)


use_paper_style()

import matplotlib.pyplot as plt
from matplotlib import gridspec


# ---------------------------------------------------------------------------
# Palette and shared styling
# ---------------------------------------------------------------------------

C_REF = "k"
C_MAMBA = "tab:purple"
C_LSTM = "tab:green"
C_TRANSFORMER = "tab:blue"
C_INPUT = "tab:orange"

#: Colour per method. Every figure in the paper uses these roles.
METHOD_COLORS = {
    "Mamba-MPC": C_MAMBA,
    "LSTM-MPC": C_LSTM,
    "Transformer-MPC": C_TRANSFORMER,
}

#: The grid used on every axis in the paper.
GRID_KW = dict(which="both", linestyle="--", linewidth=0.8, color="gray", alpha=1)

#: Axis labels are set one size up from the rcParams default.
LABEL_FS = 14

#: Boxplot styling for the noise study (paper Fig. 11).
BOX_KW = dict(
    boxprops=dict(linestyle="-", linewidth=1.0, edgecolor="black"),
    medianprops=dict(linestyle="-", linewidth=1.0, color="firebrick"),
    whiskerprops=dict(linestyle="--", linewidth=1.0, color="gray"),
    capprops=dict(linestyle="-", linewidth=1.5, color="black"),
    flierprops=dict(
        marker="o",
        markerfacecolor="gray",
        markersize=5,
        linestyle="none",
        markeredgecolor="none",
    ),
)


def grid(ax):
    """Apply the paper's grid to one axis."""
    ax.grid(True, **GRID_KW)


def label(ax, ylabel=None, xlabel=None, labelpad=None):
    """Set axis labels at the paper's label size."""
    if ylabel is not None:
        kw = {} if labelpad is None else {"labelpad": labelpad}
        ax.set_ylabel(ylabel, fontsize=LABEL_FS, **kw)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS)


def save(fig, path, dpi=600, crop=True):
    """Write `fig` to `<path>.png` and `<path>.pdf`.

    PNG for GitHub and notebook previews, PDF as the vector version to include
    in LaTeX. `crop` trims the margins (`bbox_inches="tight"`); the noise
    study is the one paper figure saved uncropped, so its generous margins are
    part of how it looks. Returns the paths written.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"bbox_inches": "tight"} if crop else {}
    written = []
    for suffix in (".png", ".pdf"):
        out = path.with_name(path.name + suffix)
        fig.savefig(out, dpi=dpi, transparent=False, **extra)
        written.append(out)
    return written


def _draw_reference(ax, ref, n, color=C_REF, linestyle="-", lw=1.0):
    """Plot a reference trace, broadcasting a scalar over `n` samples."""
    ref = np.full(n, float(ref)) if np.ndim(ref) == 0 else np.asarray(ref)
    ax.plot(ref, linestyle, color=color, lw=lw, label="Reference")


def _finish(fig, save_to, dpi, crop=True):
    # GridSpec spacing plus tight_layout is the combination the paper's figures
    # were laid out with, so it is kept even though matplotlib considers the
    # two redundant and warns about it. The resulting geometry is the paper's,
    # checked panel by panel, so the warning is noise -- and it would otherwise
    # print the author's absolute paths into every notebook's saved output.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        plt.tight_layout()
    if save_to is not None:
        save(fig, save_to, dpi=dpi, crop=crop)
    return fig


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def comp_time_curve(record):
    """Mean solve time per horizon from a saved ``(N, times)`` object array.

    The stored records hold one row per run as ``[N, array_of_solve_times]``.
    The LSTM record repeats ``N = 5`` and ``N = 10`` across several aborted
    runs and is not in ascending order, so rows are grouped by ``N`` and
    averaged -- that is what makes the curve monotone, as in the paper.

    Returns ``(N_values, mean_times)``, both sorted ascending.
    """
    per_horizon = {}
    for horizon, times in record:
        per_horizon.setdefault(int(horizon), []).append(np.mean(times))
    horizons = np.array(sorted(per_horizon))
    means = np.array([np.mean(per_horizon[n]) for n in horizons])
    return horizons, means


def _tracking_metrics(y, u, ref):
    """Per-realisation ISE, IAE and input energy over the common prefix."""
    y, u, ref = np.asarray(y), np.asarray(u), np.ravel(np.asarray(ref))
    n = min(y.shape[-1], ref.shape[-1])
    err = y[..., :n] - ref[:n]
    return (
        np.sum(err**2, axis=-1),
        np.sum(np.abs(err), axis=-1),
        np.sum(np.asarray(u) ** 2, axis=-1),
    )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def fig_initial_conditions(y, u, ref=0.0, xlim=(0, 40), save_to=None, dpi=300):
    """Stabilisation from many initial conditions (paper Fig. 7).

    `y` and `u` are ``(n_runs, T)``. The trajectories are drawn without an
    explicit colour so they cycle through the default palette, as in the
    paper; the reference is laid on top as a heavy black dotted line.
    """
    y, u = np.asarray(y), np.asarray(u)
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(y.T, "-", lw=2.0, label="Mamba-MPC")
    _draw_reference(ax1, ref, y.shape[1], linestyle=":", lw=2.0)
    label(ax1, ylabel=r"$\mathbf{y}$")
    grid(ax1)
    ax1.set_xlim(*xlim)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(u.T, lw=2.0, label="Mamba-MPC")
    label(ax2, ylabel=r"$\mathbf{u}$", xlabel="Time steps")
    grid(ax2)
    ax2.set_xlim(*xlim)

    return _finish(fig, save_to, dpi)


def fig_tracking(
    ref,
    runs,
    input_order=None,
    xlim=(0, 385),
    legend=False,
    save_to=None,
    dpi=600,
):
    """Closed-loop reference tracking, output over input (paper Figs. 7-8, 10).

    `runs` maps a method name in `METHOD_COLORS` to ``(y, u)``. Entries are
    drawn in insertion order, so the last one lands on top -- the paper draws
    Mamba last. `input_order` overrides that order for the input panel only,
    which is what the paper's Fig. 8 does.
    """
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_reference(ax1, ref, len(np.ravel(np.asarray(ref))), linestyle="--")
    for name, (y, _) in runs.items():
        ax1.plot(np.asarray(y).squeeze(), "-", color=METHOD_COLORS[name], lw=1.5, label=name)
    label(ax1, ylabel=r"$\mathbf{y}$")
    grid(ax1)
    ax1.set_xlim(*xlim)
    if legend:
        ax1.legend(loc="lower right", fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    for name in input_order or runs:
        ax2.plot(
            np.asarray(runs[name][1]).squeeze(),
            "-",
            color=METHOD_COLORS[name],
            lw=1.5,
            label=name,
        )
    label(ax2, ylabel=r"$\mathbf{u}$", xlabel="Time steps")
    grid(ax2)
    ax2.set_xlim(*xlim)

    return _finish(fig, save_to, dpi)


def fig_computation_time(series, legend=False, save_to=None, dpi=600):
    """Mean online computation time against prediction horizon (paper Fig. 9).

    `series` maps a method name to ``(N_values, mean_times)`` -- see
    `comp_time_curve`.
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    for name, (horizons, times) in series.items():
        ax.plot(horizons, times, color=METHOD_COLORS[name], marker="o", label=name)
    label(ax, ylabel="Computation Time (s)", xlabel=r"$\mathbf{N}$")
    grid(ax)
    if legend:
        ax.legend(fontsize=8)
    return _finish(fig, save_to, dpi)


def fig_noise(ref, runs, box_labels=None, save_to=None, dpi=300):
    """Monte-Carlo noise study: mean +/- std bands over boxplots (paper Fig. 11).

    `runs` maps a method name to ``(Y, U)``, each ``(n_realisations, T)``. The
    stored arrays do not all share a length, so every series is plotted
    against its own sample index and the metrics use the common prefix.
    """
    fig = plt.figure(figsize=(4, 4), dpi=300)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2])
    _draw_reference(ax1, ref, len(np.ravel(np.asarray(ref))), linestyle="--")
    for name, (y, u) in runs.items():
        color = METHOD_COLORS[name]
        for ax, data in ((ax1, np.asarray(y)), (ax2, np.asarray(u))):
            mean, std = data.mean(axis=0), data.std(axis=0)
            steps = np.arange(len(mean))
            ax.plot(steps, mean, color=color, lw=1.0, label=name)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.4)
    label(ax1, ylabel=r"$\mathbf{y}$", labelpad=-5)
    label(ax2, ylabel=r"$\mathbf{u}$", labelpad=-8)
    grid(ax1)
    grid(ax2)

    metrics = {name: _tracking_metrics(y, u, ref) for name, (y, u) in runs.items()}
    titles = ["ISE", "IAE", "Input Energy"]
    colors = [METHOD_COLORS[name] for name in runs]
    if box_labels is None:
        box_labels = [name.replace("-MPC", "") for name in runs]

    for i, title in enumerate(titles):
        ax = fig.add_subplot(gs[1, i])
        bp = ax.boxplot(
            [metrics[name][i] for name in runs],
            patch_artist=True,
            tick_labels=box_labels,
            **BOX_KW,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_title(title, fontsize=LABEL_FS, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Value")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="gray", alpha=1)
        ax.set_axisbelow(True)

    return _finish(fig, save_to, dpi, crop=False)


def fig_four_tank(ref, runs, legend=False, save_to=None, dpi=600):
    """Four Tank MIMO reference tracking (paper Fig. 12).

    `ref` is ``(T, 4)``; `runs` maps a method name to ``(X (T, 4), U (T, 2))``.
    Entries are drawn in insertion order, the last on top.
    """
    ref = np.asarray(ref)
    fig = plt.figure(figsize=(4, 4), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    cells = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    labels = [
        r"$\mathbf{x_1}$",
        r"$\mathbf{x_2}$",
        r"$\mathbf{x_3}$",
        r"$\mathbf{x_4}$",
        r"$\mathbf{u_1}$",
        r"$\mathbf{u_2}$",
    ]

    for i, (cell, ylabel) in enumerate(zip(cells, labels)):
        ax = fig.add_subplot(gs[cell])
        if i < 4:
            ax.plot(ref[:, i], "-", color=C_REF, lw=1.0, label="Reference")
        for name, (states, inputs) in runs.items():
            data = np.asarray(states)[:, i] if i < 4 else np.asarray(inputs)[:, i - 4]
            ax.plot(data, "-", color=METHOD_COLORS[name], lw=1.0, label=name)
        label(ax, ylabel=ylabel)
        grid(ax)
        if legend and i == 0:
            ax.legend(loc="lower right", fontsize=8)

    return _finish(fig, save_to, dpi)


def fig_aero2(ref, states, inputs, save_to=None, dpi=600):
    """Quanser Aero2 closed-loop reference tracking (paper Fig. 14).

    `ref` and `states` are ``(T, 3)``, `inputs` is ``(T, 2)``. States track a
    dotted black reference; the two motor voltages are drawn in orange.
    """
    ref, states, inputs = np.asarray(ref), np.asarray(states), np.asarray(inputs)
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.3, wspace=0.25)
    labels = [
        r"$\mathbf{x_1}$",
        r"$\mathbf{x_2}$",
        r"$\mathbf{x_3}$",
        r"$\mathbf{u_1}$",
        r"$\mathbf{u_2}$",
    ]

    for i, ylabel in enumerate(labels):
        ax = fig.add_subplot(gs[i, 0])
        if i < 3:
            ax.plot(ref[:, i], "k:", lw=2.0, label="Reference")
            ax.plot(states[:, i], "-", color=C_MAMBA, lw=2.0, label="Mamba-MPC")
        else:
            ax.plot(inputs[:, i - 3], "-", color=C_INPUT, lw=2.0, label="Mamba-MPC")
        label(ax, ylabel=ylabel, xlabel="Time steps" if i == len(labels) - 1 else None)
        grid(ax)

    return _finish(fig, save_to, dpi)
