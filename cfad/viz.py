"""Publication-quality visualization utilities for CFAD."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from cfad import detect
from cfad.models.cgmy import CGMYCF
from cfad.models.gaussian import GaussianCF
from cfad.models.levy_stable import LevyStableCF
from cfad.models.nig import NIGCF
from cfad.utils import simulate_levy_returns

__all__ = [
    "plot_cf_families",
    "plot_contour_anatomy",
    "plot_score_distribution",
    "plot_detection_timeline",
    "plot_roc_from_simulations",
]


def _apply_axis_style(ax) -> None:
    """Apply common minimalist style for all figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.22, linestyle=":", linewidth=0.6)
    ax.tick_params(labelsize=9)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.title.set_size(10)


def _finalize_figure(fig, savepath: Optional[str] = None):
    """Finalize layout and optionally save figure."""
    fig.tight_layout()
    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, format="png", bbox_inches="tight")
    return fig


def plot_cf_families(
    xi_max: float = 8.0,
    n_xi: int = 500,
    savepath: Optional[str] = None,
    fig=None,
    ax=None,
):
    """
    Three-panel plot comparing CF families.

    Left: |phi(xi)| (modulus) for Gaussian, NIG, CGMY, LevyStable
    Centre: Re[phi(xi)]
    Right: Im[phi(xi)]

    Use default parameters for each model. Annotate is_analytic status.
    Label branch-cut onset for NIG with a vertical dashed line.
    This is Figure 1 of the paper.
    """
    import matplotlib.pyplot as plt

    xi = np.linspace(-xi_max, xi_max, n_xi, dtype=np.float64)
    models = [
        GaussianCF(),
        NIGCF(),
        CGMYCF(),
        LevyStableCF(),
    ]
    names = ["Gaussian", "NIG", "CGMY", "LevyStable"]
    analytic_note = ", ".join(
        [f"{n}: {'yes' if m.is_analytic else 'no'}" for n, m in zip(names, models)]
    )

    if ax is not None:
        if len(ax) != 3:
            raise ValueError("ax must contain exactly 3 axes")
        axes = ax
        fig = axes[0].figure if fig is None else fig
    else:
        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        else:
            axes = fig.subplots(1, 3)

    for model, name in zip(models, names):
        phi = model.cf(xi)
        axes[0].plot(xi, np.abs(phi), linewidth=1.5, label=name)
        axes[1].plot(xi, np.real(phi), linewidth=1.5, label=name)
        axes[2].plot(xi, np.imag(phi), linewidth=1.5, label=name)

    nig = NIGCF()
    branch_onset = nig.alpha - abs(nig.beta)
    axes[1].axvline(branch_onset, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].text(
        branch_onset,
        0.92,
        "NIG branch onset",
        rotation=90,
        va="top",
        ha="right",
        transform=axes[1].get_xaxis_transform(),
        fontsize=9,
    )

    axes[0].set_title("|phi(xi)|")
    axes[1].set_title("Re[phi(xi)]")
    axes[2].set_title("Im[phi(xi)]")
    for a in axes:
        a.set_xlabel("xi")
        _apply_axis_style(a)
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=9, frameon=False)

    fig.suptitle(f"Characteristic Function Families  |  is_analytic: {analytic_note}", fontsize=10)
    return _finalize_figure(fig, savepath=savepath)


def plot_contour_anatomy(
    model_name: Literal["gaussian", "nig"] = "nig",
    savepath: Optional[str] = None,
    fig=None,
    ax=None,
):
    """
    Illustrate the rectangular contour in the complex xi-plane.

    Draw:
    - The real axis (horizontal)
    - The rectangular contour C as a directed rectangle (arrows on edges)
    - For NIG: mark the branch point location with an X
    - For Gaussian: annotate "no poles — integral = 0"
    - For NIG: annotate "branch cut ↓ integral ≠ 0"

    Axis labels: Re(ξ), Im(ξ). This is Figure 2 of the paper.
    """
    import matplotlib.pyplot as plt

    if ax is not None:
        axis = ax
        fig = axis.figure if fig is None else fig
    else:
        if fig is None:
            fig, axis = plt.subplots(figsize=(6, 5))
        else:
            axis = fig.subplots(1, 1)

    xi_min, xi_max, h = -6.0, 6.0, 1.8
    corners = np.array(
        [
            [xi_min, -h],
            [xi_max, -h],
            [xi_max, h],
            [xi_min, h],
            [xi_min, -h],
        ],
        dtype=np.float64,
    )

    axis.axhline(0.0, color="0.3", linewidth=1.0)
    axis.plot(corners[:, 0], corners[:, 1], color="tab:blue", linewidth=1.6)
    axis.text(xi_max - 0.3, h + 0.12, "C", fontsize=10)

    for p0, p1 in zip(corners[:-1], corners[1:]):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        axis.arrow(
            p0[0],
            p0[1],
            0.35 * dx,
            0.35 * dy,
            head_width=0.16,
            head_length=0.28,
            fc="tab:blue",
            ec="tab:blue",
            length_includes_head=True,
            alpha=0.9,
        )

    if model_name == "gaussian":
        axis.text(-5.6, 2.35, "no poles - integral = 0", fontsize=10, color="tab:green")
        axis.set_title("Contour anatomy (Gaussian)")
    elif model_name == "nig":
        nig = NIGCF()
        branch_im = nig.alpha - abs(nig.beta)
        branch_im = float(np.clip(branch_im, -2.6, 2.6))
        axis.plot([0.0], [branch_im], marker="x", markersize=10, color="tab:red", mew=2)
        axis.text(0.15, branch_im + 0.12, "branch point", color="tab:red", fontsize=9)
        axis.text(-5.6, 2.35, "branch cut -> integral != 0", fontsize=10, color="tab:red")
        axis.set_title("Contour anatomy (NIG)")
    else:
        raise ValueError("model_name must be 'gaussian' or 'nig'")

    axis.set_xlabel("Re(xi)")
    axis.set_ylabel("Im(xi)")
    axis.set_xlim(-6.5, 6.5)
    axis.set_ylim(-3.0, 3.0)
    _apply_axis_style(axis)
    return _finalize_figure(fig, savepath=savepath)


def plot_score_distribution(
    report: "AnomalyReport",
    n_bins: int = 40,
    fit_normal: bool = True,
    savepath: Optional[str] = None,
    fig=None,
    ax=None,
):
    """
    Histogram of the raw score series with:
    - Fitted normal distribution overlay (if fit_normal=True)
    - Vertical line at the CUSUM calibration mean (mu0)
    - Vertical line at the alarm threshold level
    - Annotation: "x% of scores above threshold"
    """
    import matplotlib.pyplot as plt

    scores = np.asarray(report.scores, dtype=np.float64)
    if ax is not None:
        axis = ax
        fig = axis.figure if fig is None else fig
    else:
        if fig is None:
            fig, axis = plt.subplots(figsize=(7, 4.5))
        else:
            axis = fig.subplots(1, 1)

    axis.hist(scores, bins=n_bins, density=True, alpha=0.65, color="tab:blue", edgecolor="white")

    mu = float(np.mean(scores))
    sigma = float(np.std(scores, ddof=1)) + 1e-12
    if fit_normal:
        x_grid = np.linspace(np.min(scores), np.max(scores), 300)
        pdf = np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        axis.plot(x_grid, pdf, color="black", linewidth=1.4, label="Normal fit")

    score_threshold = float(report.mu0 + report.threshold * report.sigma0)
    axis.axvline(report.mu0, color="tab:green", linestyle="--", linewidth=1.2, label="mu0")
    axis.axvline(score_threshold, color="tab:red", linestyle="--", linewidth=1.2, label="threshold")

    pct_above = 100.0 * float(np.mean(scores > score_threshold))
    axis.text(
        0.98,
        0.95,
        f"{pct_above:.1f}% of scores above threshold",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    axis.set_title("Score distribution")
    axis.set_xlabel("Residue score")
    axis.set_ylabel("Density")
    axis.legend(frameon=False, fontsize=9)
    _apply_axis_style(axis)
    return _finalize_figure(fig, savepath=savepath)


def _resolve_event_x(event_value, report) -> float | object | None:
    """Resolve event location to x coordinate given report date/index mode."""
    if report.dates is not None and len(report.dates) > 0:
        import pandas as pd

        if isinstance(event_value, (int, np.integer)):
            idx = int(np.clip(event_value, 0, len(report.dates) - 1))
            return report.dates[idx]
        try:
            return pd.Timestamp(event_value)
        except Exception:
            return None

    if isinstance(event_value, (int, float, np.integer, np.floating)):
        return float(event_value)
    return None


def plot_detection_timeline(
    report: "AnomalyReport",
    returns: Optional[NDArray[np.float64]] = None,
    events: Optional[dict] = None,
    savepath: Optional[str] = None,
    fig=None,
    ax=None,
):
    """
    Full four-panel diagnostic timeline.

    Panel 1: Return series (if provided)
    Panel 2: Raw residue score with mu0 ± 3*sigma0 bands
    Panel 3: CUSUM S+ and S- with alarm threshold h
    Panel 4: Alarm flags (stem plot, one spike per alarm)

    events: dict mapping label strings to date/index positions
    e.g. {"COVID crash": pd.Timestamp("2020-03-16")}
    Each event gets a vertical annotation line across all panels.
    """
    import matplotlib.pyplot as plt

    n_scores = len(report.scores)
    valid_alarm_idx = report.alarm_indices[
        (report.alarm_indices >= 0) & (report.alarm_indices < n_scores)
    ]

    if report.dates is not None and len(report.dates) > 0 and len(report.window_end_indices) >= n_scores:
        score_idx = np.clip(report.window_end_indices[:n_scores] - 1, 0, len(report.dates) - 1)
        x_scores = report.dates[score_idx]
    else:
        x_scores = np.arange(n_scores, dtype=np.int64)

    if ax is not None:
        if len(ax) != 4:
            raise ValueError("ax must contain exactly 4 axes")
        axes = ax
        fig = axes[0].figure if fig is None else fig
    else:
        if fig is None:
            fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=False)
        else:
            axes = fig.subplots(4, 1)

    ax1, ax2, ax3, ax4 = axes

    if returns is not None:
        returns_arr = np.asarray(returns, dtype=np.float64)
        n_ret = len(returns_arr)
        if report.dates is not None and len(report.dates) >= n_ret:
            x_returns = report.dates[:n_ret]
        else:
            x_returns = np.arange(n_ret, dtype=np.int64)
        ax1.plot(x_returns, returns_arr, color="tab:blue", linewidth=1.0)
        ax1.set_ylabel("Returns")
    else:
        ax1.text(0.5, 0.5, "Returns not provided", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_ylabel("Returns")

    ax2.plot(x_scores, report.scores, color="tab:blue", linewidth=1.2, label="Score")
    ax2.axhline(report.mu0, color="tab:green", linestyle="--", linewidth=1.0, label="mu0")
    ax2.axhline(report.mu0 + 3 * report.sigma0, color="0.4", linestyle=":", linewidth=1.0)
    ax2.axhline(report.mu0 - 3 * report.sigma0, color="0.4", linestyle=":", linewidth=1.0)
    ax2.set_ylabel("Score")
    ax2.legend(frameon=False, fontsize=9)

    ax3.plot(x_scores, report.cusum_pos, color="tab:orange", linewidth=1.2, label="S+")
    ax3.plot(x_scores, report.cusum_neg, color="tab:purple", linewidth=1.2, label="S-")
    ax3.axhline(report.threshold, color="black", linestyle="--", linewidth=1.0, label="h")
    ax3.set_ylabel("CUSUM")
    ax3.legend(frameon=False, fontsize=9)

    alarm_x = x_scores[valid_alarm_idx] if len(valid_alarm_idx) else np.array([], dtype=float)
    if len(alarm_x):
        ax4.stem(alarm_x, np.ones(len(alarm_x)), linefmt="tab:red", markerfmt=" ", basefmt="k-")
    ax4.set_ylim(0.0, 1.2)
    ax4.set_ylabel("Alarm")
    ax4.set_xlabel("Date" if report.dates is not None else "Index")

    if events is not None:
        for i, (label, pos) in enumerate(events.items()):
            x_ev = _resolve_event_x(pos, report)
            if x_ev is None:
                continue
            for axis in axes:
                axis.axvline(x_ev, color="0.2", linestyle="--", linewidth=0.9, alpha=0.85)
            ax1.text(
                x_ev,
                0.98 - 0.07 * i,
                label,
                transform=ax1.get_xaxis_transform(),
                fontsize=9,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

    for a in axes:
        _apply_axis_style(a)
    fig.suptitle("Detection timeline", fontsize=10)
    return _finalize_figure(fig, savepath=savepath)


def plot_roc_from_simulations(
    n_sim: int = 100,
    T: int = 400,
    shift_at: int = 300,
    h_values: Optional[NDArray[np.float64]] = None,
    savepath: Optional[str] = None,
    fig=None,
    ax=None,
):
    """
    Self-contained ROC curve generator. Runs the simulation internally.
    Uses cfad.simulate for data generation.
    Computes AUC via numpy.trapezoid.
    Adds diagonal reference line and AUC in legend.
    This is Figure 3 of the paper.
    """
    import matplotlib.pyplot as plt

    if h_values is None:
        h_grid = np.linspace(1.0, 10.0, 30, dtype=np.float64)
    else:
        h_grid = np.asarray(h_values, dtype=np.float64)

    rng = np.random.default_rng(0)
    null_series = [rng.normal(0.0, 0.01, T).astype(np.float64) for _ in range(n_sim)]
    alt_series = []
    for i in range(n_sim):
        left = rng.normal(0.0, 0.01, shift_at)
        right = simulate_levy_returns(
            T - shift_at,
            alpha=1.5,
            scale=0.012,
            seed=10_000 + i,
        )
        alt_series.append(np.concatenate([left, right]).astype(np.float64))

    def _positive(series: NDArray[np.float64], h: float) -> bool:
        rep = detect(series, window=60, step=5, calibration_frac=0.4, h=float(h))
        if len(rep.alarm_indices) == 0:
            return False
        cutoff = max(0, len(rep.scores) - 60)
        return bool(np.any(rep.alarm_indices >= cutoff))

    fpr = np.zeros_like(h_grid, dtype=np.float64)
    tpr = np.zeros_like(h_grid, dtype=np.float64)
    for i, h in enumerate(h_grid):
        fpr[i] = np.mean([_positive(s, h) for s in null_series])
        tpr[i] = np.mean([_positive(s, h) for s in alt_series])

    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]
    auc = float(np.trapezoid(tpr_sorted, fpr_sorted))

    if ax is not None:
        axis = ax
        fig = axis.figure if fig is None else fig
    else:
        if fig is None:
            fig, axis = plt.subplots(figsize=(6.5, 5))
        else:
            axis = fig.subplots(1, 1)

    axis.plot(fpr_sorted, tpr_sorted, marker="o", linewidth=1.4, label=f"CFAD (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1.0)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC from simulations")
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    _apply_axis_style(axis)
    return _finalize_figure(fig, savepath=savepath)
