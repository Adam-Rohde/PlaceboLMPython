"""
Visualization functions for PlaceboLM results.

This module provides functions for creating contour plots, line plots,
and other visualizations of sensitivity analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import Optional, List, Tuple, Dict, Union
import warnings

from .core import (
    PlaceboLMResult,
    bias_adjustment,
    bootstrap_estimates,
    get_estimate_at_params,
    suppress_print,
    placebo_lm,
)
from .tables import _bootstrap_estimates_quiet


def placebo_lm_contour_plot(
    plm: PlaceboLMResult,
    gran: int = 100,
    k_range: Optional[Tuple[float, float]] = None,
    coef_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    xlabel: str = "k (confounding ratio)",
    ylabel: str = "Direct effect (coef_P_D_given_XZ)",
    cmap: str = "RdBu_r",
    levels: int = 20,
    add_colorbar: bool = True,
    add_contour_lines: bool = True,
    add_zero_contour: bool = True,
    add_benchmarks: bool = True,
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Create a contour plot of bias-adjusted estimates over parameter space.

    This visualization shows how the estimated treatment effect varies
    as assumptions about confounding (k) and placebo imperfection
    (coef_P_D_given_XZ) change.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    gran : int, optional
        Number of grid points in each dimension (default 100).
    k_range : Tuple[float, float], optional
        Range for k axis. If None, uses plm.partialIDparam_minmax["k"].
    coef_range : Tuple[float, float], optional
        Range for coef axis. If None, uses plm.partialIDparam_minmax["coef_P_D_given_XZ"].
    title : str, optional
        Plot title. If None, uses "Bias-Adjusted Treatment Effect".
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    cmap : str, optional
        Colormap name (default "RdBu_r").
    levels : int, optional
        Number of contour levels (default 20).
    add_colorbar : bool, optional
        Whether to add a colorbar (default True).
    add_contour_lines : bool, optional
        Whether to add labeled contour lines (default True).
    add_zero_contour : bool, optional
        Whether to highlight the zero effect contour (default True).
    add_benchmarks : bool, optional
        Whether to add benchmark points (k=0, k=1, DID) (default True).
    figsize : Tuple[float, float], optional
        Figure size (default (10, 8)).
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> result = placebo_lm(data=df, outcome="re78", treatment="treat",
    ...                      placebo_outcome="re74", observed_covariates=covs)
    >>> fig = placebo_lm_contour_plot(result, gran=50)
    >>> plt.show()
    """
    # Get parameter ranges
    if k_range is None:
        k_range = tuple(plm.partialIDparam_minmax.get("k", [-2, 2]))
    if coef_range is None:
        coef_range = tuple(plm.partialIDparam_minmax.get("coef_P_D_given_XZ", [-15000, 15000]))

    # Create grid
    k_vals = np.linspace(k_range[0], k_range[1], gran)
    coef_vals = np.linspace(coef_range[0], coef_range[1], gran)
    K, C = np.meshgrid(k_vals, coef_vals)

    # Compute estimates on grid
    Z = np.zeros_like(K)
    for i in range(gran):
        for j in range(gran):
            Z[i, j] = bias_adjustment(plm, K[i, j], C[i, j])

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine color limits (symmetric around zero if reasonable)
    vmax = max(abs(Z.min()), abs(Z.max()))
    vmin = -vmax

    # Create filled contour plot
    cf = ax.contourf(K, C, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add contour lines
    if add_contour_lines:
        cs = ax.contour(K, C, Z, levels=levels, colors='k', linewidths=0.5, alpha=0.3)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    # Highlight zero contour
    if add_zero_contour:
        cs_zero = ax.contour(K, C, Z, levels=[0], colors='black', linewidths=2)
        ax.clabel(cs_zero, inline=True, fontsize=10, fmt='%.0f')

    # Add benchmark points
    if add_benchmarks:
        # No confounding (k=0, coef=0)
        ax.scatter([0], [0], s=100, c='green', marker='o', edgecolors='black',
                   linewidths=2, label='No Confounding (k=0)', zorder=5)

        # Perfect placebo (k=1, coef=0)
        ax.scatter([1], [0], s=100, c='blue', marker='s', edgecolors='black',
                   linewidths=2, label='Perfect Placebo (k=1)', zorder=5)

        # DID equivalent if applicable
        if plm.beta_N_D_given_X != 0:
            m_did = plm.beta_Y_D_given_X / plm.beta_N_D_given_X
            if k_range[0] <= m_did <= k_range[1]:
                ax.scatter([m_did], [0], s=100, c='red', marker='^', edgecolors='black',
                           linewidths=2, label=f'DID (m=1, k={m_did:.2f})', zorder=5)

        ax.legend(loc='upper right')

    # Add colorbar
    if add_colorbar:
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('Bias-Adjusted Estimate', fontsize=12)

    # Labels
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = "Bias-Adjusted Treatment Effect"
    ax.set_title(title, fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def placebo_lm_line_plot(
    plm: PlaceboLMResult,
    focus_param: str = "k",
    ptile_param: str = "coef_P_D_given_XZ",
    ptiles: Optional[List[float]] = None,
    gran: int = 50,
    bootstrap: bool = True,
    n_boot: int = 500,
    alpha: float = 0.05,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[plt.Figure]:
    """
    Create line plots showing estimates as a function of one parameter.

    For each percentile of the secondary parameter, creates a line plot
    showing how the bias-adjusted estimate changes with the focus parameter.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    focus_param : str, optional
        Parameter to vary on x-axis: "k" or "coef_P_D_given_XZ" (default "k").
    ptile_param : str, optional
        Parameter to hold at percentiles: "k" or "coef_P_D_given_XZ".
    ptiles : List[float], optional
        Percentiles for secondary parameter (default [0, 0.5, 1]).
    gran : int, optional
        Number of grid points for focus parameter (default 50).
    bootstrap : bool, optional
        Whether to compute bootstrap confidence intervals (default True).
    n_boot : int, optional
        Number of bootstrap iterations (default 500).
    alpha : float, optional
        Significance level for CI (default 0.05).
    figsize : Tuple[float, float], optional
        Figure size (default (10, 6)).
    title : str, optional
        Plot title prefix.
    seed : int, optional
        Random seed.

    Returns
    -------
    List[plt.Figure]
        List of figure objects, one per percentile.

    Examples
    --------
    >>> result = placebo_lm(data=df, outcome="re78", treatment="treat",
    ...                      placebo_outcome="re74", observed_covariates=covs)
    >>> figs = placebo_lm_line_plot(result, focus_param="k", n_boot=100)
    >>> plt.show()
    """
    if ptiles is None:
        ptiles = [0, 0.5, 1]

    if seed is not None:
        np.random.seed(seed)

    # Get parameter ranges
    k_range = plm.partialIDparam_minmax.get("k", [-2, 2])
    coef_range = plm.partialIDparam_minmax.get("coef_P_D_given_XZ", [-15000, 15000])

    # Determine focus and ptile values
    if focus_param == "k":
        focus_vals = np.linspace(k_range[0], k_range[1], gran)
        ptile_vals = [coef_range[0] + p * (coef_range[1] - coef_range[0]) for p in ptiles]
        ptile_label = "coef_P_D_given_XZ"
        focus_label = "k (confounding ratio)"
    else:
        focus_vals = np.linspace(coef_range[0], coef_range[1], gran)
        ptile_vals = [k_range[0] + p * (k_range[1] - k_range[0]) for p in ptiles]
        ptile_label = "k"
        focus_label = "Direct effect (coef_P_D_given_XZ)"

    figures = []

    for idx, (ptile, ptile_val) in enumerate(zip(ptiles, ptile_vals)):
        fig, ax = plt.subplots(figsize=figsize)

        # Compute estimates
        estimates = []
        ci_lows = []
        ci_highs = []

        for fv in focus_vals:
            if focus_param == "k":
                k, coef = fv, ptile_val
            else:
                k, coef = ptile_val, fv

            est = bias_adjustment(plm, k, coef)
            estimates.append(est)

            if bootstrap:
                boot_ests = _bootstrap_estimates_quiet(plm, k, coef, n_boot)
                ci_lows.append(np.percentile(boot_ests, 100 * alpha / 2))
                ci_highs.append(np.percentile(boot_ests, 100 * (1 - alpha / 2)))

        estimates = np.array(estimates)

        # Plot
        ax.plot(focus_vals, estimates, 'b-', linewidth=2, label='Estimate')

        if bootstrap:
            ci_lows = np.array(ci_lows)
            ci_highs = np.array(ci_highs)
            ax.fill_between(focus_vals, ci_lows, ci_highs, alpha=0.3, color='blue',
                           label=f'{100*(1-alpha):.0f}% CI')

        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Add benchmark points
        if focus_param == "k":
            # Mark k=0 and k=1
            est_k0 = bias_adjustment(plm, 0, ptile_val)
            est_k1 = bias_adjustment(plm, 1, ptile_val)
            ax.scatter([0], [est_k0], s=80, c='green', marker='o', zorder=5,
                       label='No Confounding (k=0)')
            ax.scatter([1], [est_k1], s=80, c='red', marker='s', zorder=5,
                       label='Perfect Placebo (k=1)')

        ax.set_xlabel(focus_label, fontsize=12)
        ax.set_ylabel('Bias-Adjusted Estimate', fontsize=12)

        if title is None:
            plot_title = f"Sensitivity: {ptile_label} = {ptile_val:.0f} ({ptile*100:.0f}th percentile)"
        else:
            plot_title = f"{title}: {ptile_label} = {ptile_val:.0f}"

        ax.set_title(plot_title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    return figures


def placebo_lm_heatmap(
    plm: PlaceboLMResult,
    gran: int = 20,
    k_range: Optional[Tuple[float, float]] = None,
    coef_range: Optional[Tuple[float, float]] = None,
    annotate: bool = True,
    fmt: str = ".0f",
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (12, 10),
) -> plt.Figure:
    """
    Create a heatmap of bias-adjusted estimates.

    Similar to contour plot but with discrete cells and optional annotations.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    gran : int, optional
        Number of grid points (default 20).
    k_range : Tuple[float, float], optional
        Range for k axis.
    coef_range : Tuple[float, float], optional
        Range for coef axis.
    annotate : bool, optional
        Whether to annotate cells with values (default True).
    fmt : str, optional
        Format string for annotations (default ".0f").
    cmap : str, optional
        Colormap name (default "RdBu_r").
    figsize : Tuple[float, float], optional
        Figure size (default (12, 10)).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Get parameter ranges
    if k_range is None:
        k_range = tuple(plm.partialIDparam_minmax.get("k", [-2, 2]))
    if coef_range is None:
        coef_range = tuple(plm.partialIDparam_minmax.get("coef_P_D_given_XZ", [-15000, 15000]))

    # Create grid
    k_vals = np.linspace(k_range[0], k_range[1], gran)
    coef_vals = np.linspace(coef_range[0], coef_range[1], gran)

    # Compute estimates
    Z = np.zeros((gran, gran))
    for i, coef in enumerate(coef_vals):
        for j, k in enumerate(k_vals):
            Z[i, j] = bias_adjustment(plm, k, coef)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Symmetric color limits
    vmax = max(abs(Z.min()), abs(Z.max()))

    im = ax.imshow(Z, cmap=cmap, aspect='auto', origin='lower',
                   extent=[k_range[0], k_range[1], coef_range[0], coef_range[1]],
                   vmin=-vmax, vmax=vmax)

    if annotate and gran <= 15:
        for i in range(gran):
            for j in range(gran):
                text = ax.text(k_vals[j], coef_vals[i], f"{Z[i, j]:{fmt}}",
                              ha="center", va="center", fontsize=8)

    ax.set_xlabel('k (confounding ratio)', fontsize=12)
    ax.set_ylabel('Direct effect (coef_P_D_given_XZ)', fontsize=12)
    ax.set_title('Bias-Adjusted Treatment Effect', fontsize=14)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Estimate', fontsize=12)

    plt.tight_layout()
    return fig


def placebo_lm_breakdown_plot(
    plm: PlaceboLMResult,
    coef_P_D_given_XZ: float = 0.0,
    k_range: Optional[Tuple[float, float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    figsize: Tuple[float, float] = (10, 6),
    seed: Optional[int] = None,
) -> plt.Figure:
    """
    Create a breakdown plot showing the value of k where estimates cross zero.

    This visualization helps identify the "breakdown point" - the value of k
    at which the treatment effect estimate becomes statistically insignificant
    or changes sign.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    coef_P_D_given_XZ : float, optional
        Direct effect parameter (default 0).
    k_range : Tuple[float, float], optional
        Range for k values to plot.
    n_boot : int, optional
        Number of bootstrap iterations.
    alpha : float, optional
        Significance level.
    figsize : Tuple[float, float], optional
        Figure size.
    seed : int, optional
        Random seed.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    if seed is not None:
        np.random.seed(seed)

    if k_range is None:
        k_range = tuple(plm.partialIDparam_minmax.get("k", [-2, 2]))

    k_vals = np.linspace(k_range[0], k_range[1], 100)

    # Compute estimates and CIs
    estimates = []
    ci_lows = []
    ci_highs = []

    for k in k_vals:
        est = bias_adjustment(plm, k, coef_P_D_given_XZ)
        estimates.append(est)

        boot_ests = _bootstrap_estimates_quiet(plm, k, coef_P_D_given_XZ, n_boot)
        ci_lows.append(np.percentile(boot_ests, 100 * alpha / 2))
        ci_highs.append(np.percentile(boot_ests, 100 * (1 - alpha / 2)))

    estimates = np.array(estimates)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    # Find breakdown points
    sign_changes = np.where(np.diff(np.sign(estimates)))[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot estimate and CI
    ax.plot(k_vals, estimates, 'b-', linewidth=2, label='Estimate')
    ax.fill_between(k_vals, ci_lows, ci_highs, alpha=0.3, color='blue',
                    label=f'{100*(1-alpha):.0f}% CI')

    # Reference line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Mark breakdown points
    for sc in sign_changes:
        k_breakdown = k_vals[sc]
        ax.axvline(x=k_breakdown, color='red', linestyle=':', linewidth=2,
                   label=f'Sign change at k={k_breakdown:.2f}')

    # Mark benchmark points
    ax.axvline(x=0, color='green', linestyle='-', linewidth=1, alpha=0.5,
               label='k=0 (No confounding)')
    ax.axvline(x=1, color='orange', linestyle='-', linewidth=1, alpha=0.5,
               label='k=1 (Perfect placebo)')

    ax.set_xlabel('k (confounding ratio)', fontsize=12)
    ax.set_ylabel('Bias-Adjusted Estimate', fontsize=12)
    ax.set_title(f'Breakdown Analysis (coef_P_D_given_XZ = {coef_P_D_given_XZ})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_all_plots(
    plm: PlaceboLMResult,
    output_dir: str = ".",
    prefix: str = "placebolm",
    dpi: int = 150,
    **kwargs,
) -> List[str]:
    """
    Generate and save all standard plots.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    output_dir : str, optional
        Directory to save plots (default ".").
    prefix : str, optional
        Filename prefix (default "placebolm").
    dpi : int, optional
        Resolution (default 150).
    **kwargs
        Additional arguments passed to plotting functions.

    Returns
    -------
    List[str]
        List of saved file paths.
    """
    import os

    saved_files = []

    # Contour plot
    fig = placebo_lm_contour_plot(plm, **kwargs)
    path = os.path.join(output_dir, f"{prefix}_contour.png")
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path)

    # Line plots
    figs = placebo_lm_line_plot(plm, **kwargs)
    for i, fig in enumerate(figs):
        path = os.path.join(output_dir, f"{prefix}_line_{i}.png")
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)

    # Breakdown plot
    fig = placebo_lm_breakdown_plot(plm, **kwargs)
    path = os.path.join(output_dir, f"{prefix}_breakdown.png")
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path)

    return saved_files
