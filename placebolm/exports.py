"""
Additional visualization and export functions for PlaceboLM.

This module adds:
1. Breakdown Frontier Plot - shows where estimates cross zero
2. Confidence Band Ribbon Plot - continuous uncertainty visualization
3. LaTeX/Markdown Table Export - publication-ready tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import Optional, List, Tuple, Dict, Union
from scipy import optimize

from .core import (
    PlaceboLMResult,
    bias_adjustment,
    bootstrap_ci,
)
from .tables import _bootstrap_estimates_quiet


# =============================================================================
# Breakdown Frontier Plot
# =============================================================================

def find_breakdown_k(
    plm: PlaceboLMResult,
    coef_P_D_given_XZ: float = 0.0,
    target: float = 0.0,
) -> float:
    """
    Find the value of k where the estimate equals the target.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    coef_P_D_given_XZ : float
        Direct effect of treatment on placebo.
    target : float
        Target value (default 0 for breakdown point).

    Returns
    -------
    float
        Value of k where estimate equals target, or np.nan if undefined.
    """
    # Solve: beta_Y - k * (beta_N - coef) = target
    # k = (beta_Y - target) / (beta_N - coef)
    denom = plm.beta_N_D_given_X - coef_P_D_given_XZ
    if abs(denom) < 1e-10:
        return np.nan
    return (plm.beta_Y_D_given_X - target) / denom


def find_breakdown_coef(
    plm: PlaceboLMResult,
    k: float,
    target: float = 0.0,
) -> float:
    """
    Find the value of coef_P_D_given_XZ where the estimate equals the target.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k : float
        Confounding ratio.
    target : float
        Target value (default 0 for breakdown point).

    Returns
    -------
    float
        Value of coef where estimate equals target, or np.nan if k=0.
    """
    # Solve: beta_Y - k * (beta_N - coef) = target
    # coef = beta_N - (beta_Y - target) / k
    if abs(k) < 1e-10:
        return np.nan
    return plm.beta_N_D_given_X - (plm.beta_Y_D_given_X - target) / k


def breakdown_frontier(
    plm: PlaceboLMResult,
    target: float = 0.0,
    k_range: Optional[Tuple[float, float]] = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Compute the breakdown frontier - combinations of (k, coef) where estimate equals target.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    target : float
        Target value (default 0).
    k_range : Tuple[float, float], optional
        Range of k values to consider.
    n_points : int
        Number of points on frontier.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'k' and 'coef_P_D_given_XZ' defining the frontier.
    """
    if k_range is None:
        k_range = plm.partialIDparam_minmax.get("k", [-2, 2])

    # Generate k values (excluding 0 to avoid division issues)
    k_vals = np.linspace(k_range[0], k_range[1], n_points)
    k_vals = k_vals[np.abs(k_vals) > 1e-6]

    coef_vals = [find_breakdown_coef(plm, k, target) for k in k_vals]

    return pd.DataFrame({
        'k': k_vals,
        'coef_P_D_given_XZ': coef_vals,
    })


def placebo_lm_breakdown_frontier_plot(
    plm: PlaceboLMResult,
    target: float = 0.0,
    k_range: Optional[Tuple[float, float]] = None,
    coef_range: Optional[Tuple[float, float]] = None,
    n_points: int = 100,
    show_regions: bool = True,
    show_benchmarks: bool = True,
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot the breakdown frontier showing where estimates cross the target value.

    The frontier divides the parameter space into regions where the estimate
    is above vs. below the target (typically zero). This shows what combinations
    of assumptions would be needed to overturn the finding.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    target : float
        Target value for breakdown (default 0).
    k_range : Tuple[float, float], optional
        Range for k axis.
    coef_range : Tuple[float, float], optional
        Range for coef axis.
    n_points : int
        Number of points on frontier line.
    show_regions : bool
        Whether to shade regions above/below target.
    show_benchmarks : bool
        Whether to show benchmark points (k=0, k=1).
    figsize : Tuple[float, float]
        Figure size.
    title : str, optional
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    if k_range is None:
        k_range = plm.partialIDparam_minmax.get("k", [-2, 2])
    if coef_range is None:
        coef_range = plm.partialIDparam_minmax.get("coef_P_D_given_XZ", [-15000, 15000])

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute frontier
    frontier = breakdown_frontier(plm, target=target, k_range=k_range, n_points=n_points)

    # Filter to plot range
    mask = (frontier['coef_P_D_given_XZ'] >= coef_range[0]) & \
           (frontier['coef_P_D_given_XZ'] <= coef_range[1])
    frontier_plot = frontier[mask]

    if show_regions and len(frontier_plot) > 0:
        # Determine which side is positive
        test_k = (k_range[0] + k_range[1]) / 2
        test_coef = coef_range[0]
        test_est = bias_adjustment(plm, test_k, test_coef)

        # Create polygon for positive region
        k_vals = frontier_plot['k'].values
        coef_vals = frontier_plot['coef_P_D_given_XZ'].values

        # Sort by k for proper polygon
        sort_idx = np.argsort(k_vals)
        k_vals = k_vals[sort_idx]
        coef_vals = coef_vals[sort_idx]

        # Fill above/below frontier
        if test_est > target:
            # Below frontier is positive
            ax.fill_between(k_vals, coef_range[0], coef_vals,
                           alpha=0.2, color='blue', label=f'Estimate > {target}')
            ax.fill_between(k_vals, coef_vals, coef_range[1],
                           alpha=0.2, color='red', label=f'Estimate < {target}')
        else:
            # Above frontier is positive
            ax.fill_between(k_vals, coef_vals, coef_range[1],
                           alpha=0.2, color='blue', label=f'Estimate > {target}')
            ax.fill_between(k_vals, coef_range[0], coef_vals,
                           alpha=0.2, color='red', label=f'Estimate < {target}')

    # Plot frontier line
    if len(frontier_plot) > 0:
        ax.plot(frontier_plot['k'], frontier_plot['coef_P_D_given_XZ'],
                'k-', linewidth=2.5, label=f'Breakdown frontier (est = {target})')

    # Add benchmarks
    if show_benchmarks:
        # k=0 line (no confounding)
        ax.axvline(x=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='No confounding (k=0)')

        # k=1 line (equiconfounding/parallel trends)
        ax.axvline(x=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Equiconfounding (k=1)')

        # Perfect placebo line
        ax.axhline(y=0, color='purple', linestyle=':', linewidth=1.5, alpha=0.7,
                   label='Perfect placebo (coef=0)')

        # Mark special points
        # Breakdown k at perfect placebo
        k_breakdown = find_breakdown_k(plm, coef_P_D_given_XZ=0, target=target)
        if not np.isnan(k_breakdown) and k_range[0] <= k_breakdown <= k_range[1]:
            ax.scatter([k_breakdown], [0], s=150, c='black', marker='*', zorder=10,
                      label=f'Breakdown k = {k_breakdown:.2f}')

    # Labels and styling
    ax.set_xlabel('k (confounding ratio)', fontsize=12)
    ax.set_ylabel('Direct effect (coef_P_D_given_XZ)', fontsize=12)
    ax.set_xlim(k_range)
    ax.set_ylim(coef_range)

    if title is None:
        title = f'Breakdown Frontier: What Would Overturn the Finding?'
    ax.set_title(title, fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# Confidence Band Ribbon Plot
# =============================================================================

def placebo_lm_ribbon_plot(
    plm: PlaceboLMResult,
    focus_param: str = "k",
    fixed_value: float = 0.0,
    param_range: Optional[Tuple[float, float]] = None,
    n_points: int = 50,
    n_boot: int = 500,
    alpha: float = 0.05,
    show_benchmarks: bool = True,
    show_zero_line: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    color: str = 'steelblue',
    seed: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Create a line plot with continuous bootstrap confidence bands.

    This provides a cleaner visualization of uncertainty compared to
    discrete confidence intervals at specific points.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    focus_param : str
        Parameter to vary: "k" or "coef_P_D_given_XZ".
    fixed_value : float
        Value for the non-varying parameter.
    param_range : Tuple[float, float], optional
        Range for focus parameter.
    n_points : int
        Number of points for the line.
    n_boot : int
        Number of bootstrap iterations.
    alpha : float
        Significance level for CI.
    show_benchmarks : bool
        Whether to show benchmark points.
    show_zero_line : bool
        Whether to show horizontal line at y=0.
    figsize : Tuple[float, float]
        Figure size.
    title : str, optional
        Plot title.
    color : str
        Color for line and bands.
    seed : int, optional
        Random seed for reproducibility.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get parameter range
    if param_range is None:
        if focus_param == "k":
            param_range = plm.partialIDparam_minmax.get("k", [-2, 2])
        else:
            param_range = plm.partialIDparam_minmax.get("coef_P_D_given_XZ", [-15000, 15000])

    # Generate parameter values
    param_vals = np.linspace(param_range[0], param_range[1], n_points)

    # Compute estimates and CIs
    estimates = []
    ci_lows = []
    ci_highs = []

    for pv in param_vals:
        if focus_param == "k":
            k, coef = pv, fixed_value
        else:
            k, coef = fixed_value, pv

        est = bias_adjustment(plm, k, coef)
        estimates.append(est)

        # Bootstrap CI
        boot_ests = _bootstrap_estimates_quiet(plm, k, coef, n_boot)
        ci_lows.append(np.percentile(boot_ests, 100 * alpha / 2))
        ci_highs.append(np.percentile(boot_ests, 100 * (1 - alpha / 2)))

    estimates = np.array(estimates)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot confidence band
    ax.fill_between(param_vals, ci_lows, ci_highs, alpha=0.3, color=color,
                    label=f'{100*(1-alpha):.0f}% Confidence Band')

    # Plot estimate line
    ax.plot(param_vals, estimates, '-', linewidth=2.5, color=color, label='Point Estimate')

    # Reference lines
    if show_zero_line:
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Benchmarks
    if show_benchmarks and focus_param == "k":
        # k=0 (no confounding)
        est_k0 = bias_adjustment(plm, 0, fixed_value)
        ax.scatter([0], [est_k0], s=100, c='green', marker='o', zorder=5,
                   edgecolors='black', linewidths=1.5)
        ax.annotate(f'k=0\n({est_k0:.2f})', (0, est_k0), textcoords="offset points",
                   xytext=(10, 10), fontsize=9)

        # k=1 (equiconfounding)
        est_k1 = bias_adjustment(plm, 1, fixed_value)
        ax.scatter([1], [est_k1], s=100, c='red', marker='s', zorder=5,
                   edgecolors='black', linewidths=1.5)
        ax.annotate(f'k=1\n({est_k1:.2f})', (1, est_k1), textcoords="offset points",
                   xytext=(10, -20), fontsize=9)

        # Breakdown point (where CI crosses zero)
        k_breakdown = find_breakdown_k(plm, coef_P_D_given_XZ=fixed_value, target=0)
        if not np.isnan(k_breakdown) and param_range[0] <= k_breakdown <= param_range[1]:
            ax.axvline(x=k_breakdown, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.annotate(f'Breakdown\nk={k_breakdown:.2f}', (k_breakdown, ax.get_ylim()[0]),
                       textcoords="offset points", xytext=(5, 20), fontsize=9, color='gray')

    # Labels
    if focus_param == "k":
        ax.set_xlabel('k (confounding ratio)', fontsize=12)
    else:
        ax.set_xlabel('Direct effect (coef_P_D_given_XZ)', fontsize=12)

    ax.set_ylabel('Bias-Adjusted Estimate', fontsize=12)

    if title is None:
        fixed_label = "coef_P_D_given_XZ" if focus_param == "k" else "k"
        title = f'Sensitivity Analysis ({fixed_label} = {fixed_value})'
    ax.set_title(title, fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# LaTeX and Markdown Table Export
# =============================================================================

def format_estimate(value: float, se: Optional[float] = None,
                   ci: Optional[Tuple[float, float]] = None,
                   decimals: int = 3, format_type: str = "estimate_se") -> str:
    """
    Format an estimate with standard error or confidence interval.

    Parameters
    ----------
    value : float
        Point estimate.
    se : float, optional
        Standard error.
    ci : Tuple[float, float], optional
        Confidence interval (low, high).
    decimals : int
        Number of decimal places.
    format_type : str
        One of "estimate_se", "estimate_ci", "estimate_only".

    Returns
    -------
    str
        Formatted string.
    """
    fmt = f"{{:.{decimals}f}}"

    if format_type == "estimate_only":
        return fmt.format(value)
    elif format_type == "estimate_se" and se is not None:
        return f"{fmt.format(value)} ({fmt.format(se)})"
    elif format_type == "estimate_ci" and ci is not None:
        return f"{fmt.format(value)} [{fmt.format(ci[0])}, {fmt.format(ci[1])}]"
    else:
        return fmt.format(value)


def placebo_lm_table_latex(
    plm: PlaceboLMResult,
    k_values: Optional[List[float]] = None,
    coef_values: Optional[List[float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    decimals: int = 3,
    include_benchmarks: bool = True,
    caption: str = "PlaceboLM Sensitivity Analysis",
    label: str = "tab:placebolm",
    notes: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Generate a publication-ready LaTeX table.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k_values : List[float], optional
        Values of k to include (default [0, 0.5, 1, 1.5, 2]).
    coef_values : List[float], optional
        Values of coef to include (default [0]).
    n_boot : int
        Number of bootstrap iterations for CIs.
    alpha : float
        Significance level.
    decimals : int
        Decimal places.
    include_benchmarks : bool
        Whether to include labeled benchmark rows.
    caption : str
        Table caption.
    label : str
        LaTeX label.
    notes : str, optional
        Notes to add below table.
    seed : int, optional
        Random seed.

    Returns
    -------
    str
        LaTeX table code.
    """
    if seed is not None:
        np.random.seed(seed)

    if k_values is None:
        k_values = [0, 0.5, 1, 1.5, 2]
    if coef_values is None:
        coef_values = [0]

    # Build table data
    rows = []

    for coef in coef_values:
        for k in k_values:
            est = bias_adjustment(plm, k, coef)
            boot_ests = _bootstrap_estimates_quiet(plm, k, coef, n_boot)
            se = np.std(boot_ests)
            ci_low = np.percentile(boot_ests, 100 * alpha / 2)
            ci_high = np.percentile(boot_ests, 100 * (1 - alpha / 2))

            # Determine benchmark label
            if include_benchmarks:
                if k == 0 and coef == 0:
                    label_str = "No Confounding"
                elif k == 1 and coef == 0:
                    label_str = "Equiconfounding (DID)"
                else:
                    label_str = ""
            else:
                label_str = ""

            rows.append({
                'k': k,
                'coef': coef,
                'estimate': est,
                'se': se,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'label': label_str,
            })

    # Build LaTeX
    fmt = f"{{:.{decimals}f}}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\hline\hline",
        r"Assumption & $k$ & Estimate & Std. Error & 95\% CI \\",
        r"\hline",
    ]

    for row in rows:
        k_str = fmt.format(row['k'])
        est_str = fmt.format(row['estimate'])
        se_str = fmt.format(row['se'])
        ci_str = f"[{fmt.format(row['ci_low'])}, {fmt.format(row['ci_high'])}]"

        if row['label']:
            label_str = row['label']
        else:
            label_str = f"$k = {k_str}$"

        lines.append(f"{label_str} & {k_str} & {est_str} & {se_str} & {ci_str} \\\\")

    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
    ])

    if notes:
        lines.extend([
            r"\begin{tablenotes}",
            r"\small",
            f"\\item {notes}",
            r"\end{tablenotes}",
        ])

    lines.append(r"\end{table}")

    return "\n".join(lines)


def placebo_lm_table_markdown(
    plm: PlaceboLMResult,
    k_values: Optional[List[float]] = None,
    coef_values: Optional[List[float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    decimals: int = 3,
    include_benchmarks: bool = True,
    title: str = "PlaceboLM Sensitivity Analysis",
    seed: Optional[int] = None,
) -> str:
    """
    Generate a Markdown table.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k_values : List[float], optional
        Values of k to include.
    coef_values : List[float], optional
        Values of coef to include.
    n_boot : int
        Number of bootstrap iterations.
    alpha : float
        Significance level.
    decimals : int
        Decimal places.
    include_benchmarks : bool
        Whether to include labeled benchmark rows.
    title : str
        Table title.
    seed : int, optional
        Random seed.

    Returns
    -------
    str
        Markdown table.
    """
    if seed is not None:
        np.random.seed(seed)

    if k_values is None:
        k_values = [0, 0.5, 1, 1.5, 2]
    if coef_values is None:
        coef_values = [0]

    fmt = f"{{:.{decimals}f}}"

    lines = [
        f"### {title}",
        "",
        "| Assumption | k | Estimate | Std. Error | 95% CI |",
        "|------------|---|----------|------------|--------|",
    ]

    for coef in coef_values:
        for k in k_values:
            est = bias_adjustment(plm, k, coef)
            boot_ests = _bootstrap_estimates_quiet(plm, k, coef, n_boot)
            se = np.std(boot_ests)
            ci_low = np.percentile(boot_ests, 100 * alpha / 2)
            ci_high = np.percentile(boot_ests, 100 * (1 - alpha / 2))

            if include_benchmarks:
                if k == 0 and coef == 0:
                    label = "No Confounding"
                elif k == 1 and coef == 0:
                    label = "Equiconfounding (DID)"
                else:
                    label = f"k = {fmt.format(k)}"
            else:
                label = f"k = {fmt.format(k)}"

            ci_str = f"[{fmt.format(ci_low)}, {fmt.format(ci_high)}]"
            lines.append(f"| {label} | {fmt.format(k)} | {fmt.format(est)} | {fmt.format(se)} | {ci_str} |")

    # Add notes
    lines.extend([
        "",
        f"*Notes:* Bootstrap standard errors and confidence intervals based on {n_boot} iterations.",
        f"Naive estimate (no adjustment): {fmt.format(plm.beta_Y_D_given_X)}",
    ])

    return "\n".join(lines)


def export_results(
    plm: PlaceboLMResult,
    filepath: str,
    format: str = "auto",
    **kwargs,
) -> None:
    """
    Export PlaceboLM results to a file.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    filepath : str
        Output file path.
    format : str
        Output format: "latex", "markdown", "csv", or "auto" (detect from extension).
    **kwargs
        Additional arguments passed to the formatting function.
    """
    if format == "auto":
        if filepath.endswith(".tex"):
            format = "latex"
        elif filepath.endswith(".md"):
            format = "markdown"
        elif filepath.endswith(".csv"):
            format = "csv"
        else:
            format = "markdown"

    if format == "latex":
        content = placebo_lm_table_latex(plm, **kwargs)
    elif format == "markdown":
        content = placebo_lm_table_markdown(plm, **kwargs)
    elif format == "csv":
        # Generate DataFrame and save
        from .tables import sensitivity_table
        df = sensitivity_table(plm, **kwargs)
        df.to_csv(filepath, index=False)
        return
    else:
        raise ValueError(f"Unknown format: {format}")

    with open(filepath, 'w') as f:
        f.write(content)


# =============================================================================
# Summary Panel
# =============================================================================

def placebo_lm_summary_panel(
    plm: PlaceboLMResult,
    # Ribbon plot parameters
    ribbon_focus_param: str = "k",
    ribbon_fixed_value: float = 0.0,
    ribbon_param_range: Optional[Tuple[float, float]] = None,
    ribbon_n_points: int = 40,
    ribbon_n_boot: int = 300,
    ribbon_alpha: float = 0.05,
    ribbon_color: str = 'steelblue',
    # Breakdown frontier parameters
    frontier_target: float = 0.0,
    frontier_k_range: Optional[Tuple[float, float]] = None,
    frontier_coef_range: Optional[Tuple[float, float]] = None,
    frontier_show_regions: bool = True,
    # Table parameters
    table_k_values: Optional[List[float]] = None,
    table_decimals: int = 2,
    # General parameters
    true_effect: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 11),
    seed: Optional[int] = None,
) -> plt.Figure:
    """
    Generate a comprehensive 4-panel summary figure.

    Panels:
    - A (top-left): Ribbon plot showing sensitivity to confounding
    - B (top-right): Breakdown frontier showing what would overturn findings
    - C (bottom-left): Key statistics and estimates table
    - D (bottom-right): Interpretation guide

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().

    Ribbon Plot Parameters
    ----------------------
    ribbon_focus_param : str
        Parameter to vary: "k" or "coef_P_D_given_XZ" (default "k").
    ribbon_fixed_value : float
        Value for the non-varying parameter (default 0.0).
    ribbon_param_range : Tuple[float, float], optional
        Range for focus parameter. If None, uses plm.partialIDparam_minmax.
    ribbon_n_points : int
        Number of points for the line (default 40).
    ribbon_n_boot : int
        Number of bootstrap iterations (default 300).
    ribbon_alpha : float
        Significance level for CI (default 0.05).
    ribbon_color : str
        Color for line and bands (default 'steelblue').

    Breakdown Frontier Parameters
    -----------------------------
    frontier_target : float
        Target value for breakdown (default 0).
    frontier_k_range : Tuple[float, float], optional
        Range for k axis. If None, uses plm.partialIDparam_minmax.
    frontier_coef_range : Tuple[float, float], optional
        Range for coef axis. If None, uses plm.partialIDparam_minmax.
    frontier_show_regions : bool
        Whether to shade regions above/below target (default True).

    Table Parameters
    ----------------
    table_k_values : List[float], optional
        Values of k to include in summary (default [0, 0.5, 1, 1.5, 2]).
    table_decimals : int
        Decimal places for estimates (default 2).

    General Parameters
    ------------------
    true_effect : float, optional
        If provided, adds a horizontal line showing the true effect.
    title : str, optional
        Main title for the figure.
    figsize : Tuple[float, float]
        Figure size (default (14, 11)).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    if seed is not None:
        np.random.seed(seed)

    # Set defaults from plm if not provided
    if ribbon_param_range is None:
        if ribbon_focus_param == "k":
            ribbon_param_range = tuple(plm.partialIDparam_minmax.get("k", [-2, 2]))
        else:
            ribbon_param_range = tuple(plm.partialIDparam_minmax.get("coef_P_D_given_XZ", [-15000, 15000]))

    if frontier_k_range is None:
        frontier_k_range = tuple(plm.partialIDparam_minmax.get("k", [-2, 2]))

    if frontier_coef_range is None:
        frontier_coef_range = tuple(plm.partialIDparam_minmax.get("coef_P_D_given_XZ", [-15000, 15000]))

    if table_k_values is None:
        table_k_values = [0, 0.5, 1, 1.5, 2]

    # Compute key values
    est_k0 = bias_adjustment(plm, k=0, coef_P_D_given_XZ=0)
    est_k1 = bias_adjustment(plm, k=1, coef_P_D_given_XZ=0)
    k_breakdown = find_breakdown_k(plm, coef_P_D_given_XZ=0, target=frontier_target)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # =========================================================================
    # Panel A: Ribbon Plot
    # =========================================================================
    ax1 = axes[0, 0]

    # Generate parameter values
    param_vals = np.linspace(ribbon_param_range[0], ribbon_param_range[1], ribbon_n_points)

    # Compute estimates and CIs
    estimates = []
    ci_lows = []
    ci_highs = []

    for pv in param_vals:
        if ribbon_focus_param == "k":
            k, coef = pv, ribbon_fixed_value
        else:
            k, coef = ribbon_fixed_value, pv

        est = bias_adjustment(plm, k, coef)
        estimates.append(est)

        boot_ests = _bootstrap_estimates_quiet(plm, k, coef, ribbon_n_boot)
        ci_lows.append(np.percentile(boot_ests, 100 * ribbon_alpha / 2))
        ci_highs.append(np.percentile(boot_ests, 100 * (1 - ribbon_alpha / 2)))

    estimates = np.array(estimates)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    # Plot
    ax1.fill_between(param_vals, ci_lows, ci_highs, alpha=0.3, color=ribbon_color,
                     label=f'{100*(1-ribbon_alpha):.0f}% CI')
    ax1.plot(param_vals, estimates, '-', linewidth=2.5, color=ribbon_color, label='Estimate')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    if true_effect is not None:
        ax1.axhline(y=true_effect, color='green', linestyle=':', linewidth=2,
                    label=f'True Effect ({true_effect:.0f})')

    # Benchmarks for k
    if ribbon_focus_param == "k":
        est_at_0 = bias_adjustment(plm, 0, ribbon_fixed_value)
        est_at_1 = bias_adjustment(plm, 1, ribbon_fixed_value)
        ax1.scatter([0], [est_at_0], s=100, c='green', marker='o', zorder=5,
                    edgecolors='black', linewidths=1.5, label='k=0 (no confounding)')
        ax1.scatter([1], [est_at_1], s=100, c='red', marker='s', zorder=5,
                    edgecolors='black', linewidths=1.5, label='k=1 (equiconfounding)')

        # Breakdown point
        k_bd = find_breakdown_k(plm, coef_P_D_given_XZ=ribbon_fixed_value, target=frontier_target)
        if not np.isnan(k_bd) and ribbon_param_range[0] <= k_bd <= ribbon_param_range[1]:
            ax1.axvline(x=k_bd, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            ax1.annotate(f'Breakdown\nk={k_bd:.2f}', (k_bd, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 0),
                        textcoords="offset points", xytext=(5, 10), fontsize=9, color='gray')

        ax1.set_xlabel('k (confounding ratio)', fontsize=11)
    else:
        ax1.set_xlabel('Direct effect (coef_P_D_given_XZ)', fontsize=11)

    ax1.set_ylabel('Bias-Adjusted Estimate', fontsize=11)
    ax1.set_title('A. Sensitivity to Assumptions', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel B: Breakdown Frontier
    # =========================================================================
    ax2 = axes[0, 1]

    # Compute frontier
    frontier = breakdown_frontier(plm, target=frontier_target, k_range=frontier_k_range, n_points=100)

    # Filter to plot range
    mask = (frontier['coef_P_D_given_XZ'] >= frontier_coef_range[0]) & \
           (frontier['coef_P_D_given_XZ'] <= frontier_coef_range[1])
    frontier_plot = frontier[mask]

    if frontier_show_regions and len(frontier_plot) > 0:
        # Determine which side is positive
        test_k = (frontier_k_range[0] + frontier_k_range[1]) / 2
        test_coef = frontier_coef_range[0]
        test_est = bias_adjustment(plm, test_k, test_coef)

        k_vals = frontier_plot['k'].values
        coef_vals = frontier_plot['coef_P_D_given_XZ'].values
        sort_idx = np.argsort(k_vals)
        k_vals = k_vals[sort_idx]
        coef_vals = coef_vals[sort_idx]

        if test_est > frontier_target:
            ax2.fill_between(k_vals, frontier_coef_range[0], coef_vals,
                            alpha=0.2, color='blue', label=f'Estimate > {frontier_target}')
            ax2.fill_between(k_vals, coef_vals, frontier_coef_range[1],
                            alpha=0.2, color='red', label=f'Estimate < {frontier_target}')
        else:
            ax2.fill_between(k_vals, coef_vals, frontier_coef_range[1],
                            alpha=0.2, color='blue', label=f'Estimate > {frontier_target}')
            ax2.fill_between(k_vals, frontier_coef_range[0], coef_vals,
                            alpha=0.2, color='red', label=f'Estimate < {frontier_target}')

    if len(frontier_plot) > 0:
        ax2.plot(frontier_plot['k'], frontier_plot['coef_P_D_given_XZ'],
                'k-', linewidth=2.5, label='Breakdown frontier')

    # Reference lines
    ax2.axvline(x=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)

    if not np.isnan(k_breakdown) and frontier_k_range[0] <= k_breakdown <= frontier_k_range[1]:
        ax2.scatter([k_breakdown], [0], s=150, c='black', marker='*', zorder=10,
                   label=f'Breakdown k = {k_breakdown:.2f}')

    ax2.set_xlabel('k (confounding ratio)', fontsize=11)
    ax2.set_ylabel('Direct effect (coef_P_D_given_XZ)', fontsize=11)
    ax2.set_xlim(frontier_k_range)
    ax2.set_ylim(frontier_coef_range)
    ax2.set_title('B. Breakdown Frontier', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel C: Summary Statistics
    # =========================================================================
    ax3 = axes[1, 0]
    ax3.axis('off')

    # Build summary text
    fmt = f"{{:.{table_decimals}f}}"

    lines = [
        "KEY RESULTS",
        "=" * 45,
        "",
        f"Naive Estimate (beta_Y_D|X):    {fmt.format(plm.beta_Y_D_given_X)}",
        f"Placebo Coefficient (beta_N_D|X): {fmt.format(plm.beta_N_D_given_X)}",
        f"Sample Size:                   {plm.n_obs}",
        "",
        "BIAS-ADJUSTED ESTIMATES:",
    ]

    for k in table_k_values:
        est = bias_adjustment(plm, k, 0)
        if k == 0:
            label = "k=0 (no confounding)"
        elif k == 1:
            label = "k=1 (equiconfounding)"
        else:
            label = f"k={k}"
        lines.append(f"  {label:25s} {fmt.format(est)}")

    lines.extend([
        "",
        "BREAKDOWN ANALYSIS:",
        f"  Breakdown k (est = {frontier_target}):     {k_breakdown:.2f}" if not np.isnan(k_breakdown) else "  Breakdown k: N/A",
        "",
    ])

    # Add interpretation
    if not np.isnan(k_breakdown) and k_breakdown > 0:
        if k_breakdown > 1:
            interp = (f"  The finding is ROBUST: confounding would\n"
                     f"  need to be {k_breakdown:.1f}x stronger than in the\n"
                     f"  placebo to nullify the effect.")
        else:
            interp = (f"  The finding is SENSITIVE: even {k_breakdown:.1f}x\n"
                     f"  the placebo confounding would nullify it.")
        lines.append(interp)

    summary_text = "\n".join(lines)
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.set_title('C. Summary Statistics', fontsize=12, fontweight='bold')

    # =========================================================================
    # Panel D: Interpretation Guide
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.axis('off')

    guide_text = (
        "INTERPRETATION GUIDE\n"
        "=" * 44 + "\n"
        "\n"
        "THE k PARAMETER (Confounding Ratio):\n"
        "  k = 0: No unobserved confounding\n"
        "  Treatment is as good as randomly assigned\n"
        "\n"
        "  k = 1: Equiconfounding (Parallel Trends)\n"
        "  Confounding in outcome = confounding in placebo\n"
        "  This is the standard DID assumption\n"
        "\n"
        "  k > 1: Outcome more confounded than placebo\n"
        "  k < 1: Outcome less confounded than placebo\n"
        "\n"
        "BREAKDOWN POINT:\n"
        "  The k value where estimate equals zero\n"
        '  Answers: "How wrong would equiconfounding\n'
        '  need to be to nullify the finding?"\n'
        "  Higher breakdown k = more robust finding\n"
        "\n"
        "PLACEBO IMPERFECTION (coef_P_D_given_XZ):\n"
        "  Direct effect of treatment on placebo\n"
        '  Zero for "perfect" placebos\n'
        "  Non-zero allows for imperfect placebos\n"
    )
    ax4.text(0.05, 0.95, guide_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax4.set_title('D. Interpretation Guide', fontsize=12, fontweight='bold')

    # =========================================================================
    # Final touches
    # =========================================================================
    if title is None:
        title = 'PlaceboLM Summary: Sensitivity Analysis'

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    return fig
