"""
Table generation for PlaceboLM results.

This module provides functions for creating summary tables of bias-adjusted
estimates across a grid of partial identification parameters.
"""

import numpy as np
import pandas as pd
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


def placebo_lm_table(
    plm: PlaceboLMResult,
    n_boot: int = 1000,
    ptiles: Optional[List[float]] = None,
    alpha: float = 0.05,
    include_benchmarks: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate a summary table of bias-adjusted estimates.

    Creates a table showing estimates, standard errors, and confidence intervals
    for various assumptions about confounding (k) and placebo imperfection
    (coef_P_D_given_XZ).

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    n_boot : int, optional
        Number of bootstrap iterations for SE and CI (default 1000).
    ptiles : List[float], optional
        Percentiles to use for grid over partial ID parameters (default [0.25, 0.5, 0.75]).
    alpha : float, optional
        Significance level for confidence intervals (default 0.05).
    include_benchmarks : bool, optional
        Whether to include benchmark rows like "No Unobserved Confounding",
        "DID (m=1)", and "Perfect Placebo" (default True).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print progress (default True).

    Returns
    -------
    pd.DataFrame
        Table with columns:
        - "Scenario": Description of the assumption
        - "k": Value of confounding ratio parameter
        - "coef_P_D_given_XZ": Value of direct treatment-placebo effect
        - "Estimate": Bias-adjusted point estimate
        - "Std. Error": Bootstrap standard error
        - "CI Low": Lower bound of confidence interval
        - "CI High": Upper bound of confidence interval

    Examples
    --------
    >>> result = placebo_lm(data=df, outcome="re78", treatment="treat",
    ...                      placebo_outcome="re74", observed_covariates=covs)
    >>> table = placebo_lm_table(result, n_boot=500)
    >>> print(table)
    """
    if ptiles is None:
        ptiles = [0.25, 0.5, 0.75]

    if seed is not None:
        np.random.seed(seed)

    # Get parameter ranges
    k_range = plm.partialIDparam_minmax.get("k", [-2, 2])
    coef_range = plm.partialIDparam_minmax.get("coef_P_D_given_XZ", [-15000, 15000])

    # Generate grid values
    k_min, k_max = k_range
    coef_min, coef_max = coef_range

    k_grid = [k_min + p * (k_max - k_min) for p in ptiles]
    coef_grid = [coef_min + p * (coef_max - coef_min) for p in ptiles]

    results = []

    # Helper function to compute row
    def compute_row(scenario: str, k: float, coef: float) -> Dict:
        if verbose:
            print(f"  Computing: {scenario} (k={k:.2f}, coef={coef:.0f})")

        estimate = bias_adjustment(plm, k, coef)

        # Bootstrap for SE and CI
        boot_ests = _bootstrap_estimates_quiet(plm, k, coef, n_boot)
        se = np.std(boot_ests, ddof=1)
        ci_low = np.percentile(boot_ests, 100 * alpha / 2)
        ci_high = np.percentile(boot_ests, 100 * (1 - alpha / 2))

        return {
            "Scenario": scenario,
            "k": k,
            "coef_P_D_given_XZ": coef,
            "Estimate": estimate,
            "Std. Error": se,
            "CI Low": ci_low,
            "CI High": ci_high,
        }

    # Add benchmark rows
    if include_benchmarks:
        if verbose:
            print("Computing benchmark estimates...")

        # No Unobserved Confounding (k=0)
        results.append(compute_row("No Unobserved Confounding", 0, 0))

        # DID equivalent (k = beta_Y/beta_N if beta_N != 0)
        if plm.beta_N_D_given_X != 0:
            m_did = plm.beta_Y_D_given_X / plm.beta_N_D_given_X
            results.append(compute_row(f"DID (m=1)", m_did, 0))

        # Perfect Placebo (k=1)
        results.append(compute_row("Perfect Placebo, k=1", 1, 0))

    # Add grid rows
    if verbose:
        print("Computing grid estimates...")

    for k in k_grid:
        for coef in coef_grid:
            results.append(compute_row("Grid", k, coef))

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Format for display
    df["k"] = df["k"].round(5)
    df["Estimate"] = df["Estimate"].round(3)
    df["Std. Error"] = df["Std. Error"].round(3)
    df["CI Low"] = df["CI Low"].round(3)
    df["CI High"] = df["CI High"].round(3)

    return df


def _bootstrap_estimates_quiet(
    plm: PlaceboLMResult,
    k: float,
    coef_P_D_given_XZ: float,
    n_boot: int,
) -> np.ndarray:
    """Bootstrap estimates without printing."""
    n = len(plm.data)
    boot_estimates = np.zeros(n_boot)

    for b in range(n_boot):
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_data = plm.data.iloc[boot_idx].reset_index(drop=True)

        try:
            with suppress_print():
                boot_plm = placebo_lm(
                    data=boot_data,
                    outcome=plm.outcome,
                    treatment=plm.treatment,
                    placebo_outcome=plm.placebo_outcome,
                    placebo_treatment=plm.placebo_treatment,
                    observed_covariates=plm.observed_covariates,
                    partialIDparam_minmax=plm.partialIDparam_minmax,
                )
            boot_estimates[b] = bias_adjustment(boot_plm, k, coef_P_D_given_XZ)
        except Exception:
            boot_estimates[b] = np.nan

    return boot_estimates[~np.isnan(boot_estimates)]


def placebo_lm_bounds(
    plm: PlaceboLMResult,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute bounds on the treatment effect over the parameter space.

    This function finds the minimum and maximum bias-adjusted estimates
    over the specified ranges of k and coef_P_D_given_XZ, providing
    partial identification bounds.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    n_boot : int, optional
        Number of bootstrap iterations.
    alpha : float, optional
        Significance level.
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Table with bounds on the treatment effect.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get parameter ranges
    k_range = plm.partialIDparam_minmax.get("k", [-2, 2])
    coef_range = plm.partialIDparam_minmax.get("coef_P_D_given_XZ", [-15000, 15000])

    # Find corners of parameter space
    corners = [
        (k_range[0], coef_range[0]),
        (k_range[0], coef_range[1]),
        (k_range[1], coef_range[0]),
        (k_range[1], coef_range[1]),
    ]

    # For linear bias adjustment, extremes are at corners
    estimates = [bias_adjustment(plm, k, c) for k, c in corners]

    min_est = min(estimates)
    max_est = max(estimates)

    # Bootstrap for uncertainty
    boot_mins = []
    boot_maxs = []
    n = len(plm.data)

    for b in range(n_boot):
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_data = plm.data.iloc[boot_idx].reset_index(drop=True)

        try:
            with suppress_print():
                boot_plm = placebo_lm(
                    data=boot_data,
                    outcome=plm.outcome,
                    treatment=plm.treatment,
                    placebo_outcome=plm.placebo_outcome,
                    placebo_treatment=plm.placebo_treatment,
                    observed_covariates=plm.observed_covariates,
                    partialIDparam_minmax=plm.partialIDparam_minmax,
                )
            boot_ests = [bias_adjustment(boot_plm, k, c) for k, c in corners]
            boot_mins.append(min(boot_ests))
            boot_maxs.append(max(boot_ests))
        except Exception:
            pass

    boot_mins = np.array(boot_mins)
    boot_maxs = np.array(boot_maxs)

    return pd.DataFrame({
        "Bound": ["Lower", "Upper"],
        "Estimate": [min_est, max_est],
        "CI Low": [
            np.percentile(boot_mins, 100 * alpha / 2),
            np.percentile(boot_maxs, 100 * alpha / 2),
        ],
        "CI High": [
            np.percentile(boot_mins, 100 * (1 - alpha / 2)),
            np.percentile(boot_maxs, 100 * (1 - alpha / 2)),
        ],
    })


def sensitivity_table(
    plm: PlaceboLMResult,
    k_values: Optional[List[float]] = None,
    coef_values: Optional[List[float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a sensitivity analysis table for specific parameter values.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k_values : List[float], optional
        Specific k values to evaluate. If None, uses [-1, 0, 0.5, 1, 2].
    coef_values : List[float], optional
        Specific coef_P_D_given_XZ values. If None, uses [0].
    n_boot : int, optional
        Number of bootstrap iterations.
    alpha : float, optional
        Significance level.
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Sensitivity table with estimates and CIs.
    """
    if k_values is None:
        k_values = [-1, 0, 0.5, 1, 2]

    if coef_values is None:
        coef_values = [0]

    if seed is not None:
        np.random.seed(seed)

    results = []

    for k in k_values:
        for coef in coef_values:
            estimate = bias_adjustment(plm, k, coef)
            boot_ests = _bootstrap_estimates_quiet(plm, k, coef, n_boot)
            se = np.std(boot_ests, ddof=1)
            ci_low = np.percentile(boot_ests, 100 * alpha / 2)
            ci_high = np.percentile(boot_ests, 100 * (1 - alpha / 2))

            results.append({
                "k": k,
                "coef_P_D_given_XZ": coef,
                "Estimate": round(estimate, 3),
                "Std. Error": round(se, 3),
                "CI Low": round(ci_low, 3),
                "CI High": round(ci_high, 3),
                "Significant": "Yes" if ci_low > 0 or ci_high < 0 else "No",
            })

    return pd.DataFrame(results)
