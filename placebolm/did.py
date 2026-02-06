"""
Difference-in-Differences (DID) specific functionality.

This module provides specialized functions for the DID case, where
pre-treatment outcomes serve as placebo outcomes and the parallel
trends assumption corresponds to equiconfounding (k=1).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union

# Try to import statsmodels, fall back to core implementation
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from .core import (
    PlaceboLMResult,
    placebo_lm,
    bias_adjustment,
    add_constant,
    fit_ols,
)


def did_placebo_lm(
    data: pd.DataFrame,
    outcome_post: str,
    outcome_pre: str,
    treatment: str,
    observed_covariates: Optional[List[str]] = None,
    k_range: Tuple[float, float] = (0, 2),
) -> PlaceboLMResult:
    """
    Fit a PlaceboLM model specifically for difference-in-differences settings.

    In DID, the pre-treatment outcome serves as a placebo outcome. The
    parallel trends assumption is equivalent to assuming equiconfounding
    (k=1) on the additive scale.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing outcomes, treatment, and covariates.
    outcome_post : str
        Name of the post-treatment outcome variable (Y).
    outcome_pre : str
        Name of the pre-treatment outcome variable (N), which serves
        as the placebo outcome.
    treatment : str
        Name of the treatment variable (D).
    observed_covariates : List[str], optional
        List of observed covariates to include.
    k_range : Tuple[float, float], optional
        Range for k parameter (default (0, 2)). k=1 is parallel trends.

    Returns
    -------
    PlaceboLMResult
        Result object configured for DID analysis.

    Notes
    -----
    The DID estimator is equivalent to placeboLM with k=1 and coef_P_D_given_XZ=0.

    Under parallel trends, the pre-treatment difference in outcomes between
    treatment and control groups reflects the same confounding bias as the
    post-treatment difference. Subtracting removes this bias.

    The PlaceboLM framework allows examining what happens when parallel trends
    is relaxed (k != 1).

    Examples
    --------
    >>> # Standard DID setup
    >>> result = did_placebo_lm(
    ...     data=df,
    ...     outcome_post="re78",
    ...     outcome_pre="re75",
    ...     treatment="treat",
    ...     observed_covariates=["age", "education"]
    ... )
    >>>
    >>> # DID estimate (k=1)
    >>> did_estimate = bias_adjustment(result, k=1, coef_P_D_given_XZ=0)
    """
    partialIDparam_minmax = {
        "k": list(k_range),
        "coef_P_D_given_XZ": [0, 0],  # No direct effect assumed in DID
    }

    return placebo_lm(
        data=data,
        outcome=outcome_post,
        treatment=treatment,
        placebo_outcome=outcome_pre,
        observed_covariates=observed_covariates,
        partialIDparam_minmax=partialIDparam_minmax,
    )


def did_estimate(plm: PlaceboLMResult) -> float:
    """
    Compute the standard DID estimate.

    This is equivalent to bias_adjustment with k=1 and coef_P_D_given_XZ=0.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from did_placebo_lm() or placebo_lm().

    Returns
    -------
    float
        The DID estimate.
    """
    return bias_adjustment(plm, k=1, coef_P_D_given_XZ=0)


def parallel_trends_sensitivity(
    plm: PlaceboLMResult,
    k_values: Optional[List[float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Conduct sensitivity analysis for departures from parallel trends.

    This function examines how the treatment effect estimate changes
    as we relax the parallel trends assumption (k != 1).

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from did_placebo_lm() or placebo_lm().
    k_values : List[float], optional
        Values of k to evaluate. Default [0, 0.5, 0.75, 1, 1.25, 1.5, 2].
    n_boot : int, optional
        Number of bootstrap iterations.
    alpha : float, optional
        Significance level.
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Table with estimates and CIs for each k value.

    Notes
    -----
    k < 1: Pre-treatment difference overstates the confounding in the
           post-treatment period (e.g., treatment group "catching up").
    k = 1: Parallel trends holds.
    k > 1: Pre-treatment difference understates the confounding in the
           post-treatment period (e.g., diverging trends).
    """
    if k_values is None:
        k_values = [0, 0.5, 0.75, 1, 1.25, 1.5, 2]

    from .tables import sensitivity_table

    return sensitivity_table(
        plm,
        k_values=k_values,
        coef_values=[0],  # No direct effect in DID
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )


def did_breakdown_k(plm: PlaceboLMResult) -> float:
    """
    Find the value of k where the DID estimate crosses zero.

    This is the "breakdown point" for parallel trends - the minimum
    departure from parallel trends needed to nullify the effect.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from did_placebo_lm() or placebo_lm().

    Returns
    -------
    float
        The value of k where estimate equals zero.
        Returns np.inf if beta_N_D_given_X == 0.
    """
    if plm.beta_N_D_given_X == 0:
        return np.inf

    # Solve: beta_Y - k * beta_N = 0
    # k = beta_Y / beta_N
    return plm.beta_Y_D_given_X / plm.beta_N_D_given_X


def compute_m_parameter(plm: PlaceboLMResult) -> float:
    """
    Compute the m parameter (ratio of effects on outcome vs placebo).

    In the DID context, m represents the ratio of the treatment's
    effect on the real outcome relative to its (spurious) effect
    on the pre-treatment outcome due to confounding.

    When m=1, the effect on the real outcome equals the effect on
    the placebo, implying all the observed effect is due to confounding.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().

    Returns
    -------
    float
        The m parameter.
    """
    if plm.beta_N_D_given_X == 0:
        return np.inf

    return plm.beta_Y_D_given_X / plm.beta_N_D_given_X


def did_comparison(
    data: pd.DataFrame,
    outcome_post: str,
    outcome_pre: str,
    treatment: str,
    observed_covariates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare standard DID with PlaceboLM under different assumptions.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    outcome_post : str
        Post-treatment outcome.
    outcome_pre : str
        Pre-treatment outcome.
    treatment : str
        Treatment variable.
    observed_covariates : List[str], optional
        Covariates.

    Returns
    -------
    pd.DataFrame
        Comparison of estimates under different methods/assumptions.
    """
    if observed_covariates is None:
        observed_covariates = []

    # Fit PlaceboLM
    plm = did_placebo_lm(
        data=data,
        outcome_post=outcome_post,
        outcome_pre=outcome_pre,
        treatment=treatment,
        observed_covariates=observed_covariates,
    )

    # Standard OLS estimate (no adjustment)
    ols_estimate = plm.beta_Y_D_given_X

    # DID estimate (k=1)
    did_est = did_estimate(plm)

    # Change score model: Y_post - Y_pre ~ D + X
    data_copy = data.copy()
    data_copy['_change'] = data_copy[outcome_post] - data_copy[outcome_pre]

    y_change = data_copy['_change'].values
    X_vars = [treatment] + observed_covariates

    if HAS_STATSMODELS:
        X = sm.add_constant(data_copy[X_vars])
        change_model = sm.OLS(y_change, X).fit()
        change_estimate = change_model.params[treatment]
    else:
        X = add_constant(data_copy[X_vars].values)
        var_names = ["const"] + X_vars
        change_model = fit_ols(y_change, X, var_names)
        change_estimate = change_model.params[treatment]

    # k=0 (no confounding)
    no_confounding = bias_adjustment(plm, k=0, coef_P_D_given_XZ=0)

    # Breakdown k
    breakdown_k = did_breakdown_k(plm)

    results = {
        "Method": [
            "OLS (no adjustment)",
            "DID / Change Score (k=1)",
            "PlaceboLM k=0 (no confounding)",
            f"PlaceboLM k={breakdown_k:.2f} (zero effect)",
        ],
        "Estimate": [
            round(ols_estimate, 3),
            round(did_est, 3),
            round(no_confounding, 3),
            0,
        ],
        "Interpretation": [
            "Assumes no confounding",
            "Assumes parallel trends",
            "Lower bound if confounding increases effect",
            f"Breakdown point for parallel trends",
        ],
    }

    return pd.DataFrame(results)


def create_did_panel(
    data: pd.DataFrame,
    unit_id: str,
    time_var: str,
    outcome: str,
    treatment: str,
    pre_period: Union[int, str],
    post_period: Union[int, str],
    covariates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert long panel data to wide format for DID analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    unit_id : str
        Column identifying units (individuals, firms, etc.).
    time_var : str
        Column identifying time periods.
    outcome : str
        Outcome variable.
    treatment : str
        Treatment indicator (should be constant within unit).
    pre_period : int or str
        Value of time_var for pre-treatment period.
    post_period : int or str
        Value of time_var for post-treatment period.
    covariates : List[str], optional
        Time-invariant covariates.

    Returns
    -------
    pd.DataFrame
        Wide-format data suitable for did_placebo_lm().
    """
    if covariates is None:
        covariates = []

    # Get pre and post observations
    pre_data = data[data[time_var] == pre_period][[unit_id, outcome]].copy()
    pre_data = pre_data.rename(columns={outcome: f"{outcome}_pre"})

    post_data = data[data[time_var] == post_period][[unit_id, outcome]].copy()
    post_data = post_data.rename(columns={outcome: f"{outcome}_post"})

    # Merge
    wide_data = pre_data.merge(post_data, on=unit_id)

    # Add treatment and covariates (from either period)
    treat_data = data[data[time_var] == pre_period][[unit_id, treatment] + covariates].drop_duplicates()
    wide_data = wide_data.merge(treat_data, on=unit_id)

    return wide_data
