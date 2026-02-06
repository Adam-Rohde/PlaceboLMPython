"""
Utility functions for PlaceboLM.

This module provides helper functions for data preparation, diagnostics,
and sensitivity analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union, Any
from scipy import stats

# Try to import statsmodels, fall back to numpy implementation
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    from .core import add_constant, fit_ols


def partial_r2(
    data: pd.DataFrame,
    y_var: str,
    x_var: str,
    control_vars: Optional[List[str]] = None,
) -> float:
    """
    Compute the partial R-squared of x_var on y_var controlling for control_vars.

    The partial R-squared measures the proportion of residual variance in y
    explained by x, after removing the linear influence of control variables.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    y_var : str
        Outcome variable.
    x_var : str
        Variable of interest.
    control_vars : List[str], optional
        Variables to control for.

    Returns
    -------
    float
        Partial R-squared value between 0 and 1.
    """
    if control_vars is None:
        control_vars = []

    data_clean = data[[y_var, x_var] + control_vars].dropna()
    y = data_clean[y_var].values

    if HAS_STATSMODELS:
        if control_vars:
            X_reduced = sm.add_constant(data_clean[control_vars])
            model_reduced = sm.OLS(y, X_reduced).fit()
            ssr_reduced = model_reduced.ssr

            X_full = sm.add_constant(data_clean[control_vars + [x_var]])
            model_full = sm.OLS(y, X_full).fit()
            ssr_full = model_full.ssr
        else:
            X = sm.add_constant(data_clean[[x_var]])
            model = sm.OLS(y, X).fit()
            return model.rsquared
    else:
        if control_vars:
            X_reduced = add_constant(data_clean[control_vars].values)
            var_names_reduced = ["const"] + control_vars
            model_reduced = fit_ols(y, X_reduced, var_names_reduced)
            ssr_reduced = model_reduced.ssr

            X_full = add_constant(data_clean[control_vars + [x_var]].values)
            var_names_full = ["const"] + control_vars + [x_var]
            model_full = fit_ols(y, X_full, var_names_full)
            ssr_full = model_full.ssr
        else:
            X = add_constant(data_clean[[x_var]].values)
            var_names = ["const", x_var]
            model = fit_ols(y, X, var_names)
            return model.rsquared

    partial_r2_val = (ssr_reduced - ssr_full) / ssr_reduced
    return partial_r2_val


def partial_correlation(
    data: pd.DataFrame,
    y_var: str,
    x_var: str,
    control_vars: Optional[List[str]] = None,
) -> float:
    """
    Compute the partial correlation between y_var and x_var.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    y_var : str
        First variable.
    x_var : str
        Second variable.
    control_vars : List[str], optional
        Variables to control for.

    Returns
    -------
    float
        Partial correlation coefficient between -1 and 1.
    """
    pr2 = partial_r2(data, y_var, x_var, control_vars)

    if control_vars is None:
        control_vars = []

    data_clean = data[[y_var, x_var] + control_vars].dropna()
    y = data_clean[y_var].values

    if HAS_STATSMODELS:
        X = sm.add_constant(data_clean[control_vars + [x_var]] if control_vars else data_clean[[x_var]])
        model = sm.OLS(y, X).fit()
        sign = np.sign(model.params[x_var])
    else:
        X_cols = control_vars + [x_var] if control_vars else [x_var]
        X = add_constant(data_clean[X_cols].values)
        var_names = ["const"] + X_cols
        model = fit_ols(y, X, var_names)
        sign = np.sign(model.params[x_var])

    return sign * np.sqrt(pr2)


def robustness_value(
    plm: 'PlaceboLMResult',
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute the robustness value for the treatment effect.

    The robustness value (RV) quantifies how strong confounding would
    need to be to explain away the effect, following Cinelli and Hazlett (2020).

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    alpha : float, optional
        Significance level for RV_alpha.

    Returns
    -------
    Tuple[float, float]
        (RV, RV_alpha) where RV is for point estimate = 0,
        and RV_alpha is for CI to include 0.
    """
    t_stat = plm.beta_Y_D_given_X / plm.se_Y_D_given_X
    df = plm.n_obs - len(plm.observed_covariates) - 2

    rv = t_stat**2 / (t_stat**2 + df)

    t_crit = stats.t.ppf(1 - alpha/2, df)
    if abs(t_stat) > t_crit:
        rv_alpha = (abs(t_stat) - t_crit)**2 / ((abs(t_stat) - t_crit)**2 + df)
    else:
        rv_alpha = 0

    return rv, rv_alpha


def sensitivity_stats(
    plm: 'PlaceboLMResult',
    benchmark_covariate: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute sensitivity statistics for the placebo analysis.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    benchmark_covariate : str, optional
        Covariate to use as benchmark for confounder strength.

    Returns
    -------
    Dict[str, float]
        Dictionary with sensitivity statistics.
    """
    stats_dict = {
        "beta_Y_D_given_X": plm.beta_Y_D_given_X,
        "beta_N_D_given_X": plm.beta_N_D_given_X,
        "se_Y_D_given_X": plm.se_Y_D_given_X,
        "se_N_D_given_X": plm.se_N_D_given_X,
        "t_stat_Y": plm.beta_Y_D_given_X / plm.se_Y_D_given_X,
        "t_stat_N": plm.beta_N_D_given_X / plm.se_N_D_given_X if plm.se_N_D_given_X > 0 else np.nan,
        "n_obs": plm.n_obs,
    }

    if plm.beta_N_D_given_X != 0:
        stats_dict["m_parameter"] = plm.beta_Y_D_given_X / plm.beta_N_D_given_X
    else:
        stats_dict["m_parameter"] = np.inf

    rv, rv_alpha = robustness_value(plm)
    stats_dict["rv"] = rv
    stats_dict["rv_alpha"] = rv_alpha

    if benchmark_covariate is not None and benchmark_covariate in plm.observed_covariates:
        other_covs = [c for c in plm.observed_covariates if c != benchmark_covariate]
        pr2_Y = partial_r2(plm.data, plm.outcome, benchmark_covariate, [plm.treatment] + other_covs)
        pr2_D = partial_r2(plm.data, plm.treatment, benchmark_covariate, other_covs)

        stats_dict[f"partial_r2_{benchmark_covariate}_Y"] = pr2_Y
        stats_dict[f"partial_r2_{benchmark_covariate}_D"] = pr2_D

    return stats_dict


def check_placebo_validity(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    placebo_outcome: str,
    observed_covariates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Diagnostic checks for placebo validity assumptions.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    outcome : str
        Outcome variable.
    treatment : str
        Treatment variable.
    placebo_outcome : str
        Placebo outcome variable.
    observed_covariates : List[str], optional
        Observed covariates.

    Returns
    -------
    Dict[str, Any]
        Dictionary with diagnostic results.
    """
    if observed_covariates is None:
        observed_covariates = []

    diagnostics = {}

    data_clean = data[[outcome, treatment, placebo_outcome] + observed_covariates].dropna()
    y_placebo = data_clean[placebo_outcome].values

    X_cols = [treatment] + observed_covariates
    if HAS_STATSMODELS:
        X = sm.add_constant(data_clean[X_cols])
        model = sm.OLS(y_placebo, X).fit()
        diagnostics["placebo_coef"] = model.params[treatment]
        diagnostics["placebo_se"] = model.bse[treatment]
        diagnostics["placebo_pvalue"] = model.pvalues[treatment]
    else:
        X = add_constant(data_clean[X_cols].values)
        var_names = ["const"] + X_cols
        model = fit_ols(y_placebo, X, var_names)
        diagnostics["placebo_coef"] = model.params[treatment]
        diagnostics["placebo_se"] = model.bse[treatment]
        # Compute p-value manually
        t_stat = diagnostics["placebo_coef"] / diagnostics["placebo_se"]
        df = len(y_placebo) - len(var_names)
        diagnostics["placebo_pvalue"] = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    diagnostics["placebo_significant"] = diagnostics["placebo_pvalue"] < 0.05
    diagnostics["outcome_placebo_corr"] = data_clean[outcome].corr(data_clean[placebo_outcome])

    if observed_covariates:
        balance_stats = {}
        for cov in observed_covariates:
            treated = data_clean[data_clean[treatment] == 1][cov]
            control = data_clean[data_clean[treatment] == 0][cov]

            pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
            if pooled_sd > 0:
                std_diff = (treated.mean() - control.mean()) / pooled_sd
            else:
                std_diff = 0

            balance_stats[cov] = {
                "treated_mean": treated.mean(),
                "control_mean": control.mean(),
                "std_diff": std_diff,
            }

        diagnostics["covariate_balance"] = balance_stats

    messages = []
    if diagnostics["placebo_significant"]:
        messages.append(
            f"WARNING: Treatment has significant effect on placebo outcome "
            f"(coef={diagnostics['placebo_coef']:.3f}, p={diagnostics['placebo_pvalue']:.4f}). "
            f"May indicate imperfect placebo."
        )
    else:
        messages.append(
            f"Treatment effect on placebo is not significant "
            f"(coef={diagnostics['placebo_coef']:.3f}, p={diagnostics['placebo_pvalue']:.4f})."
        )

    diagnostics["messages"] = messages
    return diagnostics


def simulate_placebo_data(
    n: int = 1000,
    true_effect: float = 1.0,
    confounding_strength: float = 0.5,
    placebo_imperfection: float = 0.0,
    noise_sd: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate data for testing PlaceboLM.

    Parameters
    ----------
    n : int, optional
        Sample size (default 1000).
    true_effect : float, optional
        True causal effect of D on Y (default 1.0).
    confounding_strength : float, optional
        Effect of confounder on both D and Y (default 0.5).
    placebo_imperfection : float, optional
        Direct effect of D on N (default 0.0, meaning perfect placebo).
    noise_sd : float, optional
        Standard deviation of noise (default 1.0).
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Simulated dataset with columns: Y, D, N, U, X1, X2.
    """
    if seed is not None:
        np.random.seed(seed)

    U = np.random.randn(n)
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)

    D_latent = confounding_strength * U + 0.5 * X1 + np.random.randn(n) * noise_sd
    D = (D_latent > 0).astype(int)

    N = confounding_strength * U + 0.5 * X2 + placebo_imperfection * D + np.random.randn(n) * noise_sd
    Y = true_effect * D + confounding_strength * U + 0.5 * X1 + np.random.randn(n) * noise_sd

    return pd.DataFrame({
        "Y": Y,
        "D": D,
        "N": N,
        "U": U,
        "X1": X1,
        "X2": X2,
    })


def monte_carlo_coverage(
    n_sim: int = 500,
    n: int = 500,
    true_effect: float = 1.0,
    confounding_strength: float = 0.5,
    k_assumed: float = 1.0,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Monte Carlo simulation to assess coverage of PlaceboLM confidence intervals.

    Parameters
    ----------
    n_sim : int, optional
        Number of simulations (default 500).
    n : int, optional
        Sample size per simulation (default 500).
    true_effect : float, optional
        True treatment effect (default 1.0).
    confounding_strength : float, optional
        Confounding strength (default 0.5).
    k_assumed : float, optional
        Value of k assumed in analysis (default 1.0).
    alpha : float, optional
        Significance level (default 0.05).
    seed : int, optional
        Random seed.

    Returns
    -------
    Dict[str, float]
        Dictionary with coverage, bias, and RMSE statistics.
    """
    from .core import placebo_lm, bias_adjustment

    if seed is not None:
        np.random.seed(seed)

    estimates = []
    covers = []

    for _ in range(n_sim):
        df = simulate_placebo_data(
            n=n,
            true_effect=true_effect,
            confounding_strength=confounding_strength,
            placebo_imperfection=0.0,
        )

        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            plm = placebo_lm(
                data=df,
                outcome="Y",
                treatment="D",
                placebo_outcome="N",
                observed_covariates=["X1", "X2"],
            )
        finally:
            sys.stdout = old_stdout

        est = bias_adjustment(plm, k=k_assumed, coef_P_D_given_XZ=0)
        estimates.append(est)

        se = np.sqrt(plm.se_Y_D_given_X**2 + k_assumed**2 * plm.se_N_D_given_X**2)
        z = stats.norm.ppf(1 - alpha/2)
        ci_low = est - z * se
        ci_high = est + z * se

        covers.append(ci_low <= true_effect <= ci_high)

    estimates = np.array(estimates)

    return {
        "coverage": np.mean(covers),
        "bias": np.mean(estimates) - true_effect,
        "rmse": np.sqrt(np.mean((estimates - true_effect)**2)),
        "mean_estimate": np.mean(estimates),
        "std_estimate": np.std(estimates),
    }


def format_table(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    """Format a DataFrame as a nice string table."""
    return df.to_string(float_format=lambda x: f"{x:{float_fmt}}")
