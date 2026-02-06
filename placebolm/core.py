"""
PlaceboLM: Causal Progress with Imperfect Placebos

A Python implementation of the methods described in:
Rohde and Hazlett (2023) "Causal progress with imperfect placebo treatments and outcomes"

This module provides tools for partial identification of causal effects using
placebo outcomes and treatments, allowing for imperfect placebos and unequal confounding.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Any
from scipy import stats
import warnings

# Try to import statsmodels, fall back to numpy implementation
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class SimpleOLSResult:
    """Simple OLS result class when statsmodels is not available."""

    def __init__(self, y: np.ndarray, X: np.ndarray, var_names: List[str]):
        """Fit OLS via normal equations."""
        self.n, self.k = X.shape
        self.var_names = var_names

        # OLS: (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y

        try:
            self.params_arr = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self.params_arr = np.linalg.lstsq(X, y, rcond=None)[0]

        # Residuals and MSE
        self.resid = y - X @ self.params_arr
        self.df_resid = self.n - self.k
        self.ssr = np.sum(self.resid**2)
        self.mse_resid = self.ssr / self.df_resid

        # Standard errors
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        self.cov_params = self.mse_resid * XtX_inv
        self.bse_arr = np.sqrt(np.diag(self.cov_params))

        # Create named parameter access
        self.params = {name: self.params_arr[i] for i, name in enumerate(var_names)}
        self.bse = {name: self.bse_arr[i] for i, name in enumerate(var_names)}

        # R-squared
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean)**2)
        self.rsquared = 1 - self.ssr / ss_tot if ss_tot > 0 else 0


def add_constant(X: np.ndarray) -> np.ndarray:
    """Add constant column to design matrix."""
    n = X.shape[0]
    return np.column_stack([np.ones(n), X])


class StatsmodelsOLSWrapper:
    """Wrapper around statsmodels OLS result to provide dict-based param access."""

    def __init__(self, result, var_names: List[str]):
        self._result = result
        self.var_names = var_names
        self.params = {name: result.params[i] for i, name in enumerate(var_names)}
        self.bse = {name: result.bse[i] for i, name in enumerate(var_names)}
        self.mse_resid = result.mse_resid
        self.resid = result.resid
        self.ssr = result.ssr
        self.rsquared = result.rsquared
        self.df_resid = result.df_resid


def fit_ols(y: np.ndarray, X: np.ndarray, var_names: List[str]):
    """Fit OLS using statsmodels if available, else numpy."""
    if HAS_STATSMODELS:
        model = sm.OLS(y, X).fit()
        return StatsmodelsOLSWrapper(model, var_names)
    else:
        return SimpleOLSResult(y, X, var_names)


@dataclass
class PlaceboLMResult:
    """Container for PlaceboLM analysis results.

    Attributes
    ----------
    data : pd.DataFrame
        The dataset used for analysis.
    outcome : str
        Name of the outcome variable.
    treatment : str
        Name of the treatment variable.
    placebo_outcome : str
        Name of the placebo outcome variable.
    placebo_treatment : str
        Name of the placebo treatment variable (if any).
    observed_covariates : List[str]
        List of observed covariates included in the model.
    placebo_type : str
        Description of the placebo type used.
    reg1 : Any
        Results from the main outcome regression.
    reg2 : Any
        Results from the placebo outcome/treatment regression.
    beta_Y_D_given_X : float
        Coefficient of treatment on outcome given covariates.
    beta_N_D_given_X : float
        Coefficient of treatment on placebo outcome given covariates.
    se_Y_D_given_X : float
        Standard error of beta_Y_D_given_X.
    se_N_D_given_X : float
        Standard error of beta_N_D_given_X.
    residual_sd_Y : float
        Residual standard deviation from outcome regression.
    residual_sd_N : float
        Residual standard deviation from placebo regression.
    n_obs : int
        Number of observations.
    partialIDparam_minmax : Dict
        Dictionary with ranges for partial identification parameters.
    """
    data: pd.DataFrame
    outcome: str
    treatment: str
    placebo_outcome: str
    placebo_treatment: str
    observed_covariates: List[str]
    placebo_type: str
    reg1: Any
    reg2: Any
    beta_Y_D_given_X: float
    beta_N_D_given_X: float
    se_Y_D_given_X: float
    se_N_D_given_X: float
    residual_sd_Y: float
    residual_sd_N: float
    n_obs: int
    partialIDparam_minmax: Dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"PlaceboLMResult(\n"
            f"  placebo_type='{self.placebo_type}',\n"
            f"  n_obs={self.n_obs},\n"
            f"  beta_Y_D_given_X={self.beta_Y_D_given_X:.4f},\n"
            f"  beta_N_D_given_X={self.beta_N_D_given_X:.4f}\n"
            f")"
        )


def placebo_lm(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    placebo_outcome: str = "",
    placebo_treatment: str = "",
    observed_covariates: Optional[List[str]] = None,
    dp: str = "",
    py: str = "",
    partialIDparam_minmax: Optional[Dict] = None,
    placebo_data: Optional[pd.DataFrame] = None,
) -> PlaceboLMResult:
    """
    Fit a PlaceboLM model for partial identification of causal effects.

    This function implements the omitted variable bias framework for making
    causal progress with imperfect placebos, as described in Rohde and Hazlett (2023).

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing treatment, outcome, and covariates.
    outcome : str
        Name of the outcome variable (Y).
    treatment : str
        Name of the treatment variable (D).
    placebo_outcome : str, optional
        Name of the placebo outcome variable (N). This is a variable that should
        not be affected by the treatment (e.g., pre-treatment outcome in DID).
    placebo_treatment : str, optional
        Name of the placebo treatment variable (P). This is a variable that
        affects the outcome similarly to confounders but has no direct causal effect.
    observed_covariates : List[str], optional
        List of observed covariates to include in the model.
    dp : str, optional
        Name of variable representing direct effect of D on placebo outcome
        (for imperfect placebo outcome).
    py : str, optional
        Name of variable representing direct effect of placebo treatment on Y
        (for imperfect placebo treatment).
    partialIDparam_minmax : Dict, optional
        Dictionary specifying ranges for partial identification parameters:
        - 'k': Range for the ratio of confounding (default [-2, 2])
        - 'coef_P_D_given_XZ': Range for direct effect of treatment on placebo
    placebo_data : pd.DataFrame, optional
        Separate dataset for placebo analysis (if different from main data).

    Returns
    -------
    PlaceboLMResult
        Object containing regression results and parameters needed for
        bias adjustment and partial identification.

    Notes
    -----
    The key parameters for sensitivity analysis are:

    - k: The ratio of confounding in the placebo relationship relative to the
      real treatment-outcome relationship. k=1 implies "equiconfounding"
      (the parallel trends assumption in DID). k=0 implies no confounding
      in the observed treatment-outcome relationship.

    - coef_P_D_given_XZ: The direct effect of treatment on the placebo outcome.
      For a "perfect" placebo, this is zero. Non-zero values allow for
      "imperfect" placebos.

    Examples
    --------
    >>> import pandas as pd
    >>> from placebolm import placebo_lm
    >>>
    >>> # Load data
    >>> df = pd.read_csv("lalonde.csv")
    >>>
    >>> # Fit PlaceboLM with pre-treatment earnings as placebo outcome
    >>> result = placebo_lm(
    ...     data=df,
    ...     outcome="re78",
    ...     treatment="treat",
    ...     placebo_outcome="re74",
    ...     observed_covariates=["age", "education", "black", "hispanic", "married", "nodegree"],
    ...     partialIDparam_minmax={"k": [-2, 2], "coef_P_D_given_XZ": [-15000, 15000]}
    ... )
    """
    # Input validation
    if observed_covariates is None:
        observed_covariates = []

    if partialIDparam_minmax is None:
        partialIDparam_minmax = {"k": [-2, 2], "coef_P_D_given_XZ": [-15000, 15000]}

    # Use placebo_data if provided, otherwise use main data
    analysis_data = placebo_data if placebo_data is not None else data.copy()

    # Drop missing values for relevant columns
    relevant_cols = [outcome, treatment] + observed_covariates
    if placebo_outcome:
        relevant_cols.append(placebo_outcome)
    if placebo_treatment:
        relevant_cols.append(placebo_treatment)

    analysis_data = analysis_data.dropna(subset=relevant_cols)

    # Determine placebo type
    has_placebo_outcome = bool(placebo_outcome)
    has_placebo_treatment = bool(placebo_treatment)
    has_dp = bool(dp)
    has_py = bool(py)

    if has_placebo_outcome and not has_placebo_treatment:
        if has_dp:
            placebo_type = "Single Placebo, Imperfect Placebo Outcome (D->N effect)"
        else:
            placebo_type = "Single Placebo, No Direct Relationships, Placebo Outcome"
            print("Placebo assumed to have no direct relationship with either treatment or outcome.")
    elif has_placebo_treatment and not has_placebo_outcome:
        if has_py:
            placebo_type = "Single Placebo, Imperfect Placebo Treatment (P->Y effect)"
        else:
            placebo_type = "Single Placebo, No Direct Relationships, Placebo Treatment"
    elif has_placebo_outcome and has_placebo_treatment:
        placebo_type = "Double Placebo"
    else:
        raise ValueError("Must specify at least placebo_outcome or placebo_treatment")

    print(f"\nPlacebo Type: {placebo_type}\n")

    # Build regression formulas
    covariate_str = " + ".join(observed_covariates) if observed_covariates else ""

    # Regression 1: Y ~ D + X
    if covariate_str:
        formula1 = f"{outcome} ~ {treatment} + {covariate_str}"
    else:
        formula1 = f"{outcome} ~ {treatment}"

    print(f"Regression 1: {formula1}")

    # Fit regression 1
    y1 = analysis_data[outcome].values
    X1_vars = [treatment] + observed_covariates
    X1_data = analysis_data[X1_vars].values
    X1 = add_constant(X1_data)
    var_names1 = ["const"] + X1_vars
    reg1 = fit_ols(y1, X1, var_names1)

    # Regression 2: N ~ D + X (for placebo outcome)
    if has_placebo_outcome:
        if covariate_str:
            formula2 = f"{placebo_outcome} ~ {treatment} + {covariate_str}"
        else:
            formula2 = f"{placebo_outcome} ~ {treatment}"

        print(f"Regression 2: {formula2}\n")

        y2 = analysis_data[placebo_outcome].values
        X2 = add_constant(analysis_data[X1_vars].values)
        reg2 = fit_ols(y2, X2, var_names1)

        beta_N_D_given_X = reg2.params[treatment]
        se_N_D_given_X = reg2.bse[treatment]
        residual_sd_N = np.sqrt(reg2.mse_resid)
    elif has_placebo_treatment:
        # For placebo treatment: Y ~ P + X
        if covariate_str:
            formula2 = f"{outcome} ~ {placebo_treatment} + {covariate_str}"
        else:
            formula2 = f"{outcome} ~ {placebo_treatment}"

        print(f"Regression 2: {formula2}\n")

        y2 = analysis_data[outcome].values
        X2_vars = [placebo_treatment] + observed_covariates
        X2 = add_constant(analysis_data[X2_vars].values)
        var_names2 = ["const"] + X2_vars
        reg2 = fit_ols(y2, X2, var_names2)

        beta_N_D_given_X = reg2.params[placebo_treatment]
        se_N_D_given_X = reg2.bse[placebo_treatment]
        residual_sd_N = np.sqrt(reg2.mse_resid)
    else:
        reg2 = None
        beta_N_D_given_X = 0.0
        se_N_D_given_X = 0.0
        residual_sd_N = 0.0

    # Extract key quantities from regression 1
    beta_Y_D_given_X = reg1.params[treatment]
    se_Y_D_given_X = reg1.bse[treatment]
    residual_sd_Y = np.sqrt(reg1.mse_resid)

    return PlaceboLMResult(
        data=analysis_data,
        outcome=outcome,
        treatment=treatment,
        placebo_outcome=placebo_outcome,
        placebo_treatment=placebo_treatment,
        observed_covariates=observed_covariates,
        placebo_type=placebo_type,
        reg1=reg1,
        reg2=reg2,
        beta_Y_D_given_X=beta_Y_D_given_X,
        beta_N_D_given_X=beta_N_D_given_X,
        se_Y_D_given_X=se_Y_D_given_X,
        se_N_D_given_X=se_N_D_given_X,
        residual_sd_Y=residual_sd_Y,
        residual_sd_N=residual_sd_N,
        n_obs=len(analysis_data),
        partialIDparam_minmax=partialIDparam_minmax,
    )


def bias_adjustment(
    plm: PlaceboLMResult,
    k: float,
    coef_P_D_given_XZ: float = 0.0,
) -> float:
    """
    Calculate the bias-adjusted treatment effect estimate.

    This function implements the bias adjustment formula from the omitted
    variable bias framework, allowing for imperfect placebos (coef_P_D_given_XZ != 0)
    and unequal confounding (k != 1).

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k : float
        The ratio of confounding. k=1 implies equiconfounding (parallel trends
        in DID). k=0 implies no unobserved confounding. Values > 1 imply the
        placebo relationship has more confounding; values < 1 imply less.
    coef_P_D_given_XZ : float, optional
        The direct effect of treatment on the placebo outcome (default 0).
        For a perfect placebo, this is zero.

    Returns
    -------
    float
        The bias-adjusted estimate of the treatment effect.

    Notes
    -----
    The bias adjustment is computed as:

        adjusted = beta_Y_D_given_X - k * (beta_N_D_given_X - coef_P_D_given_XZ)

    where:
    - beta_Y_D_given_X is the observed treatment-outcome coefficient
    - beta_N_D_given_X is the observed treatment-placebo coefficient
    - k scales the bias (k=1 for equiconfounding)
    - coef_P_D_given_XZ allows for direct D->N effects

    When k=1 and coef_P_D_given_XZ=0, this reduces to the standard DID estimator
    (or perfect placebo correction).
    """
    # Bias in placebo relationship
    bias_in_placebo = plm.beta_N_D_given_X - coef_P_D_given_XZ

    # Scaled bias (accounting for unequal confounding)
    scaled_bias = k * bias_in_placebo

    # Adjusted estimate
    adjusted_estimate = plm.beta_Y_D_given_X - scaled_bias

    return adjusted_estimate


def bias_adjustment_r2(
    plm: PlaceboLMResult,
    k: float,
    r2_P_D_given_XZ: float = 0.0,
) -> float:
    """
    Calculate bias-adjusted treatment effect using R-squared parameterization.

    This implements the reparameterization in terms of partial variance explained,
    following Cinelli and Hazlett (2020). This can be more interpretable when
    the placebo and outcome are on different scales.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k : float
        The ratio of confounding on the R-squared scale.
    r2_P_D_given_XZ : float, optional
        The partial R-squared of treatment on placebo outcome (default 0).
        Represents the proportion of variance in the placebo explained by
        the direct treatment effect.

    Returns
    -------
    float
        The bias-adjusted estimate of the treatment effect.
    """
    # Get residual standard deviations
    sd_Y = plm.residual_sd_Y
    sd_N = plm.residual_sd_N

    # Scale factor for bias adjustment
    scale = sd_Y / sd_N if sd_N > 0 else 1.0

    # Compute direct effect on coefficient scale from R-squared
    if r2_P_D_given_XZ > 0:
        # Get variance of treatment residual
        treatment = plm.treatment
        X_vars = plm.observed_covariates
        if X_vars:
            X = add_constant(plm.data[X_vars].values)
            D = plm.data[treatment].values
            var_names = ["const"] + X_vars
            reg_D = fit_ols(D, X, var_names)
            D_resid_var = reg_D.mse_resid
        else:
            D_resid_var = plm.data[treatment].var()

        # Convert R-squared to coefficient
        coef_P_D_given_XZ = np.sqrt(r2_P_D_given_XZ * sd_N**2 / D_resid_var)
    else:
        coef_P_D_given_XZ = 0.0

    return bias_adjustment(plm, k * scale, coef_P_D_given_XZ)


def get_estimate_at_params(
    plm: PlaceboLMResult,
    k: float,
    coef_P_D_given_XZ: float = 0.0,
) -> Tuple[float, float]:
    """
    Get the bias-adjusted estimate and its standard error at given parameters.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k : float
        The ratio of confounding.
    coef_P_D_given_XZ : float, optional
        The direct effect of treatment on placebo outcome.

    Returns
    -------
    Tuple[float, float]
        (estimate, standard_error)
    """
    estimate = bias_adjustment(plm, k, coef_P_D_given_XZ)

    # Standard error via delta method
    # Var(beta_Y - k*beta_N) = Var(beta_Y) + k^2*Var(beta_N) - 2*k*Cov(beta_Y, beta_N)
    # Assuming independence between regressions (conservative):
    se = np.sqrt(plm.se_Y_D_given_X**2 + k**2 * plm.se_N_D_given_X**2)

    return estimate, se


def bootstrap_estimates(
    plm: PlaceboLMResult,
    k: float,
    coef_P_D_given_XZ: float = 0.0,
    n_boot: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Bootstrap the bias-adjusted estimate.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k : float
        The ratio of confounding.
    coef_P_D_given_XZ : float, optional
        The direct effect of treatment on placebo outcome.
    n_boot : int, optional
        Number of bootstrap iterations (default 1000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of bootstrap estimates.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(plm.data)
    boot_estimates = np.zeros(n_boot)

    for b in range(n_boot):
        # Resample data
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_data = plm.data.iloc[boot_idx].reset_index(drop=True)

        # Refit model on bootstrap sample
        try:
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


def bootstrap_se(
    plm: PlaceboLMResult,
    k: float,
    coef_P_D_given_XZ: float = 0.0,
    n_boot: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """
    Calculate bootstrap standard error for the bias-adjusted estimate.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k : float
        The ratio of confounding.
    coef_P_D_given_XZ : float, optional
        The direct effect of treatment on placebo outcome.
    n_boot : int, optional
        Number of bootstrap iterations.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        Bootstrap standard error.
    """
    boot_ests = bootstrap_estimates(plm, k, coef_P_D_given_XZ, n_boot, seed)
    return np.std(boot_ests, ddof=1)


def bootstrap_ci(
    plm: PlaceboLMResult,
    k: float,
    coef_P_D_given_XZ: float = 0.0,
    n_boot: int = 1000,
    alpha: float = 0.05,
    method: str = "percentile",
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for the bias-adjusted estimate.

    Parameters
    ----------
    plm : PlaceboLMResult
        Result from placebo_lm().
    k : float
        The ratio of confounding.
    coef_P_D_given_XZ : float, optional
        The direct effect of treatment on placebo outcome.
    n_boot : int, optional
        Number of bootstrap iterations.
    alpha : float, optional
        Significance level (default 0.05 for 95% CI).
    method : str, optional
        CI method: "percentile" or "normal" (default "percentile").
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) of confidence interval.
    """
    boot_ests = bootstrap_estimates(plm, k, coef_P_D_given_XZ, n_boot, seed)

    if method == "percentile":
        lower = np.percentile(boot_ests, 100 * alpha / 2)
        upper = np.percentile(boot_ests, 100 * (1 - alpha / 2))
    elif method == "normal":
        point_est = bias_adjustment(plm, k, coef_P_D_given_XZ)
        se = np.std(boot_ests, ddof=1)
        z = stats.norm.ppf(1 - alpha / 2)
        lower = point_est - z * se
        upper = point_est + z * se
    else:
        raise ValueError(f"Unknown method: {method}")

    return lower, upper


# Suppress print during bootstrap
import contextlib
import io

@contextlib.contextmanager
def suppress_print():
    """Context manager to suppress print statements."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield
