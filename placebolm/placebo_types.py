"""
Comprehensive implementation of all placebo types from Tables 1 and 2.

This module implements the full taxonomy of single placebo graphs from
Rohde & Hazlett (2023), Appendix Tables 1 and 2:

Table 1:
    [a] Placebo Outcome / Placebo Treatment (basic cases)
    [b] Imperfect Placebo Outcome (D -> P edge)
    [c] Imperfect Placebo Treatment (P -> Y edge) / Observed Confounder 1
    [d] Mediator (D -> P -> Y) - not recommended

Table 2:
    [e/f] Observed Confounder 2 (P -> D structure)
    [g/h] Post-Outcome (Y -> P structure)

Appendix B: Double Placebos (both P and N available)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from scipy import stats

# Import from core
from .core import (
    add_constant, fit_ols, HAS_STATSMODELS,
    PlaceboLMResult
)


@dataclass
class PlaceboLMResultExtended(PlaceboLMResult):
    """Extended result class with additional fields for different placebo types."""

    # Additional regression results for complex cases
    reg3: Any = None

    # Scale factors for different placebo types
    scale_factor: float = 1.0
    scale_factor_formula: str = "sd(Y⊥D,X) / sd(N⊥D,X)"

    # Coefficients for placebo treatment cases
    beta_Y_P_given_DX: float = 0.0
    se_Y_P_given_DX: float = 0.0
    beta_Y_D_given_PX: float = 0.0
    se_Y_D_given_PX: float = 0.0

    # Additional residual SDs
    residual_sd_D_given_PX: float = 0.0
    residual_sd_P_given_DX: float = 0.0
    residual_sd_Y_given_DPX: float = 0.0


def compute_scale_factor_placebo_outcome(
    residual_sd_Y: float,
    residual_sd_N: float,
) -> float:
    """
    Compute scale factor for placebo outcome case (Table 1[a]).

    SF = sd(Y⊥D,X) / sd(N⊥D,X)

    This is the ratio of residual standard deviations from:
    - Regression of Y on D and X
    - Regression of N on D and X
    """
    if residual_sd_N == 0:
        return np.inf
    return residual_sd_Y / residual_sd_N


def compute_scale_factor_placebo_treatment(
    data: pd.DataFrame,
    treatment: str,
    placebo_treatment: str,
    observed_covariates: List[str],
) -> Tuple[float, float, float]:
    """
    Compute scale factor for placebo treatment case (Table 1[a]).

    SF = sd(P⊥D,X) / sd(D⊥P,X)

    This requires regressing:
    - P on D and X to get residuals
    - D on P and X to get residuals

    Returns
    -------
    scale_factor : float
    residual_sd_P_given_DX : float
    residual_sd_D_given_PX : float
    """
    X_vars = observed_covariates if observed_covariates else []

    # Regression: P ~ D + X
    y_P = data[placebo_treatment].values
    X_P = add_constant(data[[treatment] + X_vars].values)
    var_names_P = ["const", treatment] + X_vars
    reg_P = fit_ols(y_P, X_P, var_names_P)
    residual_sd_P_given_DX = np.sqrt(reg_P.mse_resid)

    # Regression: D ~ P + X
    y_D = data[treatment].values
    X_D = add_constant(data[[placebo_treatment] + X_vars].values)
    var_names_D = ["const", placebo_treatment] + X_vars
    reg_D = fit_ols(y_D, X_D, var_names_D)
    residual_sd_D_given_PX = np.sqrt(reg_D.mse_resid)

    if residual_sd_D_given_PX == 0:
        scale_factor = np.inf
    else:
        scale_factor = residual_sd_P_given_DX / residual_sd_D_given_PX

    return scale_factor, residual_sd_P_given_DX, residual_sd_D_given_PX


def compute_scale_factor_observed_confounder_1(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    placebo: str,
    observed_covariates: List[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute scale factor for Observed Confounder 1 case (Table 1[c]).

    SF = [sd(Y⊥D,P,X) / sd(D⊥P,X)] x [sd(D⊥X) / sd(P⊥D,X)]

    This case is for when P -> Y exists and P is treated as a confounder.
    """
    X_vars = observed_covariates if observed_covariates else []

    # Regression: Y ~ D + P + X -> get sd(Y⊥D,P,X)
    y_Y = data[outcome].values
    X_Y = add_constant(data[[treatment, placebo] + X_vars].values)
    var_names_Y = ["const", treatment, placebo] + X_vars
    reg_Y = fit_ols(y_Y, X_Y, var_names_Y)
    residual_sd_Y_given_DPX = np.sqrt(reg_Y.mse_resid)

    # Regression: D ~ P + X -> get sd(D⊥P,X)
    y_D_1 = data[treatment].values
    X_D_1 = add_constant(data[[placebo] + X_vars].values)
    var_names_D_1 = ["const", placebo] + X_vars
    reg_D_1 = fit_ols(y_D_1, X_D_1, var_names_D_1)
    residual_sd_D_given_PX = np.sqrt(reg_D_1.mse_resid)

    # Regression: D ~ X -> get sd(D⊥X)
    if X_vars:
        X_D_2 = add_constant(data[X_vars].values)
        var_names_D_2 = ["const"] + X_vars
    else:
        X_D_2 = add_constant(np.zeros((len(data), 0)))
        var_names_D_2 = ["const"]
    reg_D_2 = fit_ols(y_D_1, X_D_2, var_names_D_2)
    residual_sd_D_given_X = np.sqrt(reg_D_2.mse_resid)

    # Regression: P ~ D + X -> get sd(P⊥D,X)
    y_P = data[placebo].values
    X_P = add_constant(data[[treatment] + X_vars].values)
    var_names_P = ["const", treatment] + X_vars
    reg_P = fit_ols(y_P, X_P, var_names_P)
    residual_sd_P_given_DX = np.sqrt(reg_P.mse_resid)

    # Compute scale factor
    # SF = [sd(Y⊥D,P,X) / sd(D⊥P,X)] x [sd(D⊥X) / sd(P⊥D,X)]
    if residual_sd_D_given_PX == 0 or residual_sd_P_given_DX == 0:
        scale_factor = np.inf
    else:
        scale_factor = (residual_sd_Y_given_DPX / residual_sd_D_given_PX) * \
                       (residual_sd_D_given_X / residual_sd_P_given_DX)

    residuals = {
        "Y_given_DPX": residual_sd_Y_given_DPX,
        "D_given_PX": residual_sd_D_given_PX,
        "D_given_X": residual_sd_D_given_X,
        "P_given_DX": residual_sd_P_given_DX,
    }

    return scale_factor, residuals


def compute_scale_factor_observed_confounder_2(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    placebo: str,
    observed_covariates: List[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute scale factor for Observed Confounder 2 case (Table 2[e/f]).

    SF = [sd(Y⊥D,P,X) / sd(D⊥P,X)] x [sd(P⊥X) / sd(D⊥P,X)]

    This case is for P -> D structure.
    """
    X_vars = observed_covariates if observed_covariates else []

    # Regression: Y ~ D + P + X -> get sd(Y⊥D,P,X)
    y_Y = data[outcome].values
    X_Y = add_constant(data[[treatment, placebo] + X_vars].values)
    var_names_Y = ["const", treatment, placebo] + X_vars
    reg_Y = fit_ols(y_Y, X_Y, var_names_Y)
    residual_sd_Y_given_DPX = np.sqrt(reg_Y.mse_resid)

    # Regression: D ~ P + X -> get sd(D⊥P,X)
    y_D = data[treatment].values
    X_D = add_constant(data[[placebo] + X_vars].values)
    var_names_D = ["const", placebo] + X_vars
    reg_D = fit_ols(y_D, X_D, var_names_D)
    residual_sd_D_given_PX = np.sqrt(reg_D.mse_resid)

    # Regression: P ~ X -> get sd(P⊥X)
    y_P = data[placebo].values
    if X_vars:
        X_P = add_constant(data[X_vars].values)
        var_names_P = ["const"] + X_vars
    else:
        X_P = add_constant(np.zeros((len(data), 0)))
        var_names_P = ["const"]
    reg_P = fit_ols(y_P, X_P, var_names_P)
    residual_sd_P_given_X = np.sqrt(reg_P.mse_resid)

    # Compute scale factor
    # SF = [sd(Y⊥D,P,X) / sd(D⊥P,X)] x [sd(P⊥X) / sd(D⊥P,X)]
    if residual_sd_D_given_PX == 0:
        scale_factor = np.inf
    else:
        scale_factor = (residual_sd_Y_given_DPX / residual_sd_D_given_PX) * \
                       (residual_sd_P_given_X / residual_sd_D_given_PX)

    residuals = {
        "Y_given_DPX": residual_sd_Y_given_DPX,
        "D_given_PX": residual_sd_D_given_PX,
        "P_given_X": residual_sd_P_given_X,
    }

    return scale_factor, residuals


def compute_scale_factor_post_outcome(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    placebo: str,
    observed_covariates: List[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute scale factor for Post-Outcome case (Table 2[g/h]).

    SF = [sd(Y⊥D,X) / sd(D⊥X)] x [sd(Y⊥D,X) / sd(P⊥Y,D,X)]

    This case is for Y -> P structure (placebo is downstream of outcome).
    """
    X_vars = observed_covariates if observed_covariates else []

    # Regression: Y ~ D + X -> get sd(Y⊥D,X)
    y_Y = data[outcome].values
    X_Y = add_constant(data[[treatment] + X_vars].values)
    var_names_Y = ["const", treatment] + X_vars
    reg_Y = fit_ols(y_Y, X_Y, var_names_Y)
    residual_sd_Y_given_DX = np.sqrt(reg_Y.mse_resid)

    # Regression: D ~ X -> get sd(D⊥X)
    y_D = data[treatment].values
    if X_vars:
        X_D = add_constant(data[X_vars].values)
        var_names_D = ["const"] + X_vars
    else:
        X_D = add_constant(np.zeros((len(data), 0)))
        var_names_D = ["const"]
    reg_D = fit_ols(y_D, X_D, var_names_D)
    residual_sd_D_given_X = np.sqrt(reg_D.mse_resid)

    # Regression: P ~ Y + D + X -> get sd(P⊥Y,D,X)
    y_P = data[placebo].values
    X_P = add_constant(data[[outcome, treatment] + X_vars].values)
    var_names_P = ["const", outcome, treatment] + X_vars
    reg_P = fit_ols(y_P, X_P, var_names_P)
    residual_sd_P_given_YDX = np.sqrt(reg_P.mse_resid)

    # Compute scale factor
    # SF = [sd(Y⊥D,X) / sd(D⊥X)] x [sd(Y⊥D,X) / sd(P⊥Y,D,X)]
    if residual_sd_D_given_X == 0 or residual_sd_P_given_YDX == 0:
        scale_factor = np.inf
    else:
        scale_factor = (residual_sd_Y_given_DX / residual_sd_D_given_X) * \
                       (residual_sd_Y_given_DX / residual_sd_P_given_YDX)

    residuals = {
        "Y_given_DX": residual_sd_Y_given_DX,
        "D_given_X": residual_sd_D_given_X,
        "P_given_YDX": residual_sd_P_given_YDX,
    }

    return scale_factor, residuals


def compute_scale_factor_mediator(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    placebo: str,
    observed_covariates: List[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute scale factor for Mediator case (Table 1[d]).

    SF = [sd(P⊥D,X) / sd(D⊥X)] x [sd(Y⊥D,X) / sd(Y⊥D,P,X)]

    Note: The authors do not recommend this approach for mediators.
    """
    X_vars = observed_covariates if observed_covariates else []

    # Regression: P ~ D + X -> get sd(P⊥D,X)
    y_P = data[placebo].values
    X_P = add_constant(data[[treatment] + X_vars].values)
    var_names_P = ["const", treatment] + X_vars
    reg_P = fit_ols(y_P, X_P, var_names_P)
    residual_sd_P_given_DX = np.sqrt(reg_P.mse_resid)

    # Regression: D ~ X -> get sd(D⊥X)
    y_D = data[treatment].values
    if X_vars:
        X_D = add_constant(data[X_vars].values)
        var_names_D = ["const"] + X_vars
    else:
        X_D = add_constant(np.zeros((len(data), 0)))
        var_names_D = ["const"]
    reg_D = fit_ols(y_D, X_D, var_names_D)
    residual_sd_D_given_X = np.sqrt(reg_D.mse_resid)

    # Regression: Y ~ D + X -> get sd(Y⊥D,X)
    y_Y = data[outcome].values
    X_Y_1 = add_constant(data[[treatment] + X_vars].values)
    var_names_Y_1 = ["const", treatment] + X_vars
    reg_Y_1 = fit_ols(y_Y, X_Y_1, var_names_Y_1)
    residual_sd_Y_given_DX = np.sqrt(reg_Y_1.mse_resid)

    # Regression: Y ~ D + P + X -> get sd(Y⊥D,P,X)
    X_Y_2 = add_constant(data[[treatment, placebo] + X_vars].values)
    var_names_Y_2 = ["const", treatment, placebo] + X_vars
    reg_Y_2 = fit_ols(y_Y, X_Y_2, var_names_Y_2)
    residual_sd_Y_given_DPX = np.sqrt(reg_Y_2.mse_resid)

    # Compute scale factor
    # SF = [sd(P⊥D,X) / sd(D⊥X)] x [sd(Y⊥D,X) / sd(Y⊥D,P,X)]
    if residual_sd_D_given_X == 0 or residual_sd_Y_given_DPX == 0:
        scale_factor = np.inf
    else:
        scale_factor = (residual_sd_P_given_DX / residual_sd_D_given_X) * \
                       (residual_sd_Y_given_DX / residual_sd_Y_given_DPX)

    residuals = {
        "P_given_DX": residual_sd_P_given_DX,
        "D_given_X": residual_sd_D_given_X,
        "Y_given_DX": residual_sd_Y_given_DX,
        "Y_given_DPX": residual_sd_Y_given_DPX,
    }

    return scale_factor, residuals


def placebo_lm_extended(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    placebo_outcome: str = "",
    placebo_treatment: str = "",
    observed_covariates: Optional[List[str]] = None,
    placebo_type: str = "auto",
    dp: str = "",  # D -> P direct effect indicator
    py: str = "",  # P -> Y direct effect indicator
    partialIDparam_minmax: Optional[Dict] = None,
) -> PlaceboLMResultExtended:
    """
    Fit PlaceboLM with explicit placebo type selection.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    outcome : str
        Name of outcome variable (Y).
    treatment : str
        Name of treatment variable (D).
    placebo_outcome : str
        Name of placebo outcome variable (N/P depending on context).
    placebo_treatment : str
        Name of placebo treatment variable (P).
    observed_covariates : List[str], optional
        Observed covariates.
    placebo_type : str
        One of:
        - "auto": Automatically detect based on inputs
        - "placebo_outcome": Table 1[a] placebo outcome
        - "placebo_treatment": Table 1[a] placebo treatment
        - "observed_confounder_1": Table 1[c] - P->Y, control for P
        - "mediator": Table 1[d] - D->P->Y (not recommended)
        - "observed_confounder_2": Table 2[e/f] - P->D structure
        - "post_outcome": Table 2[g/h] - Y->P structure
    dp : str
        Indicator that D->P edge exists (imperfect placebo outcome).
    py : str
        Indicator that P->Y edge exists (imperfect placebo treatment).
    partialIDparam_minmax : Dict, optional
        Parameter ranges for partial identification.

    Returns
    -------
    PlaceboLMResultExtended
        Extended result object with placebo-type-specific information.
    """
    if observed_covariates is None:
        observed_covariates = []

    if partialIDparam_minmax is None:
        partialIDparam_minmax = {"k": [-2, 2], "coef_P_D_given_XZ": [-15000, 15000]}

    # Determine the placebo variable
    placebo = placebo_outcome if placebo_outcome else placebo_treatment

    # Drop missing values
    relevant_cols = [outcome, treatment]
    if placebo:
        relevant_cols.append(placebo)
    relevant_cols.extend(observed_covariates)
    analysis_data = data.dropna(subset=relevant_cols).copy()

    X_vars = observed_covariates

    # Auto-detect placebo type
    if placebo_type == "auto":
        if placebo_outcome and not placebo_treatment:
            if dp or py:
                placebo_type = "observed_confounder_1"
            else:
                placebo_type = "placebo_outcome"
        elif placebo_treatment and not placebo_outcome:
            placebo_type = "placebo_treatment"
        elif placebo_outcome and placebo_treatment:
            placebo_type = "double_placebo"
        else:
            raise ValueError("Must specify placebo_outcome or placebo_treatment")

    # Initialize results based on placebo type
    if placebo_type == "placebo_outcome":
        # Table 1[a] - Basic placebo outcome
        y1 = analysis_data[outcome].values
        X1 = add_constant(analysis_data[[treatment] + X_vars].values)
        var_names1 = ["const", treatment] + X_vars
        reg1 = fit_ols(y1, X1, var_names1)

        y2 = analysis_data[placebo_outcome].values
        X2 = add_constant(analysis_data[[treatment] + X_vars].values)
        reg2 = fit_ols(y2, X2, var_names1)

        beta_Y_D_given_X = reg1.params[treatment]
        se_Y_D_given_X = reg1.bse[treatment]
        beta_N_D_given_X = reg2.params[treatment]
        se_N_D_given_X = reg2.bse[treatment]

        residual_sd_Y = np.sqrt(reg1.mse_resid)
        residual_sd_N = np.sqrt(reg2.mse_resid)

        scale_factor = compute_scale_factor_placebo_outcome(residual_sd_Y, residual_sd_N)
        scale_factor_formula = "sd(Y⊥D,X) / sd(N⊥D,X)"

        return PlaceboLMResultExtended(
            data=analysis_data,
            outcome=outcome,
            treatment=treatment,
            placebo_outcome=placebo_outcome,
            placebo_treatment="",
            observed_covariates=observed_covariates,
            placebo_type="placebo_outcome",
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
            scale_factor=scale_factor,
            scale_factor_formula=scale_factor_formula,
        )

    elif placebo_type == "placebo_treatment":
        # Table 1[a] - Basic placebo treatment
        y = analysis_data[outcome].values
        X = add_constant(analysis_data[[treatment, placebo_treatment] + X_vars].values)
        var_names = ["const", treatment, placebo_treatment] + X_vars
        reg1 = fit_ols(y, X, var_names)

        beta_Y_D_given_PX = reg1.params[treatment]
        se_Y_D_given_PX = reg1.bse[treatment]
        beta_Y_P_given_DX = reg1.params[placebo_treatment]
        se_Y_P_given_DX = reg1.bse[placebo_treatment]

        residual_sd_Y = np.sqrt(reg1.mse_resid)

        # Compute scale factor
        sf, res_P, res_D = compute_scale_factor_placebo_treatment(
            analysis_data, treatment, placebo_treatment, X_vars
        )

        return PlaceboLMResultExtended(
            data=analysis_data,
            outcome=outcome,
            treatment=treatment,
            placebo_outcome="",
            placebo_treatment=placebo_treatment,
            observed_covariates=observed_covariates,
            placebo_type="placebo_treatment",
            reg1=reg1,
            reg2=None,
            beta_Y_D_given_X=beta_Y_D_given_PX,  # beta_Y_D|P,X
            beta_N_D_given_X=beta_Y_P_given_DX,  # beta_Y_P|D,X (placebo coef)
            se_Y_D_given_X=se_Y_D_given_PX,
            se_N_D_given_X=se_Y_P_given_DX,
            residual_sd_Y=residual_sd_Y,
            residual_sd_N=res_P,  # sd(P⊥D,X)
            n_obs=len(analysis_data),
            partialIDparam_minmax=partialIDparam_minmax,
            scale_factor=sf,
            scale_factor_formula="sd(P⊥D,X) / sd(D⊥P,X)",
            beta_Y_P_given_DX=beta_Y_P_given_DX,
            se_Y_P_given_DX=se_Y_P_given_DX,
            beta_Y_D_given_PX=beta_Y_D_given_PX,
            se_Y_D_given_PX=se_Y_D_given_PX,
            residual_sd_P_given_DX=res_P,
            residual_sd_D_given_PX=res_D,
        )

    elif placebo_type == "observed_confounder_1":
        # Table 1[c] - P->Y exists, control for P
        placebo = placebo_outcome if placebo_outcome else placebo_treatment

        # Regression 1: Y ~ D + P + X
        y1 = analysis_data[outcome].values
        X1 = add_constant(analysis_data[[treatment, placebo] + X_vars].values)
        var_names1 = ["const", treatment, placebo] + X_vars
        reg1 = fit_ols(y1, X1, var_names1)

        # Regression 2: P ~ D + X
        y2 = analysis_data[placebo].values
        X2 = add_constant(analysis_data[[treatment] + X_vars].values)
        var_names2 = ["const", treatment] + X_vars
        reg2 = fit_ols(y2, X2, var_names2)

        beta_Y_D_given_PX = reg1.params[treatment]
        se_Y_D_given_PX = reg1.bse[treatment]
        beta_P_D_given_X = reg2.params[treatment]
        se_P_D_given_X = reg2.bse[treatment]

        residual_sd_Y = np.sqrt(reg1.mse_resid)
        residual_sd_P = np.sqrt(reg2.mse_resid)

        # Compute scale factor
        sf, residuals = compute_scale_factor_observed_confounder_1(
            analysis_data, outcome, treatment, placebo, X_vars
        )

        return PlaceboLMResultExtended(
            data=analysis_data,
            outcome=outcome,
            treatment=treatment,
            placebo_outcome=placebo_outcome,
            placebo_treatment=placebo_treatment,
            observed_covariates=observed_covariates,
            placebo_type="observed_confounder_1",
            reg1=reg1,
            reg2=reg2,
            beta_Y_D_given_X=beta_Y_D_given_PX,
            beta_N_D_given_X=beta_P_D_given_X,
            se_Y_D_given_X=se_Y_D_given_PX,
            se_N_D_given_X=se_P_D_given_X,
            residual_sd_Y=residual_sd_Y,
            residual_sd_N=residual_sd_P,
            n_obs=len(analysis_data),
            partialIDparam_minmax=partialIDparam_minmax,
            scale_factor=sf,
            scale_factor_formula="[sd(Y⊥D,P,X)/sd(D⊥P,X)] x [sd(D⊥X)/sd(P⊥D,X)]",
            beta_Y_D_given_PX=beta_Y_D_given_PX,
            se_Y_D_given_PX=se_Y_D_given_PX,
            residual_sd_Y_given_DPX=residuals["Y_given_DPX"],
            residual_sd_D_given_PX=residuals["D_given_PX"],
            residual_sd_P_given_DX=residuals["P_given_DX"],
        )

    elif placebo_type == "observed_confounder_2":
        # Table 2[e/f] - P->D structure
        placebo = placebo_outcome if placebo_outcome else placebo_treatment

        # Regression 1: Y ~ D + P + X
        y1 = analysis_data[outcome].values
        X1 = add_constant(analysis_data[[treatment, placebo] + X_vars].values)
        var_names1 = ["const", treatment, placebo] + X_vars
        reg1 = fit_ols(y1, X1, var_names1)

        # Regression 2: D ~ P + X
        y2 = analysis_data[treatment].values
        X2 = add_constant(analysis_data[[placebo] + X_vars].values)
        var_names2 = ["const", placebo] + X_vars
        reg2 = fit_ols(y2, X2, var_names2)

        beta_Y_D_given_PX = reg1.params[treatment]
        se_Y_D_given_PX = reg1.bse[treatment]
        beta_D_P_given_X = reg2.params[placebo]
        se_D_P_given_X = reg2.bse[placebo]

        residual_sd_Y = np.sqrt(reg1.mse_resid)
        residual_sd_D = np.sqrt(reg2.mse_resid)

        # Compute scale factor
        sf, residuals = compute_scale_factor_observed_confounder_2(
            analysis_data, outcome, treatment, placebo, X_vars
        )

        return PlaceboLMResultExtended(
            data=analysis_data,
            outcome=outcome,
            treatment=treatment,
            placebo_outcome=placebo_outcome,
            placebo_treatment=placebo_treatment,
            observed_covariates=observed_covariates,
            placebo_type="observed_confounder_2",
            reg1=reg1,
            reg2=reg2,
            beta_Y_D_given_X=beta_Y_D_given_PX,
            beta_N_D_given_X=beta_D_P_given_X,  # Note: this is D~P, not N~D
            se_Y_D_given_X=se_Y_D_given_PX,
            se_N_D_given_X=se_D_P_given_X,
            residual_sd_Y=residual_sd_Y,
            residual_sd_N=residual_sd_D,
            n_obs=len(analysis_data),
            partialIDparam_minmax=partialIDparam_minmax,
            scale_factor=sf,
            scale_factor_formula="[sd(Y⊥D,P,X)/sd(D⊥P,X)] x [sd(P⊥X)/sd(D⊥P,X)]",
            beta_Y_D_given_PX=beta_Y_D_given_PX,
            se_Y_D_given_PX=se_Y_D_given_PX,
            residual_sd_Y_given_DPX=residuals["Y_given_DPX"],
            residual_sd_D_given_PX=residuals["D_given_PX"],
        )

    elif placebo_type == "post_outcome":
        # Table 2[g/h] - Y->P structure
        placebo = placebo_outcome if placebo_outcome else placebo_treatment

        # Regression 1: Y ~ D + X
        y1 = analysis_data[outcome].values
        X1 = add_constant(analysis_data[[treatment] + X_vars].values)
        var_names1 = ["const", treatment] + X_vars
        reg1 = fit_ols(y1, X1, var_names1)

        # Regression 2: P ~ Y + D + X
        y2 = analysis_data[placebo].values
        X2 = add_constant(analysis_data[[outcome, treatment] + X_vars].values)
        var_names2 = ["const", outcome, treatment] + X_vars
        reg2 = fit_ols(y2, X2, var_names2)

        beta_Y_D_given_X = reg1.params[treatment]
        se_Y_D_given_X = reg1.bse[treatment]
        beta_P_Y_given_DX = reg2.params[outcome]
        se_P_Y_given_DX = reg2.bse[outcome]

        residual_sd_Y = np.sqrt(reg1.mse_resid)
        residual_sd_P = np.sqrt(reg2.mse_resid)

        # Compute scale factor
        sf, residuals = compute_scale_factor_post_outcome(
            analysis_data, outcome, treatment, placebo, X_vars
        )

        return PlaceboLMResultExtended(
            data=analysis_data,
            outcome=outcome,
            treatment=treatment,
            placebo_outcome=placebo_outcome,
            placebo_treatment=placebo_treatment,
            observed_covariates=observed_covariates,
            placebo_type="post_outcome",
            reg1=reg1,
            reg2=reg2,
            beta_Y_D_given_X=beta_Y_D_given_X,
            beta_N_D_given_X=beta_P_Y_given_DX,  # Note: this is P~Y, not N~D
            se_Y_D_given_X=se_Y_D_given_X,
            se_N_D_given_X=se_P_Y_given_DX,
            residual_sd_Y=residual_sd_Y,
            residual_sd_N=residual_sd_P,
            n_obs=len(analysis_data),
            partialIDparam_minmax=partialIDparam_minmax,
            scale_factor=sf,
            scale_factor_formula="[sd(Y⊥D,X)/sd(D⊥X)] x [sd(Y⊥D,X)/sd(P⊥Y,D,X)]",
        )

    elif placebo_type == "mediator":
        # Table 1[d] - D->P->Y (not recommended)
        import warnings
        warnings.warn(
            "Mediator placebo type is not recommended by Rohde & Hazlett (2023). "
            "The interpretation of k and imperfection parameters becomes complicated."
        )

        placebo = placebo_outcome if placebo_outcome else placebo_treatment

        # Regression 1: Y ~ D + X
        y1 = analysis_data[outcome].values
        X1 = add_constant(analysis_data[[treatment] + X_vars].values)
        var_names1 = ["const", treatment] + X_vars
        reg1 = fit_ols(y1, X1, var_names1)

        # Regression 2: Y ~ D + P + X
        X2 = add_constant(analysis_data[[treatment, placebo] + X_vars].values)
        var_names2 = ["const", treatment, placebo] + X_vars
        reg2 = fit_ols(y1, X2, var_names2)

        beta_Y_D_given_X = reg1.params[treatment]
        se_Y_D_given_X = reg1.bse[treatment]
        beta_Y_P_given_DX = reg2.params[placebo]
        se_Y_P_given_DX = reg2.bse[placebo]

        residual_sd_Y_1 = np.sqrt(reg1.mse_resid)
        residual_sd_Y_2 = np.sqrt(reg2.mse_resid)

        # Compute scale factor
        sf, residuals = compute_scale_factor_mediator(
            analysis_data, outcome, treatment, placebo, X_vars
        )

        return PlaceboLMResultExtended(
            data=analysis_data,
            outcome=outcome,
            treatment=treatment,
            placebo_outcome=placebo_outcome,
            placebo_treatment=placebo_treatment,
            observed_covariates=observed_covariates,
            placebo_type="mediator",
            reg1=reg1,
            reg2=reg2,
            beta_Y_D_given_X=beta_Y_D_given_X,
            beta_N_D_given_X=beta_Y_P_given_DX,
            se_Y_D_given_X=se_Y_D_given_X,
            se_N_D_given_X=se_Y_P_given_DX,
            residual_sd_Y=residual_sd_Y_1,
            residual_sd_N=residual_sd_Y_2,
            n_obs=len(analysis_data),
            partialIDparam_minmax=partialIDparam_minmax,
            scale_factor=sf,
            scale_factor_formula="[sd(P⊥D,X)/sd(D⊥X)] x [sd(Y⊥D,X)/sd(Y⊥D,P,X)]",
            beta_Y_P_given_DX=beta_Y_P_given_DX,
            se_Y_P_given_DX=se_Y_P_given_DX,
            residual_sd_Y_given_DPX=residuals["Y_given_DPX"],
            residual_sd_P_given_DX=residuals["P_given_DX"],
        )

    else:
        raise ValueError(f"Unknown placebo_type: {placebo_type}")


def bias_adjustment_extended(
    plm: PlaceboLMResultExtended,
    k: float,
    coef_imperfection: float = 0.0,
) -> float:
    """
    Calculate bias-adjusted estimate using the appropriate formula for the placebo type.

    The general formula is:
        adjusted = beta_target - k x (beta_placebo - coef_imperfection) x SF

    Where:
    - beta_target is the coefficient of interest (treatment effect)
    - beta_placebo is the coefficient in the placebo relationship
    - coef_imperfection is the postulated direct effect (0 for perfect placebo)
    - SF is the scale factor for the specific placebo type

    Parameters
    ----------
    plm : PlaceboLMResultExtended
        Result from placebo_lm_extended().
    k : float
        Ratio of confounding.
    coef_imperfection : float
        The imperfection parameter (direct effect that should be zero for perfect placebo).
        For placebo outcome: coef_P_D_given_XZ (D->N effect)
        For placebo treatment: coef_Y_P_given_DXZ (P->Y effect)

    Returns
    -------
    float
        Bias-adjusted estimate.
    """
    # Bias in placebo relationship
    bias_in_placebo = plm.beta_N_D_given_X - coef_imperfection

    # Apply scale factor
    scaled_bias = k * bias_in_placebo * plm.scale_factor

    # Adjusted estimate
    adjusted = plm.beta_Y_D_given_X - scaled_bias

    return adjusted
