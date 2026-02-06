"""
PlaceboLM: Causal Progress with Imperfect Placebos

A Python implementation of partial identification methods for causal effects
using placebo treatments and outcomes, as described in:

    Rohde, A. and Hazlett, C. (2023). "Causal progress with imperfect
    placebo treatments and outcomes." arXiv:2310.15266

This package provides tools for:
- Partial identification of treatment effects under placebo assumptions
- Sensitivity analysis for departures from perfect placebos and equiconfounding
- Visualization of how estimates vary with different assumptions
- Specialized support for difference-in-differences (DID) as a special case
- Support for ALL placebo types from Tables 1 and 2 in the paper

Supported Placebo Types (Tables 1 & 2)
--------------------------------------
- placebo_outcome: Table 1[a] - Basic placebo outcome (N)
- placebo_treatment: Table 1[a] - Basic placebo treatment (P)
- observed_confounder_1: Table 1[c] - P→Y with P as confounder
- mediator: Table 1[d] - D→P→Y (not recommended)
- observed_confounder_2: Table 2[e/f] - P→D structure
- post_outcome: Table 2[g/h] - Y→P structure
- double_placebo: Appendix B - Both P and N available

Main Functions
--------------
placebo_lm : Fit a PlaceboLM model (auto-detects placebo type)
placebo_lm_extended : Fit with explicit placebo type selection
placebo_lm_table : Generate summary tables
placebo_lm_contour_plot : Create contour plots of sensitivity
placebo_lm_line_plot : Create line plots across parameter values
bias_adjustment : Compute bias-adjusted estimates for given parameters

Classes
-------
PlaceboLMResult : Container for analysis results
PlaceboLMResultExtended : Extended results with scale factor info

Example
-------
>>> import pandas as pd
>>> from placebolm import placebo_lm, placebo_lm_table, placebo_lm_contour_plot
>>>
>>> # Load your data
>>> df = pd.read_csv("data.csv")
>>>
>>> # Fit PlaceboLM model
>>> result = placebo_lm(
...     data=df,
...     outcome="earnings_post",
...     treatment="treatment",
...     placebo_outcome="earnings_pre",
...     observed_covariates=["age", "education", "race"],
...     partialIDparam_minmax={"k": [-2, 2], "coef_P_D_given_XZ": [-5000, 5000]}
... )
>>>
>>> # Generate summary table
>>> table = placebo_lm_table(result, n_boot=500)
>>> print(table)
>>>
>>> # Create contour plot
>>> fig = placebo_lm_contour_plot(result)
>>> fig.savefig("sensitivity.png")
"""

__version__ = "0.1.0"
__author__ = "PlaceboLM Development Team"

# Core functionality
from .core import (
    PlaceboLMResult,
    placebo_lm,
    bias_adjustment,
    bias_adjustment_r2,
    get_estimate_at_params,
    bootstrap_estimates,
    bootstrap_se,
    bootstrap_ci,
)

# Extended placebo types (Tables 1 & 2)
from .placebo_types import (
    PlaceboLMResultExtended,
    placebo_lm_extended,
    bias_adjustment_extended,
    compute_scale_factor_placebo_outcome,
    compute_scale_factor_placebo_treatment,
    compute_scale_factor_observed_confounder_1,
    compute_scale_factor_observed_confounder_2,
    compute_scale_factor_post_outcome,
    compute_scale_factor_mediator,
)

# Table generation
from .tables import (
    placebo_lm_table,
    placebo_lm_bounds,
    sensitivity_table,
)

# Visualization
from .plotting import (
    placebo_lm_contour_plot,
    placebo_lm_line_plot,
    placebo_lm_heatmap,
    placebo_lm_breakdown_plot,
    save_all_plots,
)

# New visualizations and exports
from .exports import (
    # Breakdown frontier
    find_breakdown_k,
    find_breakdown_coef,
    breakdown_frontier,
    placebo_lm_breakdown_frontier_plot,
    # Ribbon plot
    placebo_lm_ribbon_plot,
    # Table exports
    placebo_lm_table_latex,
    placebo_lm_table_markdown,
    export_results,
    # Summary panel
    placebo_lm_summary_panel,
)

# DID-specific functions
from .did import (
    did_placebo_lm,
    did_estimate,
    parallel_trends_sensitivity,
    did_breakdown_k,
    compute_m_parameter,
    did_comparison,
    create_did_panel,
)

# Utilities
from .utils import (
    partial_r2,
    partial_correlation,
    robustness_value,
    sensitivity_stats,
    check_placebo_validity,
    simulate_placebo_data,
    monte_carlo_coverage,
    format_table,
)

__all__ = [
    # Core
    "PlaceboLMResult",
    "placebo_lm",
    "bias_adjustment",
    "bias_adjustment_r2",
    "get_estimate_at_params",
    "bootstrap_estimates",
    "bootstrap_se",
    "bootstrap_ci",
    # Extended placebo types
    "PlaceboLMResultExtended",
    "placebo_lm_extended",
    "bias_adjustment_extended",
    "compute_scale_factor_placebo_outcome",
    "compute_scale_factor_placebo_treatment",
    "compute_scale_factor_observed_confounder_1",
    "compute_scale_factor_observed_confounder_2",
    "compute_scale_factor_post_outcome",
    "compute_scale_factor_mediator",
    # Tables
    "placebo_lm_table",
    "placebo_lm_bounds",
    "sensitivity_table",
    # Plotting
    "placebo_lm_contour_plot",
    "placebo_lm_line_plot",
    "placebo_lm_heatmap",
    "placebo_lm_breakdown_plot",
    "save_all_plots",
    # New visualizations and exports
    "find_breakdown_k",
    "find_breakdown_coef",
    "breakdown_frontier",
    "placebo_lm_breakdown_frontier_plot",
    "placebo_lm_ribbon_plot",
    "placebo_lm_table_latex",
    "placebo_lm_table_markdown",
    "export_results",
    "placebo_lm_summary_panel",
    # DID
    "did_placebo_lm",
    "did_estimate",
    "parallel_trends_sensitivity",
    "did_breakdown_k",
    "compute_m_parameter",
    "did_comparison",
    "create_did_panel",
    # Utils
    "partial_r2",
    "partial_correlation",
    "robustness_value",
    "sensitivity_stats",
    "check_placebo_validity",
    "simulate_placebo_data",
    "monte_carlo_coverage",
    "format_table",
]
