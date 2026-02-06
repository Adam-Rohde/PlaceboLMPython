# PlaceboLM: Causal Progress with Imperfect Placebos

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of partial identification methods for causal effects using placebo treatments and outcomes.

Based on the methodology described in:

> Rohde, A. and Hazlett, C. (2023). "Causal progress with imperfect placebo treatments and outcomes." [arXiv:2310.15266](https://arxiv.org/abs/2310.15266)

## Overview

When researchers seek to estimate causal effects from observational data, unobserved confounding is often a concern. **PlaceboLM** provides tools for leveraging "placebo" variables—outcomes that should not be affected by treatment, or treatments that should not affect the outcome—to make progress on causal identification.

### Key Features

- **Partial identification** of treatment effects under assumptions about confounding and placebo validity
- **Sensitivity analysis** for departures from:
  - **Perfect placebos** (allowing direct D→N effects)
  - **Equiconfounding** (allowing unequal confounding via the `k` parameter)
- **Visualization tools** including contour plots, line plots, and breakdown analysis
- **DID support** treating pre-treatment outcomes as placebo outcomes, with parallel trends as a special case (k=1)
- **Bootstrap inference** for uncertainty quantification

## Installation

```bash
# From source
git clone https://github.com/Adam-Rohde/PlaceboLM-Python.git
cd PlaceboLM-Python
pip install -e .

# Or install dependencies directly
pip install numpy pandas statsmodels scipy matplotlib
```

## Quick Start

```python
import pandas as pd
from placebolm import placebo_lm, placebo_lm_table, placebo_lm_contour_plot, bias_adjustment

# Load data (using LaLonde job training data as example)
df = pd.read_csv("lalonde.csv")

# Fit PlaceboLM model
# - outcome: post-treatment earnings (re78)
# - treatment: job training program indicator
# - placebo_outcome: pre-treatment earnings (re74)
result = placebo_lm(
    data=df,
    outcome="re78",
    treatment="treat",
    placebo_outcome="re74",
    observed_covariates=["age", "education", "black", "hispanic", "married", "nodegree"],
    partialIDparam_minmax={
        "k": [-2, 2],                    # Range for confounding ratio
        "coef_P_D_given_XZ": [-15000, 15000]  # Range for direct D→N effect
    }
)

# Get specific estimates
print(f"No confounding (k=0): {bias_adjustment(result, k=0):.2f}")
print(f"Perfect placebo (k=1): {bias_adjustment(result, k=1):.2f}")

# Generate summary table with bootstrap standard errors
table = placebo_lm_table(result, n_boot=500, alpha=0.05)
print(table)

# Create contour plot
fig = placebo_lm_contour_plot(result, gran=50)
fig.savefig("sensitivity_contour.png")
```

## Core Concepts

### The `k` Parameter (Confounding Ratio)

The parameter `k` represents the ratio of confounding in the placebo relationship relative to the treatment-outcome relationship:

- **k = 0**: No unobserved confounding (OLS is unbiased)
- **k = 1**: Equal confounding ("equiconfounding" or "parallel trends" in DID)
- **k > 1**: Placebo relationship has *more* confounding
- **k < 1**: Placebo relationship has *less* confounding

### The `coef_P_D_given_XZ` Parameter (Placebo Imperfection)

This parameter allows for "imperfect" placebos where treatment has a direct effect on the placebo outcome:

- **coef_P_D_given_XZ = 0**: Perfect placebo (no D→N effect)
- **coef_P_D_given_XZ ≠ 0**: Imperfect placebo

### Bias Adjustment Formula

The bias-adjusted estimate is:

```
adjusted = β_Y_D|X - k × (β_N_D|X - coef_P_D_given_XZ)
```

where:
- β_Y_D|X is the observed treatment-outcome coefficient
- β_N_D|X is the observed treatment-placebo coefficient

## API Reference

### Core Functions

#### `placebo_lm(data, outcome, treatment, placebo_outcome, ...)`

Fit a PlaceboLM model.

```python
result = placebo_lm(
    data=df,                      # DataFrame
    outcome="Y",                  # Outcome variable name
    treatment="D",                # Treatment variable name
    placebo_outcome="N",          # Placebo outcome variable name
    observed_covariates=["X1"],   # List of covariate names
    partialIDparam_minmax={       # Parameter ranges
        "k": [-2, 2],
        "coef_P_D_given_XZ": [-1000, 1000]
    }
)
```

#### `bias_adjustment(plm, k, coef_P_D_given_XZ)`

Compute bias-adjusted estimate for specific parameters.

```python
# DID-style estimate (k=1, perfect placebo)
estimate = bias_adjustment(result, k=1, coef_P_D_given_XZ=0)
```

### Table Functions

#### `placebo_lm_table(plm, n_boot, ptiles, alpha)`

Generate summary table with bootstrap standard errors and confidence intervals.

```python
table = placebo_lm_table(result, n_boot=1000, alpha=0.05)
```

#### `sensitivity_table(plm, k_values, coef_values, n_boot)`

Generate table for specific parameter values.

```python
table = sensitivity_table(result, k_values=[0, 0.5, 1, 1.5, 2])
```

### Visualization Functions

#### `placebo_lm_contour_plot(plm, gran, ...)`

Create contour plot of estimates over parameter space.

```python
fig = placebo_lm_contour_plot(result, gran=100, cmap="RdBu_r")
```

#### `placebo_lm_line_plot(plm, focus_param, ptiles, ...)`

Create line plots showing how estimates vary with one parameter.

```python
figs = placebo_lm_line_plot(result, focus_param="k", n_boot=500)
```

#### `placebo_lm_breakdown_plot(plm, ...)`

Show where estimates cross zero (breakdown analysis).

```python
fig = placebo_lm_breakdown_plot(result)
```

### DID-Specific Functions

```python
from placebolm import did_placebo_lm, did_estimate, parallel_trends_sensitivity

# Fit DID model
result = did_placebo_lm(
    data=df,
    outcome_post="Y_post",
    outcome_pre="Y_pre",
    treatment="D"
)

# Get DID estimate (equivalent to k=1)
did_est = did_estimate(result)

# Sensitivity to parallel trends violations
table = parallel_trends_sensitivity(result, k_values=[0.5, 0.75, 1, 1.25, 1.5])
```

### Utilities

```python
from placebolm import (
    simulate_placebo_data,    # Generate test data
    check_placebo_validity,   # Diagnostic checks
    robustness_value,         # Cinelli-Hazlett robustness values
    partial_r2,               # Partial R-squared
)

# Generate simulated data
df = simulate_placebo_data(n=1000, true_effect=2.0, confounding_strength=0.5)

# Check placebo assumptions
diagnostics = check_placebo_validity(df, "Y", "D", "N", ["X1", "X2"])
```

## Examples

### Example 1: LaLonde Job Training Data

```python
import pandas as pd
from placebolm import placebo_lm, placebo_lm_table, placebo_lm_contour_plot

# Load LaLonde data (available from various sources)
df = pd.read_csv("lalonde_psid.csv")

# Use pre-treatment earnings as placebo outcome
result = placebo_lm(
    data=df,
    outcome="re78",            # 1978 earnings
    treatment="treat",          # Job training indicator
    placebo_outcome="re74",     # 1974 earnings (pre-treatment)
    observed_covariates=["age", "education", "black", "hispanic", "married", "nodegree"]
)

# The observed treatment effect on 1978 earnings
print(f"Naive OLS: ${result.beta_Y_D_given_X:,.0f}")

# Effect on pre-treatment earnings (should be ~0 if no confounding)
print(f"Effect on placebo: ${result.beta_N_D_given_X:,.0f}")

# Bias-adjusted estimates
from placebolm import bias_adjustment
print(f"k=0 (no confounding): ${bias_adjustment(result, k=0):,.0f}")
print(f"k=1 (perfect placebo/DID): ${bias_adjustment(result, k=1):,.0f}")
```

### Example 2: Monte Carlo Simulation

```python
from placebolm import simulate_placebo_data, monte_carlo_coverage

# Test coverage of confidence intervals
results = monte_carlo_coverage(
    n_sim=500,
    n=500,
    true_effect=2.0,
    confounding_strength=0.5,
    k_assumed=1.0,  # Assume k=1 (correct)
    alpha=0.05
)

print(f"Coverage: {results['coverage']:.1%}")
print(f"Bias: {results['bias']:.3f}")
print(f"RMSE: {results['rmse']:.3f}")
```

## Comparison with R Package

This Python package mirrors the functionality of the [R PlaceboLM package](https://github.com/Adam-Rohde/PlaceboLM):

| R Function | Python Function |
|------------|-----------------|
| `placeboLM()` | `placebo_lm()` |
| `placeboLM_table()` | `placebo_lm_table()` |
| `placeboLM_contour_plot()` | `placebo_lm_contour_plot()` |
| `placeboLM_line_plot()` | `placebo_lm_line_plot()` |

## References

- Rohde, A. and Hazlett, C. (2023). "Causal progress with imperfect placebo treatments and outcomes." [arXiv:2310.15266](https://arxiv.org/abs/2310.15266)
- Cinelli, C. and Hazlett, C. (2020). "Making Sense of Sensitivity: Extending Omitted Variable Bias." *Journal of the Royal Statistical Society: Series B*, 82(1), 39-67.
- LaLonde, R.J. (1986). "Evaluating the Econometric Evaluations of Training Programs with Experimental Data." *American Economic Review*, 76(4), 604-620.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
