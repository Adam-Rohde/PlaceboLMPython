"""
Example: LaLonde Job Training Data Analysis

This example demonstrates using PlaceboLM to analyze the LaLonde (1986)
job training data, using pre-treatment earnings as a placebo outcome.

The LaLonde data comes from the National Supported Work Demonstration,
a randomized job training program. Here we combine the experimental
treatment group with a non-experimental comparison group (PSID controls)
to create an observational study setting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for development
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from placebolm import (
    placebo_lm,
    bias_adjustment,
    placebo_lm_table,
    placebo_lm_contour_plot,
    placebo_lm_line_plot,
    placebo_lm_breakdown_plot,
    did_estimate,
    did_breakdown_k,
    sensitivity_table,
)


def create_lalonde_data():
    """
    Create a synthetic version of the LaLonde data for demonstration.

    In practice, you would load the real LaLonde data:
    - From R's 'qte' package: lalonde.psid
    - From various online sources

    The key variables are:
    - treat: Job training indicator (1 = trained)
    - re78: Real earnings in 1978 (post-treatment outcome)
    - re74: Real earnings in 1974 (placebo outcome)
    - re75: Real earnings in 1975 (additional placebo)
    - age, education, black, hispanic, married, nodegree: Covariates
    """
    np.random.seed(42)
    n = 2675  # Approximate size of LaLonde PSID data

    # Generate covariates similar to LaLonde data
    age = np.random.normal(25, 7, n).clip(17, 55).astype(int)
    education = np.random.normal(10, 2, n).clip(3, 16).astype(int)
    black = np.random.binomial(1, 0.4, n)
    hispanic = np.random.binomial(1, 0.1, n)
    married = np.random.binomial(1, 0.2, n)
    nodegree = (education < 12).astype(int)

    # Unobserved confounder (e.g., motivation, ability)
    U = np.random.randn(n)

    # Treatment selection (non-random in observational setting)
    # Those with lower earnings/education more likely to seek training
    treat_prob = 1 / (1 + np.exp(0.5 + 0.05 * education + 0.3 * U + 0.1 * age/10))
    treat = np.random.binomial(1, treat_prob)

    # Pre-treatment earnings (1974, 1975)
    # Affected by covariates and unobserved factors, NOT by treatment
    base_earnings = (
        2000 +
        500 * education +
        100 * age +
        -2000 * black +
        -1500 * hispanic +
        3000 * married +
        -1500 * nodegree +
        2000 * U +
        np.random.randn(n) * 5000
    ).clip(0, None)

    re74 = base_earnings + np.random.randn(n) * 3000
    re74 = re74.clip(0, None)

    re75 = base_earnings + np.random.randn(n) * 3000 + np.random.randn(n) * 1000
    re75 = re75.clip(0, None)

    # Post-treatment earnings (1978)
    # Affected by treatment (true effect ~$1800) + confounding
    true_effect = 1800
    re78 = (
        base_earnings +
        true_effect * treat +
        np.random.randn(n) * 4000 +
        500 * (age - 25)  # Some drift
    ).clip(0, None)

    return pd.DataFrame({
        "treat": treat,
        "re78": re78,
        "re74": re74,
        "re75": re75,
        "age": age,
        "education": education,
        "black": black,
        "hispanic": hispanic,
        "married": married,
        "nodegree": nodegree,
    })


def main():
    """Run the LaLonde analysis example."""

    print("=" * 70)
    print("PlaceboLM Example: LaLonde Job Training Data")
    print("=" * 70)

    # Create/load data
    print("\n1. Loading data...")
    df = create_lalonde_data()
    print(f"   Sample size: {len(df)}")
    print(f"   Treated: {df['treat'].sum()}")
    print(f"   Control: {(1 - df['treat']).sum()}")

    # Basic statistics
    print("\n2. Descriptive statistics:")
    print(f"   Mean re78 (treated): ${df[df['treat']==1]['re78'].mean():,.0f}")
    print(f"   Mean re78 (control): ${df[df['treat']==0]['re78'].mean():,.0f}")
    print(f"   Mean re74 (treated): ${df[df['treat']==1]['re74'].mean():,.0f}")
    print(f"   Mean re74 (control): ${df[df['treat']==0]['re74'].mean():,.0f}")

    # Fit PlaceboLM
    print("\n3. Fitting PlaceboLM model...")
    covariates = ["age", "education", "black", "hispanic", "married", "nodegree"]

    plm = placebo_lm(
        data=df,
        outcome="re78",
        treatment="treat",
        placebo_outcome="re74",
        observed_covariates=covariates,
        partialIDparam_minmax={
            "k": [-2, 2],
            "coef_P_D_given_XZ": [-15000, 15000]
        }
    )

    # Key results
    print("\n4. Key regression results:")
    print(f"   beta(re78 ~ treat | X) = ${plm.beta_Y_D_given_X:,.0f} (SE: ${plm.se_Y_D_given_X:,.0f})")
    print(f"   beta(re74 ~ treat | X) = ${plm.beta_N_D_given_X:,.0f} (SE: ${plm.se_N_D_given_X:,.0f})")

    # Bias-adjusted estimates under different assumptions
    print("\n5. Bias-adjusted estimates:")

    scenarios = [
        ("No confounding (k=0)", 0, 0),
        ("Perfect placebo (k=1, DID)", 1, 0),
        ("50% confounding (k=0.5)", 0.5, 0),
        ("Double confounding (k=2)", 2, 0),
    ]

    for name, k, coef in scenarios:
        est = bias_adjustment(plm, k=k, coef_P_D_given_XZ=coef)
        print(f"   {name}: ${est:,.0f}")

    # Breakdown point
    breakdown_k = did_breakdown_k(plm)
    print(f"\n   Breakdown k (estimate = 0): {breakdown_k:.2f}")

    # Sensitivity table
    print("\n6. Sensitivity table (small bootstrap for speed)...")
    table = sensitivity_table(
        plm,
        k_values=[0, 0.5, 1, 1.5, 2],
        n_boot=100,
        seed=42,
    )
    print(table.to_string(index=False))

    # Create plots
    print("\n7. Creating visualizations...")

    # Contour plot
    fig1 = placebo_lm_contour_plot(plm, gran=50)
    fig1.savefig("lalonde_contour.png", dpi=150, bbox_inches='tight')
    print("   Saved: lalonde_contour.png")

    # Breakdown plot
    fig2 = placebo_lm_breakdown_plot(plm, n_boot=100, seed=42)
    fig2.savefig("lalonde_breakdown.png", dpi=150, bbox_inches='tight')
    print("   Saved: lalonde_breakdown.png")

    # Line plots
    figs = placebo_lm_line_plot(plm, ptiles=[0, 0.5, 1], n_boot=50, gran=30, seed=42)
    for i, fig in enumerate(figs):
        fig.savefig(f"lalonde_line_{i}.png", dpi=150, bbox_inches='tight')
        print(f"   Saved: lalonde_line_{i}.png")

    plt.close('all')

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    # Interpretation
    print("\nInterpretation:")
    print("-" * 70)
    print("""
The naive OLS estimate (k=0 assumption) suggests a large positive effect.
However, we see that the treatment also 'predicts' pre-treatment earnings
(beta_N_D|X), which it cannot have caused. This indicates confounding.

The DID/perfect placebo estimate (k=1) subtracts this pre-treatment
difference, providing a bias-corrected estimate.

The contour plot shows how the estimate varies across different
assumptions about:
- k: the ratio of confounding (k=1 is parallel trends)
- coef_P_D_given_XZ: direct effect of treatment on placebo (0 = perfect)

The breakdown analysis shows at what value of k the treatment effect
becomes zero, helping assess how robust conclusions are to assumption
violations.
""")


if __name__ == "__main__":
    main()
