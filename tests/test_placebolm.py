"""
Tests for PlaceboLM package.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from placebolm import (
    placebo_lm,
    bias_adjustment,
    placebo_lm_table,
    placebo_lm_contour_plot,
    simulate_placebo_data,
    did_placebo_lm,
    did_estimate,
    did_breakdown_k,
    partial_r2,
    check_placebo_validity,
    bootstrap_se,
    sensitivity_table,
)


@pytest.fixture
def simulated_data():
    """Generate simulated data for testing."""
    return simulate_placebo_data(
        n=500,
        true_effect=2.0,
        confounding_strength=0.5,
        placebo_imperfection=0.0,
        seed=42,
    )


@pytest.fixture
def simple_data():
    """Generate simple deterministic data for testing."""
    np.random.seed(123)
    n = 200

    X1 = np.random.randn(n)
    U = np.random.randn(n)
    D = (0.5 * U + 0.3 * X1 + np.random.randn(n) > 0).astype(int)
    N = 0.5 * U + np.random.randn(n) * 0.5  # Placebo outcome
    Y = 3.0 * D + 0.5 * U + 0.3 * X1 + np.random.randn(n) * 0.5  # True effect = 3

    return pd.DataFrame({"Y": Y, "D": D, "N": N, "X1": X1, "U": U})


class TestPlaceboLM:
    """Tests for core placebo_lm functionality."""

    def test_basic_fit(self, simple_data):
        """Test that placebo_lm runs without errors."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
            observed_covariates=["X1"],
        )

        assert result is not None
        assert result.n_obs == len(simple_data)
        assert result.outcome == "Y"
        assert result.treatment == "D"
        assert result.placebo_outcome == "N"

    def test_coefficients_computed(self, simple_data):
        """Test that regression coefficients are computed."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
            observed_covariates=["X1"],
        )

        assert np.isfinite(result.beta_Y_D_given_X)
        assert np.isfinite(result.beta_N_D_given_X)
        assert np.isfinite(result.se_Y_D_given_X)
        assert np.isfinite(result.se_N_D_given_X)
        assert result.se_Y_D_given_X > 0

    def test_no_covariates(self, simple_data):
        """Test that placebo_lm works without covariates."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
        )

        assert result is not None
        assert result.observed_covariates == []


class TestBiasAdjustment:
    """Tests for bias adjustment calculations."""

    def test_k_zero_equals_naive(self, simple_data):
        """Test that k=0 returns the naive OLS estimate."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
            observed_covariates=["X1"],
        )

        adjusted = bias_adjustment(result, k=0, coef_P_D_given_XZ=0)

        # k=0 means no adjustment
        assert np.isclose(adjusted, result.beta_Y_D_given_X, atol=1e-10)

    def test_k_one_subtracts_placebo(self, simple_data):
        """Test that k=1 subtracts the placebo coefficient."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
            observed_covariates=["X1"],
        )

        adjusted = bias_adjustment(result, k=1, coef_P_D_given_XZ=0)
        expected = result.beta_Y_D_given_X - result.beta_N_D_given_X

        assert np.isclose(adjusted, expected, atol=1e-10)

    def test_coef_P_adjustment(self, simple_data):
        """Test that coef_P_D_given_XZ affects the adjustment."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
            observed_covariates=["X1"],
        )

        adj1 = bias_adjustment(result, k=1, coef_P_D_given_XZ=0)
        adj2 = bias_adjustment(result, k=1, coef_P_D_given_XZ=1.0)

        # With positive coef_P, adjustment should be larger (less negative bias removed)
        assert adj2 > adj1

    def test_adjustment_formula(self, simple_data):
        """Test the exact adjustment formula."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
        )

        k = 0.75
        coef = 0.5
        adjusted = bias_adjustment(result, k=k, coef_P_D_given_XZ=coef)
        expected = result.beta_Y_D_given_X - k * (result.beta_N_D_given_X - coef)

        assert np.isclose(adjusted, expected, atol=1e-10)


class TestDID:
    """Tests for difference-in-differences functionality."""

    def test_did_estimate_equals_k1(self, simple_data):
        """Test that DID estimate equals bias_adjustment with k=1."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
        )

        did_est = did_estimate(result)
        k1_est = bias_adjustment(result, k=1, coef_P_D_given_XZ=0)

        assert np.isclose(did_est, k1_est, atol=1e-10)

    def test_did_breakdown(self, simple_data):
        """Test DID breakdown calculation."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
        )

        breakdown = did_breakdown_k(result)

        # At breakdown k, estimate should be zero
        if np.isfinite(breakdown):
            estimate_at_breakdown = bias_adjustment(result, k=breakdown, coef_P_D_given_XZ=0)
            assert np.isclose(estimate_at_breakdown, 0, atol=1e-10)

    def test_did_placebo_lm(self, simple_data):
        """Test DID-specific model fitting."""
        # Rename columns to match DID naming
        df = simple_data.rename(columns={"Y": "Y_post", "N": "Y_pre"})

        result = did_placebo_lm(
            data=df,
            outcome_post="Y_post",
            outcome_pre="Y_pre",
            treatment="D",
            observed_covariates=["X1"],
        )

        assert result.placebo_outcome == "Y_pre"
        assert result.outcome == "Y_post"


class TestSimulation:
    """Tests for data simulation functions."""

    def test_simulate_basic(self):
        """Test basic simulation functionality."""
        df = simulate_placebo_data(n=100, seed=42)

        assert len(df) == 100
        assert set(df.columns) == {"Y", "D", "N", "U", "X1", "X2"}
        assert df["D"].isin([0, 1]).all()

    def test_simulate_reproducible(self):
        """Test that seed makes simulation reproducible."""
        df1 = simulate_placebo_data(n=100, seed=42)
        df2 = simulate_placebo_data(n=100, seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_simulate_effect_direction(self):
        """Test that positive true effect leads to positive coefficient."""
        df = simulate_placebo_data(
            n=1000,
            true_effect=5.0,
            confounding_strength=0.0,  # No confounding
            seed=42,
        )

        result = placebo_lm(
            data=df,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
            observed_covariates=["X1", "X2"],
        )

        # With no confounding, OLS should be close to true effect
        assert result.beta_Y_D_given_X > 3.0  # Allow some noise


class TestUtilities:
    """Tests for utility functions."""

    def test_partial_r2(self, simple_data):
        """Test partial R-squared calculation."""
        pr2 = partial_r2(simple_data, "Y", "D", ["X1"])

        assert 0 <= pr2 <= 1

    def test_partial_r2_no_controls(self, simple_data):
        """Test partial R-squared without controls."""
        pr2 = partial_r2(simple_data, "Y", "D")

        assert 0 <= pr2 <= 1

    def test_check_placebo_validity(self, simple_data):
        """Test placebo validity diagnostics."""
        diagnostics = check_placebo_validity(
            simple_data, "Y", "D", "N", ["X1"]
        )

        assert "placebo_coef" in diagnostics
        assert "placebo_pvalue" in diagnostics
        assert "messages" in diagnostics


class TestBootstrap:
    """Tests for bootstrap inference."""

    def test_bootstrap_se_positive(self, simple_data):
        """Test that bootstrap SE is positive."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
        )

        se = bootstrap_se(result, k=1, coef_P_D_given_XZ=0, n_boot=50, seed=42)

        assert se > 0

    def test_bootstrap_se_reasonable(self, simple_data):
        """Test that bootstrap SE is in reasonable range."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
        )

        se = bootstrap_se(result, k=1, coef_P_D_given_XZ=0, n_boot=100, seed=42)

        # SE should be smaller than the estimate's magnitude (usually)
        estimate = bias_adjustment(result, k=1, coef_P_D_given_XZ=0)
        assert se < abs(estimate) * 2


class TestTables:
    """Tests for table generation."""

    def test_sensitivity_table(self, simple_data):
        """Test sensitivity table generation."""
        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
        )

        table = sensitivity_table(
            result,
            k_values=[0, 0.5, 1],
            n_boot=20,
            seed=42,
        )

        assert len(table) == 3
        assert "k" in table.columns
        assert "Estimate" in table.columns


class TestPlotting:
    """Tests for plotting functions."""

    def test_contour_plot_runs(self, simple_data):
        """Test that contour plot runs without error."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        result = placebo_lm(
            data=simple_data,
            outcome="Y",
            treatment="D",
            placebo_outcome="N",
        )

        fig = placebo_lm_contour_plot(result, gran=10)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
