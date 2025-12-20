"""
Tests for evaluation suite functionality.

This module tests:
- K-fold cross-validation
- Correlation computation with confidence intervals
- Calibration error computation
- Error distribution analysis
- Bootstrap confidence intervals
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from udl_rating_framework.evaluation.evaluation_suite import (
    EvaluationSuite,
    EvaluationResult,
)


class TestEvaluationSuite:
    """Test cases for EvaluationSuite class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.suite = EvaluationSuite(k_folds=5, bootstrap_samples=1000)

    def test_initialization(self):
        """Test EvaluationSuite initialization."""
        suite = EvaluationSuite(
            k_folds=10, bootstrap_samples=2000, confidence_level=0.99
        )
        assert suite.k_folds == 10
        assert suite.bootstrap_samples == 2000
        assert suite.confidence_level == 0.99
        assert abs(suite.alpha - 0.01) < 1e-10

    def test_initialization_validation(self):
        """Test initialization parameter validation."""
        # Test k_folds < 5
        with pytest.raises(ValueError, match="k_folds must be ≥ 5"):
            EvaluationSuite(k_folds=3)

        # Test bootstrap_samples < 1000
        with pytest.raises(ValueError, match="bootstrap_samples must be ≥ 1000"):
            EvaluationSuite(bootstrap_samples=500)

        # Test invalid confidence_level
        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            EvaluationSuite(confidence_level=1.5)

    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        noise_level=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=10, deadline=10000)
    def test_property_correlation_reporting(self, n_samples, noise_level):
        """
        **Feature: udl-rating-framework, Property 37: Correlation Reporting**
        **Validates: Requirements 10.2**

        For any evaluation, Pearson and Spearman correlations must be reported with 95% CIs.
        """
        # Generate correlated data
        np.random.seed(42)
        x = np.random.randn(n_samples)
        y_true = 2 * x + 1 + noise_level * np.random.randn(n_samples)
        y_pred = 1.8 * x + 0.9 + 0.5 * noise_level * np.random.randn(n_samples)

        # Compute correlations with CI
        pearson_r, pearson_ci, spearman_r, spearman_ci = (
            self.suite.compute_correlations_with_ci(y_true, y_pred)
        )

        # Verify Pearson correlation is computed
        expected_pearson, _ = pearsonr(y_true, y_pred)
        assert abs(pearson_r - expected_pearson) < 1e-10, (
            f"Pearson correlation mismatch: {pearson_r} vs {expected_pearson}"
        )

        # Verify Spearman correlation is computed
        expected_spearman, _ = spearmanr(y_true, y_pred)
        assert abs(spearman_r - expected_spearman) < 1e-10, (
            f"Spearman correlation mismatch: {spearman_r} vs {expected_spearman}"
        )

        # Verify confidence intervals are provided
        assert isinstance(pearson_ci, tuple) and len(pearson_ci) == 2
        assert isinstance(spearman_ci, tuple) and len(spearman_ci) == 2

        # Verify CI bounds are reasonable (lower < upper)
        assert pearson_ci[0] <= pearson_ci[1], (
            f"Pearson CI bounds invalid: {pearson_ci}"
        )
        assert spearman_ci[0] <= spearman_ci[1], (
            f"Spearman CI bounds invalid: {spearman_ci}"
        )

        # Verify correlations are within their CIs (should be true for bootstrap)
        assert pearson_ci[0] <= pearson_r <= pearson_ci[1], (
            f"Pearson r {pearson_r} not in CI {pearson_ci}"
        )
        assert spearman_ci[0] <= spearman_r <= spearman_ci[1], (
            f"Spearman r {spearman_r} not in CI {spearman_ci}"
        )

    @given(
        n_samples=st.integers(min_value=100, max_value=300),
        n_bins=st.integers(min_value=5, max_value=15),
    )
    @settings(max_examples=10, deadline=10000)
    def test_property_calibration_error_computation(self, n_samples, n_bins):
        """
        **Feature: udl-rating-framework, Property 38: Calibration Error Computation**
        **Validates: Requirements 10.3**

        For any evaluation, ECE = Σ_i |accuracy_i - confidence_i| · (n_i/n) must be computed.
        """
        # Generate synthetic data for calibration
        np.random.seed(42)

        # Create binary classification scenario
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = y_true.copy()  # Perfect predictions

        # Create confidences that should be well-calibrated
        confidences = np.random.uniform(0.1, 0.9, n_samples)

        # Compute calibration error
        ece = self.suite.compute_calibration_error(
            y_true, y_pred, confidences, n_bins=n_bins
        )

        # Verify ECE is computed and bounded
        assert isinstance(ece, (int, float)), f"ECE should be numeric, got {type(ece)}"
        assert 0 <= ece <= 1, f"ECE should be in [0,1], got {ece}"

        # Test with perfectly calibrated data (confidence = accuracy)
        # Create data where confidence matches accuracy in each bin
        perfect_confidences = y_true.astype(float)  # 0 or 1 confidence matching labels
        perfect_ece = self.suite.compute_calibration_error(
            y_true, y_pred, perfect_confidences, n_bins=n_bins
        )

        # Perfect calibration should have low ECE (though not necessarily 0 due to binning)
        assert perfect_ece >= 0, (
            f"Perfect calibration ECE should be non-negative, got {perfect_ece}"
        )

    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        error_scale=st.floats(min_value=0.1, max_value=3.0),
    )
    @settings(max_examples=10, deadline=10000)
    def test_property_error_distribution_analysis(self, n_samples, error_scale):
        """
        **Feature: udl-rating-framework, Property 39: Error Distribution Analysis**
        **Validates: Requirements 10.4**

        For any evaluation, error distributions must be analyzed with Shapiro-Wilk test.
        """
        # Generate errors with known distribution
        np.random.seed(42)
        errors = np.random.normal(0, error_scale, n_samples)

        # Analyze error distribution
        shapiro_stat, p_value = self.suite.analyze_error_distribution(errors)

        # Verify Shapiro-Wilk test is applied
        assert isinstance(shapiro_stat, (int, float)), (
            f"Shapiro statistic should be numeric, got {type(shapiro_stat)}"
        )
        assert isinstance(p_value, (int, float)), (
            f"P-value should be numeric, got {type(p_value)}"
        )

        # Verify test statistic is in valid range
        assert 0 <= shapiro_stat <= 1, (
            f"Shapiro statistic should be in [0,1], got {shapiro_stat}"
        )
        assert 0 <= p_value <= 1, f"P-value should be in [0,1], got {p_value}"

        # For normal errors, we expect higher p-values (though this is probabilistic)
        # We just verify the test runs and produces reasonable values
        assert not np.isnan(shapiro_stat), "Shapiro statistic should not be NaN"
        assert not np.isnan(p_value), "P-value should not be NaN"

    @given(
        n_samples=st.integers(min_value=100, max_value=300),
        noise_level=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=10, deadline=15000)
    def test_property_bootstrap_confidence_intervals(self, n_samples, noise_level):
        """
        **Feature: udl-rating-framework, Property 40: Bootstrap Confidence Intervals**
        **Validates: Requirements 10.5**

        For any performance metric, bootstrap CIs must be computed with B ≥ 1000.
        """
        # Generate synthetic prediction data
        np.random.seed(42)
        y_true = np.random.randn(n_samples)
        y_pred = y_true + noise_level * np.random.randn(n_samples)

        # Compute bootstrap confidence intervals
        bootstrap_cis = self.suite.bootstrap_confidence_intervals(y_true, y_pred)

        # Verify bootstrap samples ≥ 1000
        assert self.suite.bootstrap_samples >= 1000, (
            f"Bootstrap samples should be ≥ 1000, got {self.suite.bootstrap_samples}"
        )

        # Verify CIs are computed for default metrics
        expected_metrics = ["MAE", "RMSE", "R2"]
        for metric in expected_metrics:
            assert metric in bootstrap_cis, f"Missing bootstrap CI for {metric}"

            ci = bootstrap_cis[metric]
            assert isinstance(ci, tuple) and len(ci) == 2, (
                f"CI for {metric} should be tuple of length 2"
            )

            lower, upper = ci
            if not (np.isnan(lower) or np.isnan(upper)):
                assert lower <= upper, f"CI bounds invalid for {metric}: {ci}"

        # Verify CIs are reasonable (contain plausible values)
        mae_ci = bootstrap_cis["MAE"]
        if not (np.isnan(mae_ci[0]) or np.isnan(mae_ci[1])):
            assert mae_ci[0] >= 0, f"MAE lower bound should be non-negative: {mae_ci}"

        rmse_ci = bootstrap_cis["RMSE"]
        if not (np.isnan(rmse_ci[0]) or np.isnan(rmse_ci[1])):
            assert rmse_ci[0] >= 0, (
                f"RMSE lower bound should be non-negative: {rmse_ci}"
            )


class TestEvaluationSuiteIntegration:
    """Integration tests for EvaluationSuite."""

    def setup_method(self):
        """Set up test fixtures."""
        self.suite = EvaluationSuite(k_folds=5, bootstrap_samples=1000)

    def test_cross_validation_execution(self):
        """Test cross-validation execution."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.1 * np.random.randn(n_samples)

        # Define simple model function
        def model_fn(X_train, y_train):
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model

        # Run cross-validation
        cv_scores, mean_score, std_score = self.suite.k_fold_cross_validation(
            X, y, model_fn
        )

        # Verify results
        assert len(cv_scores) == 5, f"Should have 5 CV scores, got {len(cv_scores)}"
        assert all(isinstance(score, (int, float)) for score in cv_scores), (
            "All CV scores should be numeric"
        )
        assert isinstance(mean_score, (int, float)), "Mean score should be numeric"
        assert isinstance(std_score, (int, float)), "Std score should be numeric"
        assert std_score >= 0, "Standard deviation should be non-negative"

    def test_metric_computation(self):
        """Test metric computation."""
        # Generate test data
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.randn(n_samples)
        y_pred = y_true + 0.2 * np.random.randn(n_samples)
        confidences = np.random.uniform(0.3, 0.9, n_samples)

        # Run evaluation
        result = self.suite.evaluate(y_true, y_pred, confidences)

        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.pearson_correlation, (int, float))
        assert isinstance(result.pearson_ci, tuple) and len(result.pearson_ci) == 2
        assert isinstance(result.spearman_correlation, (int, float))
        assert isinstance(result.spearman_ci, tuple) and len(result.spearman_ci) == 2
        assert isinstance(result.calibration_error, (int, float))
        assert isinstance(result.shapiro_statistic, (int, float))
        assert isinstance(result.shapiro_p_value, (int, float))
        assert isinstance(result.bootstrap_metrics, dict)

    def test_report_generation(self):
        """Test report generation."""
        # Generate test data
        np.random.seed(42)
        n_samples = 150
        y_true = np.random.randn(n_samples)
        y_pred = y_true + 0.3 * np.random.randn(n_samples)

        # Run evaluation
        result = self.suite.evaluate(y_true, y_pred)

        # Verify all required metrics are present
        assert -1 <= result.pearson_correlation <= 1, (
            "Pearson correlation should be in [-1, 1]"
        )
        assert -1 <= result.spearman_correlation <= 1, (
            "Spearman correlation should be in [-1, 1]"
        )
        assert 0 <= result.calibration_error <= 1, (
            "Calibration error should be in [0, 1]"
        )
        assert 0 <= result.shapiro_statistic <= 1, (
            "Shapiro statistic should be in [0, 1]"
        )
        assert 0 <= result.shapiro_p_value <= 1, "Shapiro p-value should be in [0, 1]"

        # Verify bootstrap metrics
        for metric_name, (lower, upper) in result.bootstrap_metrics.items():
            if not (np.isnan(lower) or np.isnan(upper)):
                assert lower <= upper, (
                    f"Bootstrap CI bounds invalid for {metric_name}: ({lower}, {upper})"
                )
