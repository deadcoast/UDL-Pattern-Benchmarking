"""
Tests for comparison engine functionality.

Tests both property-based and unit tests for UDL comparison and statistical analysis.
"""

from datetime import datetime
from typing import List

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from udl_rating_framework.core.pipeline import ComputationStep, QualityReport
from udl_rating_framework.evaluation.comparison import (
    ComparisonEngine,
)


# Test data generators
@composite
def quality_report_strategy(draw, min_score=0.0, max_score=1.0, num_metrics=4):
    """Generate a QualityReport for testing."""
    overall_score = draw(
        st.floats(
            min_value=min_score,
            max_value=max_score,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    confidence = draw(
        st.floats(min_value=0.0, max_value=1.0,
                  allow_nan=False, allow_infinity=False)
    )

    # Generate metric scores
    metric_names = [f"metric_{i}" for i in range(num_metrics)]
    metric_scores = {
        name: draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        )
        for name in metric_names
    }

    # Generate metric formulas
    metric_formulas = {name: f"formula_{name}" for name in metric_names}

    # Generate computation trace
    computation_trace = [
        ComputationStep(
            step_number=1,
            operation="test_operation",
            formula="test_formula",
            inputs={"input": 1.0},
            output=overall_score,
        )
    ]

    # Generate error bounds
    error_bounds = {
        name: (max(0.0, score - 0.1), min(1.0, score + 0.1))
        for name, score in metric_scores.items()
    }

    udl_file = draw(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        )
    )

    return QualityReport(
        overall_score=overall_score,
        confidence=confidence,
        metric_scores=metric_scores,
        metric_formulas=metric_formulas,
        computation_trace=computation_trace,
        error_bounds=error_bounds,
        timestamp=datetime.now(),
        udl_file=f"{udl_file}.udl",
    )


@composite
def multiple_reports_strategy(draw, min_reports=2, max_reports=10):
    """Generate multiple QualityReports for comparison testing."""
    num_reports = draw(st.integers(
        min_value=min_reports, max_value=max_reports))
    reports = []

    for i in range(num_reports):
        report = draw(quality_report_strategy())
        # Ensure unique filenames
        report.udl_file = f"udl_{i}_{report.udl_file}"
        reports.append(report)

    return reports


class TestComparisonEngineProperties:
    """Property-based tests for ComparisonEngine."""

    @given(multiple_reports_strategy(min_reports=2, max_reports=5))
    @settings(max_examples=50, deadline=5000)
    def test_property_29_consistent_rating_procedures(
        self, reports: List[QualityReport]
    ):
        """
        **Feature: udl-rating-framework, Property 29: Consistent Rating Procedures**

        Property: For any multiple UDLs, rating procedures must be identical.
        Validates: Requirements 8.1
        """
        engine = ComparisonEngine()

        # Perform comparison
        summary = engine.compare_udls(reports)

        # Verify that all pairwise comparisons use the same procedure
        # This means all comparisons should have the same structure and computation method

        # Check that all pairwise results have the required fields
        for result in summary.pairwise_results:
            # Verify all required fields are present and computed
            assert isinstance(result.difference, float)
            assert isinstance(result.ttest_statistic, float)
            assert isinstance(result.ttest_pvalue, float)
            assert isinstance(result.cohens_d, float)
            assert isinstance(result.is_significant, bool)
            assert isinstance(result.effect_size_interpretation, str)

            # Verify the procedure is consistent (same computation method)
            # The difference should be computed as score1 - score2
            expected_diff = result.score1 - result.score2
            assert abs(result.difference - expected_diff) < 1e-10, (
                f"Inconsistent difference computation: {result.difference} != {expected_diff}"
            )

        # Verify all rankings use the same procedure
        for ranking in summary.rankings:
            assert isinstance(ranking.score, float)
            assert isinstance(ranking.rank, int)
            assert ranking.rank >= 1
            assert ranking.rank <= len(reports)
            assert isinstance(ranking.confidence_interval, tuple)
            assert len(ranking.confidence_interval) == 2

    @given(multiple_reports_strategy(min_reports=2, max_reports=5))
    @settings(max_examples=50, deadline=5000)
    def test_property_30_pairwise_difference_computation(
        self, reports: List[QualityReport]
    ):
        """
        **Feature: udl-rating-framework, Property 30: Pairwise Difference Computation**

        Property: For any UDL pairs, Δ_ij = Q_i - Q_j must be computed correctly.
        Validates: Requirements 8.2
        """
        engine = ComparisonEngine()

        # Perform comparison
        summary = engine.compare_udls(reports)

        # Verify pairwise difference computation
        for result in summary.pairwise_results:
            # The difference should be exactly score1 - score2
            expected_difference = result.score1 - result.score2

            assert abs(result.difference - expected_difference) < 1e-10, (
                f"Incorrect pairwise difference: got {result.difference}, expected {expected_difference}"
            )

            # Verify scores are within valid range [0, 1]
            assert 0.0 <= result.score1 <= 1.0, f"Score1 out of range: {result.score1}"
            assert 0.0 <= result.score2 <= 1.0, f"Score2 out of range: {result.score2}"

            # Verify difference is within valid range [-1, 1]
            assert -1.0 <= result.difference <= 1.0, (
                f"Difference out of range: {result.difference}"
            )

    @given(multiple_reports_strategy(min_reports=2, max_reports=5))
    @settings(max_examples=50, deadline=5000)
    def test_property_31_statistical_significance_testing(
        self, reports: List[QualityReport]
    ):
        """
        **Feature: udl-rating-framework, Property 31: Statistical Significance Testing**

        Property: For any UDL comparisons, statistical tests must be applied and p-values reported.
        Validates: Requirements 8.3
        """
        engine = ComparisonEngine()

        # Perform comparison
        summary = engine.compare_udls(reports)

        # Verify statistical significance testing
        for result in summary.pairwise_results:
            # T-test results must be present and valid
            assert isinstance(result.ttest_statistic, float)
            assert isinstance(result.ttest_pvalue, float)
            assert not np.isnan(result.ttest_statistic)
            assert not np.isnan(result.ttest_pvalue)
            assert 0.0 <= result.ttest_pvalue <= 1.0, (
                f"Invalid t-test p-value: {result.ttest_pvalue}"
            )

            # Wilcoxon test results (may be None for insufficient data)
            if result.wilcoxon_pvalue is not None:
                assert isinstance(result.wilcoxon_statistic,
                                  (float, type(None)))
                assert isinstance(result.wilcoxon_pvalue, float)
                assert 0.0 <= result.wilcoxon_pvalue <= 1.0, (
                    f"Invalid Wilcoxon p-value: {result.wilcoxon_pvalue}"
                )

            # Significance determination should be based on p-values
            min_pvalue = min(result.ttest_pvalue,
                             result.wilcoxon_pvalue or 1.0)
            expected_significance = min_pvalue < engine.alpha
            assert result.is_significant == expected_significance, (
                f"Incorrect significance determination: {result.is_significant} != {expected_significance}"
            )

    @given(multiple_reports_strategy(min_reports=2, max_reports=5))
    @settings(max_examples=50, deadline=5000)
    def test_property_32_effect_size_computation(self, reports: List[QualityReport]):
        """
        **Feature: udl-rating-framework, Property 32: Effect Size Computation**

        Property: For any UDL comparisons, Cohen's d must be computed correctly.
        Validates: Requirements 8.4
        """
        engine = ComparisonEngine()

        # Perform comparison
        summary = engine.compare_udls(reports)

        # Verify effect size computation
        for result in summary.pairwise_results:
            # Cohen's d must be present and finite
            assert isinstance(result.cohens_d, float)
            assert not np.isnan(result.cohens_d)
            assert not np.isinf(result.cohens_d)

            # Effect size interpretation must be valid
            valid_interpretations = {"negligible", "small", "medium", "large"}
            assert result.effect_size_interpretation in valid_interpretations, (
                f"Invalid effect size interpretation: {result.effect_size_interpretation}"
            )

            # Verify interpretation matches Cohen's d magnitude
            abs_d = abs(result.cohens_d)
            if abs_d < 0.2:
                assert result.effect_size_interpretation == "negligible"
            elif abs_d < 0.5:
                assert result.effect_size_interpretation == "small"
            elif abs_d < 0.8:
                assert result.effect_size_interpretation == "medium"
            else:
                assert result.effect_size_interpretation == "large"


class TestComparisonEngineUnitTests:
    """Unit tests for ComparisonEngine."""

    def test_pairwise_comparisons(self):
        """Test pairwise comparisons with known values."""
        # Create test reports with known scores
        reports = [
            QualityReport(
                overall_score=0.8,
                confidence=0.9,
                metric_scores={"metric1": 0.7, "metric2": 0.9, "metric3": 0.8},
                metric_formulas={"metric1": "f1",
                                 "metric2": "f2", "metric3": "f3"},
                computation_trace=[],
                error_bounds={},
                timestamp=datetime.now(),
                udl_file="udl1.udl",
            ),
            QualityReport(
                overall_score=0.6,
                confidence=0.8,
                metric_scores={"metric1": 0.5, "metric2": 0.7, "metric3": 0.6},
                metric_formulas={"metric1": "f1",
                                 "metric2": "f2", "metric3": "f3"},
                computation_trace=[],
                error_bounds={},
                timestamp=datetime.now(),
                udl_file="udl2.udl",
            ),
            QualityReport(
                overall_score=0.9,
                confidence=0.95,
                metric_scores={"metric1": 0.9, "metric2": 0.9, "metric3": 0.9},
                metric_formulas={"metric1": "f1",
                                 "metric2": "f2", "metric3": "f3"},
                computation_trace=[],
                error_bounds={},
                timestamp=datetime.now(),
                udl_file="udl3.udl",
            ),
        ]

        engine = ComparisonEngine()
        summary = engine.compare_udls(reports)

        # Verify number of pairwise comparisons: C(3,2) = 3
        assert len(summary.pairwise_results) == 3

        # Verify specific differences
        differences = {
            (r.udl1_name, r.udl2_name): r.difference for r in summary.pairwise_results
        }

        # Check expected differences
        assert abs(differences[("udl1.udl", "udl2.udl")
                               ] - 0.2) < 1e-10  # 0.8 - 0.6
        assert abs(differences[("udl1.udl", "udl3.udl")
                               ] - (-0.1)) < 1e-10  # 0.8 - 0.9
        assert abs(differences[("udl2.udl", "udl3.udl")
                               ] - (-0.3)) < 1e-10  # 0.6 - 0.9

    def test_ranking_generation(self):
        """Test ranking generation with confidence intervals."""
        # Create test reports with different scores
        reports = [
            QualityReport(
                overall_score=0.6,  # Should be rank 3
                confidence=0.8,
                metric_scores={"metric1": 0.5, "metric2": 0.7},
                metric_formulas={"metric1": "f1", "metric2": "f2"},
                computation_trace=[],
                error_bounds={},
                timestamp=datetime.now(),
                udl_file="low_score.udl",
            ),
            QualityReport(
                overall_score=0.9,  # Should be rank 1
                confidence=0.95,
                metric_scores={"metric1": 0.9, "metric2": 0.9},
                metric_formulas={"metric1": "f1", "metric2": "f2"},
                computation_trace=[],
                error_bounds={},
                timestamp=datetime.now(),
                udl_file="high_score.udl",
            ),
            QualityReport(
                overall_score=0.75,  # Should be rank 2
                confidence=0.85,
                metric_scores={"metric1": 0.7, "metric2": 0.8},
                metric_formulas={"metric1": "f1", "metric2": "f2"},
                computation_trace=[],
                error_bounds={},
                timestamp=datetime.now(),
                udl_file="medium_score.udl",
            ),
        ]

        engine = ComparisonEngine()
        summary = engine.compare_udls(reports)

        # Verify rankings are sorted by rank
        rankings = summary.rankings
        assert len(rankings) == 3

        # Check ranking order (sorted by rank)
        assert rankings[0].rank == 1
        assert rankings[1].rank == 2
        assert rankings[2].rank == 3

        # Check that higher scores get better ranks
        assert rankings[0].udl_name == "high_score.udl"
        assert rankings[1].udl_name == "medium_score.udl"
        assert rankings[2].udl_name == "low_score.udl"

        # Verify confidence intervals are present
        for ranking in rankings:
            assert isinstance(ranking.confidence_interval, tuple)
            assert len(ranking.confidence_interval) == 2
            assert (
                ranking.confidence_interval[0]
                <= ranking.score
                <= ranking.confidence_interval[1]
            )

    def test_confidence_interval_computation(self):
        """Test confidence interval computation for rankings."""
        # Create a report with multiple metrics for bootstrap
        report = QualityReport(
            overall_score=0.8,
            confidence=0.9,
            metric_scores={f"metric_{i}": 0.8 +
                           0.1 * np.sin(i) for i in range(10)},
            metric_formulas={f"metric_{i}": f"formula_{i}" for i in range(10)},
            computation_trace=[],
            error_bounds={},
            timestamp=datetime.now(),
            udl_file="test.udl",
        )

        engine = ComparisonEngine(bootstrap_samples=100)  # Smaller for testing
        summary = engine.compare_udls(
            [report, report]
        )  # Need at least 2 for comparison

        # Verify confidence intervals are computed
        ranking = summary.rankings[0]  # First ranking
        ci_lower, ci_upper = ranking.confidence_interval

        assert ci_lower <= ranking.score <= ci_upper
        assert ci_lower >= 0.0
        assert ci_upper <= 1.0

    def test_insufficient_reports_error(self):
        """Test error handling for insufficient reports."""
        engine = ComparisonEngine()

        # Test with no reports
        with pytest.raises(ValueError, match="At least 2 UDL reports required"):
            engine.compare_udls([])

        # Test with single report
        report = QualityReport(
            overall_score=0.8,
            confidence=0.9,
            metric_scores={"metric1": 0.8},
            metric_formulas={"metric1": "f1"},
            computation_trace=[],
            error_bounds={},
            timestamp=datetime.now(),
            udl_file="single.udl",
        )

        with pytest.raises(ValueError, match="At least 2 UDL reports required"):
            engine.compare_udls([report])

    def test_summary_statistics(self):
        """Test summary statistics computation."""
        reports = [
            QualityReport(
                overall_score=0.6,
                confidence=0.8,
                metric_scores={"metric1": 0.6},
                metric_formulas={"metric1": "f1"},
                computation_trace=[],
                error_bounds={},
                timestamp=datetime.now(),
                udl_file="udl1.udl",
            ),
            QualityReport(
                overall_score=0.8,
                confidence=0.9,
                metric_scores={"metric1": 0.8},
                metric_formulas={"metric1": "f1"},
                computation_trace=[],
                error_bounds={},
                timestamp=datetime.now(),
                udl_file="udl2.udl",
            ),
        ]

        engine = ComparisonEngine()
        summary = engine.compare_udls(reports)

        # Verify summary statistics
        stats = summary.score_statistics

        assert abs(stats["mean_score"] - 0.7) < 1e-10  # (0.6 + 0.8) / 2
        assert abs(stats["min_score"] - 0.6) < 1e-10
        assert abs(stats["max_score"] - 0.8) < 1e-10
        assert abs(stats["median_score"] - 0.7) < 1e-10
        assert abs(stats["score_range"] - 0.2) < 1e-10  # 0.8 - 0.6

        # Verify comparison counts
        assert summary.total_comparisons == 1  # C(2,2) = 1
        assert summary.significant_comparisons <= 1
        assert isinstance(summary.mean_effect_size, float)
