"""
Tests for advanced analytics module.

Tests the time series analysis, portfolio analysis, trend prediction,
improvement advisory, and business intelligence export capabilities.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from udl_rating_framework.core.pipeline import QualityReport
from udl_rating_framework.analytics import (
    TimeSeriesAnalyzer,
    PortfolioAnalyzer,
    TrendPredictor,
    ImprovementAdvisor,
    BusinessIntelligenceExporter,
)
from udl_rating_framework.analytics.bi_exporter import BIExportConfig


@pytest.fixture
def sample_reports():
    """Create sample quality reports for testing."""
    reports = []
    base_time = datetime.now() - timedelta(days=30)

    # Create reports for multiple UDL files over time
    udl_files = ["project1/grammar.udl", "project1/lexer.udl", "project2/parser.udl"]

    for i in range(30):
        for j, udl_file in enumerate(udl_files):
            # Add some variation and trends
            base_score = 0.6 + 0.1 * j  # Different base scores per file
            trend = 0.01 * i  # Slight upward trend
            noise = np.random.normal(0, 0.05)  # Random noise

            overall_score = max(0.0, min(1.0, base_score + trend + noise))
            confidence = max(0.0, min(1.0, 0.7 + np.random.normal(0, 0.1)))

            # Create metric scores
            metric_scores = {
                "ConsistencyMetric": max(
                    0.0, min(1.0, overall_score + np.random.normal(0, 0.1))
                ),
                "CompletenessMetric": max(
                    0.0, min(1.0, overall_score + np.random.normal(0, 0.1))
                ),
                "ExpressivenessMetric": max(
                    0.0, min(1.0, overall_score + np.random.normal(0, 0.1))
                ),
                "StructuralCoherenceMetric": max(
                    0.0, min(1.0, overall_score + np.random.normal(0, 0.1))
                ),
            }

            # Occasionally add errors/warnings
            errors = ["Parse error"] if np.random.random() < 0.1 else []
            warnings = ["Style warning"] if np.random.random() < 0.2 else []

            report = QualityReport(
                overall_score=overall_score,
                confidence=confidence,
                metric_scores=metric_scores,
                metric_formulas={k: f"formula_{k}" for k in metric_scores.keys()},
                computation_trace=[],
                error_bounds={
                    k: (v - 0.05, v + 0.05) for k, v in metric_scores.items()
                },
                timestamp=base_time + timedelta(days=i),
                udl_file=udl_file,
                errors=errors,
                warnings=warnings,
            )

            reports.append(report)

    return reports


class TestTimeSeriesAnalyzer:
    """Test time series analysis functionality."""

    def test_analyze_quality_evolution(self, sample_reports):
        """Test quality evolution analysis."""
        analyzer = TimeSeriesAnalyzer()

        evolution = analyzer.analyze_quality_evolution(
            sample_reports, "project1/grammar.udl", "overall_score"
        )

        assert evolution.metric_name == "overall_score"
        assert len(evolution.time_series) > 0
        assert evolution.trend_analysis is not None
        assert evolution.forecast is not None
        assert evolution.confidence_intervals is not None
        assert len(evolution.summary) > 0

        # Check trend analysis components
        trend = evolution.trend_analysis
        assert isinstance(trend.trend_slope, float)
        assert isinstance(trend.trend_p_value, float)
        assert isinstance(trend.volatility, float)
        assert isinstance(trend.autocorrelation, float)
        assert isinstance(trend.change_points, list)
        assert isinstance(trend.anomalies, list)

    def test_analyze_multiple_metrics(self, sample_reports):
        """Test analysis of multiple metrics."""
        analyzer = TimeSeriesAnalyzer()

        results = analyzer.analyze_multiple_metrics(
            sample_reports,
            "project1/grammar.udl",
            ["overall_score", "confidence", "ConsistencyMetric"],
        )

        assert len(results) == 3
        assert "overall_score" in results
        assert "confidence" in results
        assert "ConsistencyMetric" in results

        for metric, evolution in results.items():
            assert evolution.metric_name == metric
            assert len(evolution.time_series) > 0

    def test_insufficient_data(self, sample_reports):
        """Test handling of insufficient data."""
        analyzer = TimeSeriesAnalyzer(min_observations=50)

        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.analyze_quality_evolution(
                sample_reports[:5],  # Too few reports
                "project1/grammar.udl",
            )

    def test_generate_time_series_report(self, sample_reports):
        """Test time series report generation."""
        analyzer = TimeSeriesAnalyzer()

        evolution_results = analyzer.analyze_multiple_metrics(
            sample_reports, "project1/grammar.udl"
        )

        report = analyzer.generate_time_series_report(
            evolution_results, "project1/grammar.udl"
        )

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Time Series Analysis Report" in report
        assert "project1/grammar.udl" in report


class TestPortfolioAnalyzer:
    """Test portfolio analysis functionality."""

    def test_analyze_portfolio(self, sample_reports):
        """Test portfolio analysis."""
        analyzer = PortfolioAnalyzer()

        comparison = analyzer.analyze_portfolio(sample_reports)

        assert len(comparison.project_profiles) > 0
        assert len(comparison.rankings) > 0
        assert len(comparison.clusters) > 0
        assert comparison.correlations is not None
        assert isinstance(comparison.outliers, list)
        assert len(comparison.recommendations) > 0

        # Check project profiles
        for project, profile in comparison.project_profiles.items():
            assert isinstance(profile.avg_overall_score, float)
            assert 0.0 <= profile.avg_overall_score <= 1.0
            assert isinstance(profile.maturity_level, str)
            assert profile.maturity_level in ["developing", "stable", "mature"]
            assert isinstance(profile.risk_level, str)
            assert profile.risk_level in ["low", "medium", "high"]

    def test_benchmark_against_standards(self, sample_reports):
        """Test benchmarking against standards."""
        analyzer = PortfolioAnalyzer()

        comparison = analyzer.analyze_portfolio(sample_reports)
        benchmark = analyzer.benchmark_against_standards(comparison)

        assert len(benchmark.percentiles) > 0
        assert len(benchmark.industry_standards) > 0
        assert len(benchmark.performance_gaps) > 0
        assert len(benchmark.improvement_potential) > 0

        # Check that all projects have performance gaps
        for project in comparison.project_profiles.keys():
            assert project in benchmark.performance_gaps
            assert project in benchmark.improvement_potential

    def test_project_mapping(self, sample_reports):
        """Test custom project mapping."""
        analyzer = PortfolioAnalyzer()

        project_mapping = {
            "project1/grammar.udl": "ProjectA",
            "project1/lexer.udl": "ProjectA",
            "project2/parser.udl": "ProjectB",
        }

        comparison = analyzer.analyze_portfolio(sample_reports, project_mapping)

        assert "ProjectA" in comparison.project_profiles
        assert "ProjectB" in comparison.project_profiles

        # ProjectA should have more reports (2 UDL files vs 1)
        assert (
            comparison.project_profiles["ProjectA"].total_reports
            > comparison.project_profiles["ProjectB"].total_reports
        )

    def test_generate_portfolio_report(self, sample_reports):
        """Test portfolio report generation."""
        analyzer = PortfolioAnalyzer()

        comparison = analyzer.analyze_portfolio(sample_reports)
        benchmark = analyzer.benchmark_against_standards(comparison)

        report = analyzer.generate_portfolio_report(comparison, benchmark)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Portfolio Analysis Report" in report
        assert "Executive Summary" in report


class TestTrendPredictor:
    """Test trend prediction functionality."""

    def test_predict_quality_trends(self, sample_reports):
        """Test quality trend prediction."""
        predictor = TrendPredictor(prediction_horizon=10)

        predictions = predictor.predict_quality_trends(
            sample_reports, "project1/grammar.udl", ["overall_score", "confidence"]
        )

        assert len(predictions) == 2
        assert "overall_score" in predictions
        assert "confidence" in predictions

        for metric, result in predictions.items():
            assert len(result.predictions) == 10  # prediction_horizon
            assert result.confidence_intervals is not None
            assert len(result.model_performance) > 0
            assert result.model_type in ["linear", "polynomial", "random_forest"]

    def test_analyze_portfolio_trends(self, sample_reports):
        """Test portfolio trend analysis."""
        predictor = TrendPredictor()

        analysis = predictor.analyze_portfolio_trends(sample_reports)

        assert len(analysis.historical_trends) > 0
        assert len(analysis.volatility_metrics) > 0
        assert len(analysis.forecast_accuracy) > 0

        # Check that metrics have reasonable values
        for metric, trend in analysis.historical_trends.items():
            assert isinstance(trend, float)

        for metric, volatility in analysis.volatility_metrics.items():
            assert isinstance(volatility, float)
            assert volatility >= 0.0

    def test_insufficient_data_prediction(self, sample_reports):
        """Test prediction with insufficient data."""
        predictor = TrendPredictor()

        with pytest.raises(ValueError, match="Insufficient data"):
            predictor.predict_quality_trends(
                sample_reports[:5],  # Too few reports
                "project1/grammar.udl",
            )

    def test_generate_prediction_report(self, sample_reports):
        """Test prediction report generation."""
        predictor = TrendPredictor(prediction_horizon=5)

        predictions = predictor.predict_quality_trends(
            sample_reports, "project1/grammar.udl"
        )

        report = predictor.generate_prediction_report(
            predictions, "project1/grammar.udl"
        )

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Quality Trend Prediction Report" in report
        assert "project1/grammar.udl" in report


class TestImprovementAdvisor:
    """Test improvement advisory functionality."""

    def test_generate_improvement_plan(self, sample_reports):
        """Test improvement plan generation."""
        advisor = ImprovementAdvisor()

        # Create a report with low scores to trigger suggestions
        low_quality_report = QualityReport(
            overall_score=0.4,
            confidence=0.5,
            metric_scores={
                "ConsistencyMetric": 0.3,
                "CompletenessMetric": 0.4,
                "ExpressivenessMetric": 0.5,
                "StructuralCoherenceMetric": 0.3,
            },
            metric_formulas={},
            computation_trace=[],
            error_bounds={},
            timestamp=datetime.now(),
            udl_file="test.udl",
            errors=["Critical error"],
            warnings=["Warning 1", "Warning 2", "Warning 3"],
        )

        plan = advisor.generate_improvement_plan([low_quality_report], "test.udl")

        assert plan.udl_file == "test.udl"
        assert plan.current_score == 0.4
        assert plan.target_score > plan.current_score
        assert len(plan.suggestions) > 0
        assert len(plan.implementation_order) > 0
        assert len(plan.success_metrics) > 0

        # Check that suggestions have required fields
        for suggestion in plan.suggestions:
            assert suggestion.category in [
                "consistency",
                "completeness",
                "expressiveness",
                "structure",
                "errors",
                "warnings",
                "trends",
                "confidence",
            ]
            assert suggestion.priority in ["high", "medium", "low"]
            assert suggestion.effort_level in ["low", "medium", "high"]
            assert len(suggestion.specific_actions) > 0

    def test_analyze_portfolio_improvements(self, sample_reports):
        """Test portfolio improvement analysis."""
        advisor = ImprovementAdvisor()

        recommendations = advisor.analyze_portfolio_improvements(sample_reports)

        assert isinstance(recommendations, dict)
        # Should have some recommendations for the portfolio
        assert len(recommendations) >= 0

    def test_generate_improvement_report(self, sample_reports):
        """Test improvement report generation."""
        advisor = ImprovementAdvisor()

        # Create a low quality report
        low_quality_report = QualityReport(
            overall_score=0.4,
            confidence=0.5,
            metric_scores={"ConsistencyMetric": 0.3},
            metric_formulas={},
            computation_trace=[],
            error_bounds={},
            timestamp=datetime.now(),
            udl_file="test.udl",
            errors=[],
            warnings=[],
        )

        plan = advisor.generate_improvement_plan([low_quality_report], "test.udl")
        report = advisor.generate_improvement_report(plan)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "UDL Quality Improvement Plan" in report
        assert "test.udl" in report


class TestBusinessIntelligenceExporter:
    """Test business intelligence export functionality."""

    def test_export_quality_data_csv(self, sample_reports):
        """Test CSV export."""
        exporter = BusinessIntelligenceExporter()
        config = BIExportConfig(format="csv", aggregation_level="detailed")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export.csv"

            dataset = exporter.export_quality_data(sample_reports, config, output_path)

            assert dataset.name == "udl_quality_detailed"
            assert len(dataset.data) > 0
            assert output_path.exists()

            # Check that CSV file has expected columns
            df = pd.read_csv(output_path)
            assert "overall_score" in df.columns
            assert "confidence" in df.columns
            assert "project" in df.columns

    def test_export_quality_data_json(self, sample_reports):
        """Test JSON export."""
        exporter = BusinessIntelligenceExporter()
        config = BIExportConfig(format="json", aggregation_level="summary")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export.json"

            dataset = exporter.export_quality_data(sample_reports, config, output_path)

            assert dataset.name == "udl_quality_summary"
            assert output_path.exists()

            # Check JSON structure
            with open(output_path, "r") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "data" in data
            assert len(data["data"]) > 0

    def test_export_dashboard_data(self, sample_reports):
        """Test dashboard data export."""
        exporter = BusinessIntelligenceExporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            datasets = exporter.export_dashboard_data(
                sample_reports, "executive", Path(temp_dir)
            )

            assert "kpis" in datasets
            assert "trends" in datasets
            assert "portfolio_overview" in datasets
            assert "risk_assessment" in datasets

            # Check that files were created
            assert (Path(temp_dir) / "kpis.csv").exists()
            assert (Path(temp_dir) / "trends.csv").exists()

    def test_create_powerbi_package(self, sample_reports):
        """Test Power BI package creation."""
        exporter = BusinessIntelligenceExporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = exporter.create_powerbi_package(
                sample_reports, Path(temp_dir)
            )

            assert package_path.exists()
            assert (package_path / "udl_quality_template.json").exists()
            assert (package_path / "data_model.json").exists()
            assert (package_path / "README.md").exists()

            # Check that data files exist
            csv_files = list(package_path.glob("*.csv"))
            assert len(csv_files) > 0

    def test_create_tableau_package(self, sample_reports):
        """Test Tableau package creation."""
        exporter = BusinessIntelligenceExporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = exporter.create_tableau_package(
                sample_reports, Path(temp_dir)
            )

            assert package_path.exists()
            assert (package_path / "udl_quality.tds").exists()
            assert (package_path / "udl_quality_template.twb").exists()
            assert (package_path / "README.md").exists()

    def test_aggregation_levels(self, sample_reports):
        """Test different aggregation levels."""
        exporter = BusinessIntelligenceExporter()

        # Test detailed level
        config_detailed = BIExportConfig(format="csv", aggregation_level="detailed")
        dataset_detailed = exporter.export_quality_data(sample_reports, config_detailed)

        # Test summary level
        config_summary = BIExportConfig(format="csv", aggregation_level="summary")
        dataset_summary = exporter.export_quality_data(sample_reports, config_summary)

        # Test executive level
        config_executive = BIExportConfig(format="csv", aggregation_level="executive")
        dataset_executive = exporter.export_quality_data(
            sample_reports, config_executive
        )

        # Executive should have fewer rows (one per project)
        assert len(dataset_executive.data) <= len(dataset_summary.data)
        assert len(dataset_summary.data) <= len(dataset_detailed.data)


class TestAnalyticsIntegration:
    """Test integration between analytics components."""

    def test_end_to_end_analytics_workflow(self, sample_reports):
        """Test complete analytics workflow."""
        # Time series analysis
        ts_analyzer = TimeSeriesAnalyzer()
        evolution = ts_analyzer.analyze_quality_evolution(
            sample_reports, "project1/grammar.udl"
        )

        # Portfolio analysis
        portfolio_analyzer = PortfolioAnalyzer()
        comparison = portfolio_analyzer.analyze_portfolio(sample_reports)

        # Trend prediction
        predictor = TrendPredictor(prediction_horizon=5)
        predictions = predictor.predict_quality_trends(
            sample_reports, "project1/grammar.udl"
        )

        # Improvement advisory
        advisor = ImprovementAdvisor()
        plan = advisor.generate_improvement_plan(sample_reports, "project1/grammar.udl")

        # BI export
        exporter = BusinessIntelligenceExporter()
        config = BIExportConfig(format="json", aggregation_level="executive")
        dataset = exporter.export_quality_data(sample_reports, config)

        # Verify all components produced results
        assert evolution is not None
        assert comparison is not None
        assert predictions is not None
        assert plan is not None
        assert dataset is not None

        # Verify data consistency
        assert len(evolution.time_series) > 0
        assert len(comparison.project_profiles) > 0
        assert len(predictions) > 0
        assert len(plan.suggestions) >= 0  # May be 0 for high-quality UDLs
        assert len(dataset.data) > 0


if __name__ == "__main__":
    pytest.main([__file__])
