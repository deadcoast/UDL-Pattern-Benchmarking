"""
Tests for the rating computation pipeline.

Tests both unit functionality and property-based correctness.
"""

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from udl_rating_framework.core.metrics.base import MetricRegistry
from udl_rating_framework.core.pipeline import (
    ComputationStep,
    QualityReport,
    RatingPipeline,
)
from udl_rating_framework.core.representation import UDLRepresentation


class TestRatingPipeline:
    """Test suite for RatingPipeline class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Ensure we have some metrics registered for testing
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric

        # Clear registry and register test metrics
        MetricRegistry.clear()
        MetricRegistry.register("consistency", ConsistencyMetric)
        MetricRegistry.register("completeness", CompletenessMetric)

    def teardown_method(self):
        """Clean up after tests."""
        MetricRegistry.clear()

    def create_test_udl(self, content: str = None) -> UDLRepresentation:
        """Create a test UDL representation."""
        if content is None:
            content = """
            # Simple test grammar
            expr ::= term '+' expr | term
            term ::= factor '*' term | factor  
            factor ::= '(' expr ')' | number
            number ::= digit+
            digit ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
            """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".udl", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            udl = UDLRepresentation(content, temp_path)
            return udl
        finally:
            os.unlink(temp_path)

    def test_pipeline_initialization(self):
        """Test pipeline initialization with valid metrics."""
        pipeline = RatingPipeline(
            metric_names=["consistency", "completeness"],
            weights={"consistency": 0.6, "completeness": 0.4},
        )

        assert len(pipeline.metrics) == 2
        assert "consistency" in pipeline.metrics
        assert "completeness" in pipeline.metrics
        assert pipeline.aggregator.weights["consistency"] == 0.6
        assert pipeline.aggregator.weights["completeness"] == 0.4

    def test_pipeline_initialization_equal_weights(self):
        """Test pipeline initialization with default equal weights."""
        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])

        assert pipeline.aggregator.weights["consistency"] == 0.5
        assert pipeline.aggregator.weights["completeness"] == 0.5

    def test_pipeline_initialization_invalid_metric(self):
        """Test pipeline initialization with invalid metric name."""
        with pytest.raises(ValueError, match="Unknown metric"):
            RatingPipeline(metric_names=["nonexistent_metric"])

    def test_compute_rating_basic(self):
        """Test basic rating computation."""
        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])
        udl = self.create_test_udl()

        report = pipeline.compute_rating(udl)

        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall_score <= 1.0
        assert 0.0 <= report.confidence <= 1.0
        assert "consistency" in report.metric_scores
        assert "completeness" in report.metric_scores
        assert len(report.computation_trace) > 0

    def test_compute_rating_with_tracing_disabled(self):
        """Test rating computation with tracing disabled."""
        pipeline = RatingPipeline(
            metric_names=["consistency", "completeness"], enable_tracing=False
        )
        udl = self.create_test_udl()

        report = pipeline.compute_rating(udl)

        assert isinstance(report, QualityReport)
        assert len(report.computation_trace) == 0

    def test_compute_batch_ratings(self):
        """Test batch rating computation."""
        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])

        udls = [
            self.create_test_udl("expr ::= 'a'"),
            self.create_test_udl("stmt ::= 'b'"),
            self.create_test_udl("rule ::= 'c'"),
        ]

        reports = pipeline.compute_batch_ratings(udls)

        assert len(reports) == 3
        for report in reports:
            assert isinstance(report, QualityReport)
            assert 0.0 <= report.overall_score <= 1.0
            assert 0.0 <= report.confidence <= 1.0

    def test_get_available_metrics(self):
        """Test getting available metrics information."""
        pipeline = RatingPipeline(metric_names=["consistency"])

        available = pipeline.get_available_metrics()

        assert isinstance(available, dict)
        assert "consistency" in available
        assert "completeness" in available

    def test_validate_pipeline(self):
        """Test pipeline validation."""
        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])
        udl = self.create_test_udl()

        results = pipeline.validate_pipeline(udl)

        assert isinstance(results, dict)
        assert "consistency_computable" in results
        assert "completeness_computable" in results
        assert "aggregation_works" in results
        assert "confidence_works" in results
        assert "pipeline_works" in results


class TestIndependentMetricComputation:
    """
    Property-based tests for independent metric computation.

    **Feature: udl-rating-framework, Property 18: Independent Metric Computation**
    **Validates: Requirements 5.1**
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Ensure we have metrics registered for testing
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
        from udl_rating_framework.core.metrics.expressiveness import (
            ExpressivenessMetric,
        )

        # Clear registry and register test metrics
        MetricRegistry.clear()
        MetricRegistry.register("consistency", ConsistencyMetric)
        MetricRegistry.register("completeness", CompletenessMetric)
        MetricRegistry.register("expressiveness", ExpressivenessMetric)

    def teardown_method(self):
        """Clean up after tests."""
        MetricRegistry.clear()

    def create_test_udl_from_content(self, content: str) -> UDLRepresentation:
        """Create a test UDL representation from content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".udl", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            udl = UDLRepresentation(content, temp_path)
            return udl
        finally:
            os.unlink(temp_path)

    @given(
        st.lists(
            st.sampled_from(
                [
                    "expr ::= term '+' expr | term",
                    "term ::= factor '*' term | factor",
                    "factor ::= '(' expr ')' | number",
                    "number ::= digit+",
                    "digit ::= '0' | '1' | '2'",
                    "stmt ::= 'if' expr 'then' stmt",
                    "stmt ::= 'while' expr 'do' stmt",
                    "stmt ::= 'begin' stmts 'end'",
                    "stmts ::= stmt ';' stmts | stmt",
                ]
            ),
            min_size=1,
            max_size=10,
        )
    )
    def test_metrics_computed_independently(self, grammar_rules):
        """
        Property: Metrics can be computed independently without side effects.

        For any UDL and any set of metrics, computing each metric independently
        should produce the same results as computing them together, and there
        should be no side effects between metric computations.
        """
        # Create UDL content from grammar rules
        content = "\n".join(grammar_rules)
        udl = self.create_test_udl_from_content(content)

        # Test with different metric combinations
        all_metrics = ["consistency", "completeness", "expressiveness"]

        # Compute each metric individually
        individual_results = {}
        for metric_name in all_metrics:
            try:
                pipeline = RatingPipeline(metric_names=[metric_name])
                report = pipeline.compute_rating(udl)
                individual_results[metric_name] = report.metric_scores.get(
                    metric_name, 0.0
                )
            except Exception:
                # Skip metrics that fail on this particular UDL
                continue

        # Skip if no metrics could be computed
        assume(len(individual_results) >= 2)

        # Compute all metrics together
        working_metrics = list(individual_results.keys())
        pipeline_combined = RatingPipeline(metric_names=working_metrics)
        combined_report = pipeline_combined.compute_rating(udl)

        # Verify that individual and combined results match
        for metric_name in working_metrics:
            individual_value = individual_results[metric_name]
            combined_value = combined_report.metric_scores.get(
                metric_name, 0.0)

            # Allow small numerical differences due to floating point precision
            assert abs(individual_value - combined_value) < 1e-10, (
                f"Metric {metric_name} differs: individual={individual_value}, combined={combined_value}"
            )

        # Verify no side effects: compute metrics in different orders
        for i in range(min(3, len(working_metrics))):  # Test up to 3 permutations
            # Rotate the metric order
            rotated_metrics = working_metrics[i:] + working_metrics[:i]
            pipeline_rotated = RatingPipeline(metric_names=rotated_metrics)
            rotated_report = pipeline_rotated.compute_rating(udl)

            # Results should be identical regardless of computation order
            for metric_name in working_metrics:
                original_value = combined_report.metric_scores.get(
                    metric_name, 0.0)
                rotated_value = rotated_report.metric_scores.get(
                    metric_name, 0.0)

                assert abs(original_value - rotated_value) < 1e-10, (
                    f"Metric {metric_name} order-dependent: original={original_value}, rotated={rotated_value}"
                )

    @given(
        st.text(
            alphabet=st.characters(
                whitelist_categories=(
                    "Lu", "Ll", "Nd", "Pc", "Pd", "Ps", "Pe", "Po")
            ),
            min_size=10,
            max_size=200,
        )
    )
    def test_metric_isolation_with_random_content(self, random_content):
        """
        Property: Metrics remain isolated even with random/malformed content.

        Even when processing random or malformed UDL content, metrics should
        not interfere with each other's computation.
        """
        # Add some basic UDL structure to random content
        structured_content = f"""
        # Random content test
        rule1 ::= '{random_content[:20]}'
        rule2 ::= 'test' | 'content'
        {random_content}
        """

        try:
            udl = self.create_test_udl_from_content(structured_content)
        except Exception:
            # Skip if UDL creation fails with this content
            assume(False)

        # Test that metrics can be computed independently even with problematic content
        metrics_to_test = ["consistency", "completeness"]

        results = {}
        for metric_name in metrics_to_test:
            try:
                pipeline = RatingPipeline(metric_names=[metric_name])
                report = pipeline.compute_rating(udl)

                # Metric should produce a valid result or fail gracefully
                if metric_name in report.metric_scores:
                    value = report.metric_scores[metric_name]
                    assert 0.0 <= value <= 1.0, (
                        f"Metric {metric_name} out of bounds: {value}"
                    )
                    results[metric_name] = value

            except Exception as e:
                # Metrics should fail gracefully, not crash the system
                assert isinstance(e, (ValueError, TypeError, AttributeError)), (
                    f"Metric {metric_name} failed with unexpected error: {type(e).__name__}"
                )

        # If multiple metrics succeeded, verify they don't interfere
        if len(results) >= 2:
            # Compute together and verify independence
            working_metrics = list(results.keys())
            try:
                pipeline_combined = RatingPipeline(
                    metric_names=working_metrics)
                combined_report = pipeline_combined.compute_rating(udl)

                for metric_name in working_metrics:
                    if metric_name in combined_report.metric_scores:
                        individual_value = results[metric_name]
                        combined_value = combined_report.metric_scores[metric_name]

                        # Values should be close (allowing for numerical precision)
                        assert abs(individual_value - combined_value) < 1e-6, (
                            f"Metric {metric_name} interference detected"
                        )

            except Exception:
                # Combined computation may fail, but individual metrics should still work
                pass

    def test_metric_state_isolation(self):
        """
        Property: Metrics maintain state isolation between computations.

        Computing metrics on different UDLs should not affect each other.
        """
        # Create two different UDLs
        udl1 = self.create_test_udl_from_content("rule1 ::= 'a' | 'b'")
        udl2 = self.create_test_udl_from_content("rule2 ::= 'x' | 'y' | 'z'")

        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])

        # Compute metrics for first UDL
        report1_first = pipeline.compute_rating(udl1)

        # Compute metrics for second UDL
        report2 = pipeline.compute_rating(udl2)

        # Compute metrics for first UDL again
        report1_second = pipeline.compute_rating(udl1)

        # Results for UDL1 should be identical before and after processing UDL2
        for metric_name in ["consistency", "completeness"]:
            if (
                metric_name in report1_first.metric_scores
                and metric_name in report1_second.metric_scores
            ):
                value_first = report1_first.metric_scores[metric_name]
                value_second = report1_second.metric_scores[metric_name]

                assert abs(value_first - value_second) < 1e-10, (
                    f"Metric {metric_name} state not isolated: {value_first} != {value_second}"
                )


class TestResultAggregation:
    """
    Property-based tests for result aggregation.

    **Feature: udl-rating-framework, Property 8: Result Aggregation**
    **Validates: Requirements 2.4**
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Ensure we have metrics registered for testing
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
        from udl_rating_framework.core.metrics.expressiveness import (
            ExpressivenessMetric,
        )

        # Clear registry and register test metrics
        MetricRegistry.clear()
        MetricRegistry.register("consistency", ConsistencyMetric)
        MetricRegistry.register("completeness", CompletenessMetric)
        MetricRegistry.register("expressiveness", ExpressivenessMetric)

    def teardown_method(self):
        """Clean up after tests."""
        MetricRegistry.clear()

    def create_test_udl_from_content(self, content: str) -> UDLRepresentation:
        """Create a test UDL representation from content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".udl", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            udl = UDLRepresentation(content, temp_path)
            return udl
        finally:
            os.unlink(temp_path)

    @given(
        st.lists(
            st.lists(
                st.sampled_from(
                    [
                        "expr ::= term '+' expr | term",
                        "term ::= factor '*' term | factor",
                        "factor ::= '(' expr ')' | number",
                        "number ::= digit+",
                        "digit ::= '0' | '1' | '2'",
                        "stmt ::= 'if' expr 'then' stmt",
                        "stmt ::= 'while' expr 'do' stmt",
                        "rule ::= 'simple'",
                        "simple ::= 'test'",
                    ]
                ),
                min_size=1,
                max_size=5,
            ),
            min_size=2,
            max_size=10,
        )
    )
    def test_all_results_included_in_summary(self, udl_contents_list):
        """
        Property: All successfully processed UDL files are included in summary.

        When processing multiple UDL files, the system must produce a summary
        report containing results for all successfully processed files.
        """
        # Create multiple UDLs from the generated content
        udls = []
        expected_files = []

        for i, grammar_rules in enumerate(udl_contents_list):
            content = f"# UDL {i}\n" + "\n".join(grammar_rules)
            try:
                udl = self.create_test_udl_from_content(content)
                udls.append(udl)
                expected_files.append(udl.file_path)
            except Exception:
                # Skip UDLs that can't be created
                continue

        # Skip if we don't have enough UDLs to test
        assume(len(udls) >= 2)

        # Process all UDLs using batch processing
        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])
        reports = pipeline.compute_batch_ratings(udls)

        # Verify all UDLs are represented in the results
        assert len(reports) == len(udls), (
            f"Expected {len(udls)} reports, got {len(reports)}"
        )

        # Verify each UDL file is represented
        report_files = [report.udl_file for report in reports]
        for expected_file in expected_files:
            assert expected_file in report_files, (
                f"UDL file {expected_file} missing from results"
            )

        # Verify each report has the expected structure
        for i, report in enumerate(reports):
            assert isinstance(report, QualityReport), (
                f"Report {i} is not a QualityReport instance"
            )

            assert hasattr(
                report, "overall_score"), f"Report {i} missing overall_score"

            assert hasattr(
                report, "confidence"), f"Report {i} missing confidence"

            assert hasattr(
                report, "metric_scores"), f"Report {i} missing metric_scores"

            assert hasattr(report, "udl_file"), f"Report {i} missing udl_file"

            # Verify the file path matches
            assert report.udl_file == expected_files[i], (
                f"Report {i} file path mismatch: expected {expected_files[i]}, got {report.udl_file}"
            )

    @given(
        st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=(
                        "Lu",
                        "Ll",
                        "Nd",
                        "Pc",
                        "Pd",
                        "Ps",
                        "Pe",
                        "Po",
                    )
                ),
                min_size=5,
                max_size=50,
            ),
            min_size=3,
            max_size=8,
        )
    )
    def test_aggregation_handles_mixed_success_failure(self, random_contents):
        """
        Property: Result aggregation handles mixed success and failure cases.

        When some UDL files process successfully and others fail, all results
        (including error reports) should be included in the aggregated summary.
        """
        udls = []

        # Create a mix of valid and potentially invalid UDLs
        for i, content in enumerate(random_contents):
            if i % 3 == 0:
                # Create a valid UDL
                valid_content = f"rule{i} ::= '{content[:10]}' | 'valid'"
            else:
                # Create potentially problematic content
                valid_content = f"# Random content {i}\n{content}\nrule ::= 'test'"

            try:
                udl = self.create_test_udl_from_content(valid_content)
                udls.append(udl)
            except Exception:
                # Skip UDLs that can't be created
                continue

        # Skip if we don't have enough UDLs
        assume(len(udls) >= 3)

        # Process all UDLs
        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])
        reports = pipeline.compute_batch_ratings(udls)

        # Verify we get a report for each UDL (success or failure)
        assert len(reports) == len(udls), (
            f"Expected {len(udls)} reports, got {len(reports)}"
        )

        # Verify each report is valid (even error reports)
        successful_reports = 0
        error_reports = 0

        for report in reports:
            assert isinstance(report, QualityReport)
            assert hasattr(report, "udl_file")
            assert hasattr(report, "overall_score")
            assert hasattr(report, "errors")

            if len(report.errors) == 0:
                successful_reports += 1
                # Successful reports should have valid scores
                assert 0.0 <= report.overall_score <= 1.0
                assert 0.0 <= report.confidence <= 1.0
            else:
                error_reports += 1
                # Error reports should still have the file information
                assert report.udl_file is not None

        # We should have at least some reports (successful or error)
        assert successful_reports + error_reports == len(udls)

    def test_aggregation_preserves_individual_results(self):
        """
        Property: Aggregation preserves individual metric results.

        When aggregating results from multiple UDLs, the individual metric
        scores for each UDL should be preserved in the final reports.
        """
        # Create test UDLs with different characteristics
        udl_contents = [
            "simple ::= 'a'",  # Very simple
            "expr ::= term '+' expr | term\nterm ::= 'x'",  # Medium complexity
            "complex ::= rule1 rule2\nrule1 ::= 'a' | 'b'\nrule2 ::= 'c' | 'd'",  # More complex
        ]

        udls = []
        for content in udl_contents:
            udl = self.create_test_udl_from_content(content)
            udls.append(udl)

        # Process individually to get baseline results
        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])
        individual_reports = []
        for udl in udls:
            report = pipeline.compute_rating(udl)
            individual_reports.append(report)

        # Process as batch
        batch_reports = pipeline.compute_batch_ratings(udls)

        # Verify batch results match individual results
        assert len(batch_reports) == len(individual_reports)

        for i, (individual, batch) in enumerate(zip(individual_reports, batch_reports)):
            # File paths should match
            assert individual.udl_file == batch.udl_file, (
                f"File path mismatch for UDL {i}"
            )

            # Metric scores should be identical
            for metric_name in ["consistency", "completeness"]:
                if (
                    metric_name in individual.metric_scores
                    and metric_name in batch.metric_scores
                ):
                    individual_score = individual.metric_scores[metric_name]
                    batch_score = batch.metric_scores[metric_name]

                    assert abs(individual_score - batch_score) < 1e-10, (
                        f"Metric {metric_name} differs for UDL {i}: individual={individual_score}, batch={batch_score}"
                    )

            # Overall scores should be identical
            assert abs(individual.overall_score - batch.overall_score) < 1e-10, (
                f"Overall score differs for UDL {i}: individual={individual.overall_score}, batch={batch.overall_score}"
            )

    def test_empty_input_aggregation(self):
        """
        Property: Aggregation handles empty input gracefully.

        When given an empty list of UDLs, the system should return an empty
        list of reports without errors.
        """
        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])

        # Test with empty list
        reports = pipeline.compute_batch_ratings([])

        assert isinstance(reports, list)
        assert len(reports) == 0


class TestRatingPipelineUnits:
    """
    Unit tests for rating pipeline functionality.

    Tests complete pipeline on sample UDL, error handling when metric fails,
    and report generation.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Ensure we have metrics registered for testing
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric

        # Clear registry and register test metrics
        MetricRegistry.clear()
        MetricRegistry.register("consistency", ConsistencyMetric)
        MetricRegistry.register("completeness", CompletenessMetric)

    def teardown_method(self):
        """Clean up after tests."""
        MetricRegistry.clear()

    def create_sample_udl(self) -> UDLRepresentation:
        """Create a sample UDL for testing."""
        content = """
        # Sample arithmetic expression grammar
        expr ::= term '+' expr | term '-' expr | term
        term ::= factor '*' term | factor '/' term | factor
        factor ::= '(' expr ')' | number | identifier
        number ::= digit+
        digit ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
        identifier ::= letter (letter | digit)*
        letter ::= 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h' | 'i' | 'j'
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".udl", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            udl = UDLRepresentation(content, temp_path)
            return udl
        finally:
            os.unlink(temp_path)

    def test_complete_pipeline_on_sample_udl(self):
        """
        Test complete pipeline on sample UDL.

        Validates: Requirements 5.1, 5.3, 5.5
        """
        # Create sample UDL
        udl = self.create_sample_udl()

        # Initialize pipeline with multiple metrics
        pipeline = RatingPipeline(
            metric_names=["consistency", "completeness"],
            weights={"consistency": 0.7, "completeness": 0.3},
            enable_tracing=True,
        )

        # Compute rating
        report = pipeline.compute_rating(udl)

        # Verify report structure
        assert isinstance(report, QualityReport)
        assert hasattr(report, "overall_score")
        assert hasattr(report, "confidence")
        assert hasattr(report, "metric_scores")
        assert hasattr(report, "metric_formulas")
        assert hasattr(report, "computation_trace")
        assert hasattr(report, "error_bounds")
        assert hasattr(report, "timestamp")
        assert hasattr(report, "udl_file")

        # Verify score bounds
        assert 0.0 <= report.overall_score <= 1.0
        assert 0.0 <= report.confidence <= 1.0

        # Verify individual metrics are computed
        assert "consistency" in report.metric_scores
        assert "completeness" in report.metric_scores
        assert 0.0 <= report.metric_scores["consistency"] <= 1.0
        assert 0.0 <= report.metric_scores["completeness"] <= 1.0

        # Verify formulas are included
        assert "consistency" in report.metric_formulas
        assert "completeness" in report.metric_formulas
        assert len(report.metric_formulas["consistency"]) > 0
        assert len(report.metric_formulas["completeness"]) > 0

        # Verify computation trace is generated
        assert len(report.computation_trace) > 0
        for step in report.computation_trace:
            assert isinstance(step, ComputationStep)
            assert hasattr(step, "step_number")
            assert hasattr(step, "operation")
            assert hasattr(step, "formula")
            assert hasattr(step, "inputs")
            assert hasattr(step, "output")

        # Verify error bounds are computed
        assert len(report.error_bounds) > 0
        for metric_name, (lower, upper) in report.error_bounds.items():
            assert 0.0 <= lower <= upper <= 1.0

        # Verify file path is correct
        assert report.udl_file == udl.file_path

        # Verify timestamp is recent
        import datetime

        time_diff = datetime.datetime.now() - report.timestamp
        assert time_diff.total_seconds() < 60  # Should be within last minute

    def test_error_handling_when_metric_fails(self):
        """
        Test error handling when metric computation fails.

        Validates: Requirements 5.1, 5.3
        """
        # Create a problematic UDL that might cause metric failures
        problematic_content = """
        # Intentionally problematic grammar
        rule1 ::= rule1  # Direct left recursion
        rule2 ::= rule3
        rule3 ::= rule2  # Indirect recursion
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".udl", delete=False) as f:
            f.write(problematic_content)
            temp_path = f.name

        try:
            udl = UDLRepresentation(problematic_content, temp_path)

            # Initialize pipeline
            pipeline = RatingPipeline(
                metric_names=["consistency", "completeness"], enable_tracing=True
            )

            # Compute rating (should handle errors gracefully)
            report = pipeline.compute_rating(udl)

            # Verify report is still generated
            assert isinstance(report, QualityReport)
            assert hasattr(report, "errors")
            assert hasattr(report, "warnings")

            # Verify error handling
            # The pipeline should either:
            # 1. Successfully compute metrics (robust implementation)
            # 2. Handle errors gracefully and report them

            # Check that we get valid bounds even with errors
            assert 0.0 <= report.overall_score <= 1.0
            assert 0.0 <= report.confidence <= 1.0

            # If there are errors, they should be logged
            if len(report.errors) > 0:
                # Verify error messages are informative
                for error in report.errors:
                    assert isinstance(error, str)
                    assert len(error) > 0

            # Verify the system continues to function
            assert report.udl_file == udl.file_path

        finally:
            os.unlink(temp_path)

    def test_report_generation_completeness(self):
        """
        Test that report generation includes all required fields.

        Validates: Requirements 5.5
        """
        udl = self.create_sample_udl()

        # Test with different pipeline configurations
        configurations = [
            {"metric_names": ["consistency"], "enable_tracing": True},
            {"metric_names": ["completeness"], "enable_tracing": False},
            {"metric_names": ["consistency", "completeness"],
                "enable_tracing": True},
        ]

        for config in configurations:
            pipeline = RatingPipeline(**config)
            report = pipeline.compute_rating(udl)

            # Verify all required fields are present
            required_fields = [
                "overall_score",
                "confidence",
                "metric_scores",
                "metric_formulas",
                "computation_trace",
                "error_bounds",
                "timestamp",
                "udl_file",
                "errors",
                "warnings",
            ]

            for field in required_fields:
                assert hasattr(
                    report, field), f"Report missing required field: {field}"

            # Verify field types
            assert isinstance(report.overall_score, (int, float))
            assert isinstance(report.confidence, (int, float))
            assert isinstance(report.metric_scores, dict)
            assert isinstance(report.metric_formulas, dict)
            assert isinstance(report.computation_trace, list)
            assert isinstance(report.error_bounds, dict)
            assert isinstance(report.udl_file, str)
            assert isinstance(report.errors, list)
            assert isinstance(report.warnings, list)

            # Verify metric-specific content
            for metric_name in config["metric_names"]:
                if metric_name in report.metric_scores:
                    assert metric_name in report.metric_formulas
                    assert isinstance(
                        report.metric_scores[metric_name], (int, float))
                    assert isinstance(report.metric_formulas[metric_name], str)

            # Verify tracing behavior
            if config["enable_tracing"]:
                assert len(report.computation_trace) > 0
            else:
                assert len(report.computation_trace) == 0

    def test_pipeline_with_custom_weights(self):
        """
        Test pipeline with custom metric weights.

        Validates: Requirements 5.3
        """
        udl = self.create_sample_udl()

        # Test different weight configurations
        weight_configs = [
            {"consistency": 1.0, "completeness": 0.0},
            {"consistency": 0.0, "completeness": 1.0},
            {"consistency": 0.3, "completeness": 0.7},
            {"consistency": 0.8, "completeness": 0.2},
        ]

        for weights in weight_configs:
            pipeline = RatingPipeline(
                metric_names=list(weights.keys()), weights=weights
            )

            report = pipeline.compute_rating(udl)

            # Verify the aggregation uses the specified weights
            if all(metric in report.metric_scores for metric in weights.keys()):
                expected_score = sum(
                    weights[metric] * report.metric_scores[metric]
                    for metric in weights.keys()
                )

                # Allow small numerical differences
                assert abs(report.overall_score - expected_score) < 1e-6, (
                    f"Aggregation incorrect: expected {expected_score}, got {report.overall_score}"
                )

    def test_pipeline_validation_functionality(self):
        """
        Test pipeline validation functionality.

        Validates: Requirements 5.1
        """
        udl = self.create_sample_udl()
        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])

        # Run validation
        validation_results = pipeline.validate_pipeline(udl)

        # Verify validation results structure
        assert isinstance(validation_results, dict)

        # Check for expected validation keys
        expected_keys = [
            "consistency_computable",
            "consistency_bounded",
            "completeness_computable",
            "completeness_bounded",
            "aggregation_works",
            "aggregation_bounded",
            "confidence_works",
            "confidence_bounded",
            "pipeline_works",
            "report_complete",
        ]

        for key in expected_keys:
            assert key in validation_results, f"Missing validation key: {key}"
            # Check that the value is boolean-like (can be True or False)
            value = validation_results[key]
            assert value in [True, False] or bool(value) in [
                True,
                False,
            ], f"Validation key {key} should be boolean-like"

        # For a working pipeline, most validations should pass
        working_validations = [
            "aggregation_works",
            "aggregation_bounded",
            "confidence_works",
            "confidence_bounded",
            "pipeline_works",
            "report_complete",
        ]

        for key in working_validations:
            assert validation_results[key], f"Basic validation failed: {key}"

    def test_batch_processing_consistency(self):
        """
        Test that batch processing produces consistent results.

        Validates: Requirements 5.1, 5.3
        """
        # Create multiple UDLs
        udl_contents = [
            "simple ::= 'test'",
            "expr ::= term | term '+' expr",
            "complex ::= rule1 rule2\nrule1 ::= 'a'\nrule2 ::= 'b'",
        ]

        udls = []
        for content in udl_contents:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".udl", delete=False
            ) as f:
                f.write(content)
                temp_path = f.name

            try:
                udl = UDLRepresentation(content, temp_path)
                udls.append(udl)
            finally:
                os.unlink(temp_path)

        pipeline = RatingPipeline(metric_names=["consistency", "completeness"])

        # Process individually
        individual_reports = []
        for udl in udls:
            report = pipeline.compute_rating(udl)
            individual_reports.append(report)

        # Process as batch
        batch_reports = pipeline.compute_batch_ratings(udls)

        # Verify consistency
        assert len(individual_reports) == len(batch_reports)

        for individual, batch in zip(individual_reports, batch_reports):
            # Scores should be identical (within numerical precision)
            assert abs(individual.overall_score - batch.overall_score) < 1e-10
            assert abs(individual.confidence - batch.confidence) < 1e-10

            # Metric scores should match
            for metric_name in individual.metric_scores:
                if metric_name in batch.metric_scores:
                    individual_score = individual.metric_scores[metric_name]
                    batch_score = batch.metric_scores[metric_name]
                    assert abs(individual_score - batch_score) < 1e-10
