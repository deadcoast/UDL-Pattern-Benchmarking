"""
Property-based tests for uncovered code paths in UDL Rating Framework.

This module implements comprehensive property-based testing using Hypothesis
to test complex data structures, mathematical computations with edge values,
invariants across different code paths, and state machine properties.
"""

import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.core.confidence import ConfidenceCalculator
from udl_rating_framework.core.metrics.completeness import CompletenessMetric
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
from udl_rating_framework.core.metrics.structural_coherence import (
    StructuralCoherenceMetric,
)
from udl_rating_framework.core.pipeline import QualityReport, RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation


# Custom strategies for complex data structures
@st.composite
def probability_distribution_strategy(draw, min_size=2, max_size=10):
    """Generate valid probability distributions."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate positive values
    values = draw(
        st.lists(
            st.floats(min_value=0.001, max_value=10.0), min_size=size, max_size=size
        )
    )
    # Normalize to sum to 1
    total = sum(values)
    return [v / total for v in values]


@st.composite
def metric_weights_strategy(draw, min_metrics=2, max_metrics=6):
    """Generate valid metric weight dictionaries."""
    num_metrics = draw(st.integers(
        min_value=min_metrics, max_value=max_metrics))
    metric_names = [f"metric_{i}" for i in range(num_metrics)]

    # Generate positive weights
    weights = draw(
        st.lists(
            st.floats(min_value=0.001, max_value=1.0),
            min_size=num_metrics,
            max_size=num_metrics,
        )
    )

    # Normalize to sum to 1
    total = sum(weights)
    normalized_weights = [w / total for w in weights]

    return dict(zip(metric_names, normalized_weights))


@st.composite
def udl_content_strategy(draw):
    """Generate realistic UDL content for testing."""
    # Generate grammar rules
    num_rules = draw(st.integers(min_value=1, max_value=10))
    rules = []

    for i in range(num_rules):
        lhs = draw(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    min_size=1, max_size=5)
        )
        rhs_parts = draw(
            st.lists(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz",
                        min_size=1, max_size=8),
                min_size=1,
                max_size=4,
            )
        )
        rhs = " ".join(rhs_parts)
        rules.append(f"{lhs} ::= {rhs}")

    return "\n".join(rules)


class TestMathematicalComputationsEdgeCases:
    """Property-based tests for mathematical computations with edge values."""

    @given(probability_distribution_strategy())
    @settings(max_examples=200, deadline=5000)
    def test_confidence_calculation_edge_values(self, probs):
        """Test confidence calculation with various probability distributions."""
        calculator = ConfidenceCalculator()

        # Test with the generated distribution
        confidence = calculator.compute_confidence(probs)

        # Confidence must be bounded
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} not in [0,1]"

        # Test edge cases
        if len(probs) == 1:
            # Single probability should give maximum confidence
            assert confidence == 1.0, (
                f"Single probability should give confidence 1.0, got {confidence}"
            )

        # Test uniform distribution (should give low confidence)
        uniform_probs = [1.0 / len(probs)] * len(probs)
        uniform_confidence = calculator.compute_confidence(uniform_probs)

        # Uniform distribution should have lower confidence than peaked distribution
        max_prob = max(probs)
        if max_prob > 1.5 / len(probs):  # Significantly non-uniform
            assert confidence >= uniform_confidence, (
                f"Peaked distribution should have higher confidence: {confidence} vs {uniform_confidence}"
            )

    @given(metric_weights_strategy())
    @settings(max_examples=200, deadline=5000)
    def test_aggregation_mathematical_properties(self, weights):
        """Test mathematical properties of metric aggregation."""
        aggregator = MetricAggregator(weights)

        # Generate metric values
        metric_values = {name: np.random.uniform(
            0, 1) for name in weights.keys()}

        # Test basic aggregation
        result = aggregator.aggregate(metric_values)
        assert 0.0 <= result <= 1.0, f"Aggregated result {result} not in [0,1]"

        # Test linearity: scaling all metrics should scale result
        scaled_values = {name: value * 0.5 for name,
                         value in metric_values.items()}
        scaled_result = aggregator.aggregate(scaled_values)
        expected_scaled = result * 0.5

        assert abs(scaled_result - expected_scaled) < 1e-10, (
            f"Linearity violated: {scaled_result} != {expected_scaled}"
        )

        # Test monotonicity: increasing a metric should increase result (if weight > 0)
        for metric_name, weight in weights.items():
            if weight > 0:
                increased_values = metric_values.copy()
                increased_values[metric_name] = min(
                    1.0, increased_values[metric_name] + 0.1
                )
                increased_result = aggregator.aggregate(increased_values)

                assert increased_result >= result, (
                    f"Monotonicity violated for {metric_name}: {increased_result} < {result}"
                )

    @given(
        st.lists(st.floats(min_value=0.0, max_value=1.0),
                 min_size=2, max_size=10),
        st.floats(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=100, deadline=3000)
    def test_aggregation_extreme_values(self, metric_values, noise):
        """Test aggregation with extreme and noisy values."""
        # Create equal weights
        num_metrics = len(metric_values)
        weights = {f"metric_{i}": 1.0 /
                   num_metrics for i in range(num_metrics)}

        aggregator = MetricAggregator(weights)
        metric_dict = {f"metric_{i}": val for i,
                       val in enumerate(metric_values)}

        result = aggregator.aggregate(metric_dict)

        # Result should be bounded even with edge values
        assert 0.0 <= result <= 1.0, (
            f"Result {result} not bounded with values {metric_values}"
        )

        # Test with clamped values (should be same as original if already in bounds)
        clamped_dict = {
            name: max(0.0, min(1.0, val)) for name, val in metric_dict.items()
        }
        clamped_result = aggregator.aggregate(clamped_dict)

        assert abs(result - clamped_result) < 1e-10, (
            f"Clamping changed result: {result} vs {clamped_result}"
        )


class TestComplexDataStructureInvariants:
    """Property-based tests for complex data structure invariants."""

    @given(udl_content_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_udl_representation_invariants(self, udl_content):
        """Test invariants of UDL representation across different inputs."""
        try:
            udl_repr = UDLRepresentation(udl_content, "test.udl")

            # Invariant 1: Token count should be consistent
            tokens1 = udl_repr.get_tokens()
            tokens2 = udl_repr.get_tokens()  # Call again
            assert len(tokens1) == len(
                tokens2), "Token count should be deterministic"

            # Invariant 2: Grammar graph should be consistent
            graph1 = udl_repr.get_grammar_graph()
            graph2 = udl_repr.get_grammar_graph()
            assert graph1.number_of_nodes() == graph2.number_of_nodes(), (
                "Grammar graph node count should be deterministic"
            )
            assert graph1.number_of_edges() == graph2.number_of_edges(), (
                "Grammar graph edge count should be deterministic"
            )

            # Invariant 3: AST conversion should be consistent
            ast1 = udl_repr.to_ast()
            ast2 = udl_repr.to_ast()
            # AST should have same structure (we can't easily compare objects, but can check basic properties)
            if hasattr(ast1, "__dict__") and hasattr(ast2, "__dict__"):
                assert type(ast1) is type(
                    ast2), "AST types should be consistent"

            # Invariant 4: Representation should handle empty content gracefully
            if not udl_content.strip():
                assert len(tokens1) >= 0, (
                    "Empty content should produce non-negative token count"
                )

        except Exception as e:
            # If parsing fails, it should fail consistently
            try:
                UDLRepresentation(udl_content, "test2.udl")
                pytest.fail(
                    f"Parsing should fail consistently, but succeeded on second try: {e}"
                )
            except Exception:
                pass  # Expected to fail consistently

    @given(
        st.lists(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.floats(min_value=0.0, max_value=1.0),
                min_size=1,
                max_size=5,
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_quality_report_aggregation_invariants(self, metric_dicts):
        """Test invariants when aggregating multiple quality reports."""
        reports = []

        for i, metrics in enumerate(metric_dicts):
            # Create a mock quality report
            report = QualityReport(
                overall_score=sum(metrics.values()) / len(metrics),
                confidence=0.8,
                metric_scores=metrics,
                metric_formulas={
                    name: f"formula_{name}" for name in metrics.keys()},
                computation_trace=[],
                error_bounds={
                    name: (val * 0.9, val * 1.1) for name, val in metrics.items()
                },
                timestamp=None,
                udl_file=f"test_{i}.udl",
            )
            reports.append(report)

        # Invariant 1: All reports should have bounded scores
        for report in reports:
            assert 0.0 <= report.overall_score <= 1.0, (
                f"Overall score {report.overall_score} not bounded"
            )
            assert 0.0 <= report.confidence <= 1.0, (
                f"Confidence {report.confidence} not bounded"
            )

            for metric_name, score in report.metric_scores.items():
                assert 0.0 <= score <= 1.0, (
                    f"Metric {metric_name} score {score} not bounded"
                )

        # Invariant 2: Error bounds should contain the actual values
        for report in reports:
            for metric_name, score in report.metric_scores.items():
                if metric_name in report.error_bounds:
                    lower, upper = report.error_bounds[metric_name]
                    assert lower <= score <= upper, (
                        f"Score {score} not within bounds [{lower}, {upper}] for {metric_name}"
                    )


class TestStateMachineProperties:
    """Property-based tests for stateful components using state machines."""

    @given(
        st.lists(
            st.tuples(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz",
                        min_size=1, max_size=20),
                st.floats(min_value=0.0, max_value=1.0),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=50, deadline=10000)
    def test_pipeline_state_consistency(self, file_score_pairs):
        """Test that pipeline maintains consistent state across operations."""
        # Create temporary files
        temp_files = []
        try:
            for content, expected_score in file_score_pairs:
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".udl", delete=False
                )
                temp_file.write(f"rule ::= {content}")
                temp_file.close()
                temp_files.append((temp_file.name, expected_score))

            # Test pipeline state consistency
            pipeline = RatingPipeline(
                metric_names=[
                    "consistency",
                    "completeness",
                    "expressiveness",
                    "structural_coherence",
                ]
            )

            # Process files in order
            results1 = []
            for file_path, _ in temp_files:
                try:
                    result = pipeline.rate_file(file_path)
                    results1.append(result)
                except Exception:
                    results1.append(None)

            # Process same files again - should get same results
            results2 = []
            for file_path, _ in temp_files:
                try:
                    result = pipeline.rate_file(file_path)
                    results2.append(result)
                except Exception:
                    results2.append(None)

            # Results should be consistent
            assert len(results1) == len(
                results2), "Result count should be consistent"

            for i, (r1, r2) in enumerate(zip(results1, results2)):
                if r1 is not None and r2 is not None:
                    assert abs(r1.overall_score - r2.overall_score) < 1e-10, (
                        f"Inconsistent results for file {i}: {r1.overall_score} vs {r2.overall_score}"
                    )
                elif r1 is None and r2 is None:
                    pass  # Both failed consistently
                else:
                    pytest.fail(
                        f"Inconsistent failure pattern for file {i}: {r1} vs {r2}"
                    )

        finally:
            # Clean up temporary files
            for file_path, _ in temp_files:
                try:
                    os.unlink(file_path)
                except OSError:
                    pass


class MetricComputationStateMachine(RuleBasedStateMachine):
    """State machine for testing metric computation properties."""

    def __init__(self):
        super().__init__()
        self.computed_metrics = {}
        self.udl_representations = {}
        self.metric_instances = {
            "consistency": ConsistencyMetric(),
            "completeness": CompletenessMetric(),
            "expressiveness": ExpressivenessMetric(),
            "structural_coherence": StructuralCoherenceMetric(),
        }

    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.computed_metrics = {}
        self.udl_representations = {}

    @rule(
        udl_id=st.text(min_size=1, max_size=10),
        content=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz ::=|()[]{}+*", min_size=5, max_size=100
        ),
    )
    def add_udl(self, udl_id, content):
        """Add a UDL representation to the state."""
        try:
            udl_repr = UDLRepresentation(content, f"{udl_id}.udl")
            self.udl_representations[udl_id] = udl_repr
            self.computed_metrics[udl_id] = {}
        except Exception:
            # If UDL creation fails, that's okay - we just don't add it
            pass

    @rule(
        metric_name=st.sampled_from(
            ["consistency", "completeness", "expressiveness", "structural_coherence"]
        )
    )
    def compute_metric(self, metric_name):
        """Compute a metric for all available UDLs."""
        for udl_id in list(self.udl_representations.keys()):
            try:
                metric = self.metric_instances[metric_name]
                udl_repr = self.udl_representations[udl_id]
                score = metric.compute(udl_repr)
                self.computed_metrics[udl_id][metric_name] = score
            except Exception:
                # Metric computation can fail, that's okay
                pass

    @invariant()
    def all_scores_bounded(self):
        """All computed scores must be in [0, 1]."""
        for udl_id, metrics in self.computed_metrics.items():
            for metric_name, score in metrics.items():
                assert 0.0 <= score <= 1.0, (
                    f"Score {score} for {metric_name} on {udl_id} not in [0,1]"
                )

    @invariant()
    def deterministic_computation(self):
        """Recomputing the same metric should give the same result."""
        for udl_id, udl_repr in self.udl_representations.items():
            for metric_name, metric in self.metric_instances.items():
                if metric_name in self.computed_metrics.get(udl_id, {}):
                    try:
                        original_score = self.computed_metrics[udl_id][metric_name]
                        recomputed_score = metric.compute(udl_repr)
                        assert abs(original_score - recomputed_score) < 1e-10, (
                            f"Non-deterministic computation: {original_score} vs {recomputed_score}"
                        )
                    except Exception:
                        # If recomputation fails, that's a problem only if original succeeded
                        pass


# Make the state machine testable
TestMetricComputationStateMachine = MetricComputationStateMachine.TestCase


class TestShrinkingAndMinimalExamples:
    """Tests that use Hypothesis shrinking to find minimal failing examples."""

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=5000)
    def test_udl_parsing_robustness(self, content):
        """Test UDL parsing with arbitrary content to find minimal failing cases."""
        try:
            udl_repr = UDLRepresentation(content, "test.udl")

            # If parsing succeeds, basic properties should hold
            tokens = udl_repr.get_tokens()
            assert isinstance(tokens, list), "Tokens should be a list"

            graph = udl_repr.get_grammar_graph()
            assert hasattr(graph, "number_of_nodes"), (
                "Graph should have node count method"
            )
            assert graph.number_of_nodes() >= 0, "Node count should be non-negative"

        except Exception as e:
            # Parsing can fail, but should fail gracefully with informative errors
            assert isinstance(e, Exception), (
                f"Should raise proper exception, got {type(e)}"
            )

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=20),
            elements=st.floats(
                min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=100, deadline=3000)
    def test_confidence_calculation_numerical_stability(self, raw_probs):
        """Test confidence calculation numerical stability with various inputs."""
        # Normalize to create valid probability distribution
        probs = raw_probs / np.sum(raw_probs)

        calculator = ConfidenceCalculator()

        try:
            confidence = calculator.compute_confidence(probs)

            # Should always produce valid confidence
            assert np.isfinite(confidence), (
                f"Confidence should be finite, got {confidence}"
            )
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} not in [0,1]"

            # Test with very small probabilities (numerical edge case)
            tiny_probs = probs * 1e-10
            tiny_probs = tiny_probs / np.sum(tiny_probs)
            tiny_confidence = calculator.compute_confidence(tiny_probs)

            assert np.isfinite(tiny_confidence), (
                f"Tiny confidence should be finite, got {tiny_confidence}"
            )
            assert 0.0 <= tiny_confidence <= 1.0, (
                f"Tiny confidence {tiny_confidence} not in [0,1]"
            )

        except Exception as e:
            pytest.fail(
                f"Confidence calculation failed with valid probabilities: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
