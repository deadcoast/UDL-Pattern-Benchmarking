"""
Property-based tests for metric aggregation and confidence calculation.

Tests the mathematical properties of:
- MetricAggregator: Q = Σ(wᵢ · mᵢ) with proper boundedness
- ConfidenceCalculator: C = 1 - H(p)/H_max with entropy-based computation
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List
from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.core.confidence import ConfidenceCalculator


class TestAggregationCorrectness:
    """Property-based tests for aggregation correctness."""

    @given(
        st.dictionaries(
            keys=st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10
            ),
            values=st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=10,
        ).filter(lambda d: len(d) > 0)
    )
    @settings(max_examples=100, deadline=None)
    def test_aggregation_correctness_property(self, metric_values: Dict[str, float]):
        """
        **Feature: udl-rating-framework, Property 4: Aggregation Correctness**
        **Validates: Requirements 1.5, 3.8**

        For any set of metric values {m₁, m₂, ..., mₙ} and weights {w₁, w₂, ..., wₙ}
        where Σwᵢ = 1, the aggregated score Q must equal Σ(wᵢ · mᵢ) and satisfy 0 ≤ Q ≤ 1.
        """
        # Generate valid weights that sum to 1
        metric_names = list(metric_values.keys())
        n_metrics = len(metric_names)

        # Generate random weights and normalize them to sum to 1
        raw_weights = np.random.uniform(0.1, 1.0, n_metrics)  # Avoid zero weights
        normalized_weights = raw_weights / np.sum(raw_weights)

        weights = {
            name: float(weight)
            for name, weight in zip(metric_names, normalized_weights)
        }

        # Verify weights sum to 1 (within numerical precision)
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 1e-10, (
            f"Weights should sum to 1.0, got {weight_sum}"
        )

        # Create aggregator
        aggregator = MetricAggregator(weights)

        # Compute aggregated score
        result = aggregator.aggregate(metric_values)

        # Verify the formula: Q = Σ(wᵢ · mᵢ)
        expected = sum(weights[name] * value for name, value in metric_values.items())
        assert abs(result - expected) < 1e-10, (
            f"Aggregation formula incorrect: expected {expected}, got {result}"
        )

        # Verify boundedness: 0 ≤ Q ≤ 1
        assert 0.0 <= result <= 1.0, (
            f"Aggregated score {result} is outside [0,1] bounds"
        )

        # Additional property: if all metrics are in [0,1] and weights sum to 1,
        # then result must be in [0,1] (which we already tested above)

        # Test monotonicity: if we increase a metric value, result should increase
        # (assuming positive weight)
        for name in metric_names:
            if weights[name] > 0 and metric_values[name] < 1.0:
                # Increase this metric value slightly
                modified_values = metric_values.copy()
                increase = min(0.1, 1.0 - metric_values[name])
                modified_values[name] += increase

                new_result = aggregator.aggregate(modified_values)
                expected_increase = weights[name] * increase

                assert new_result >= result, (
                    f"Increasing metric {name} should not decrease result: "
                    f"original={result}, new={new_result}"
                )

                assert abs((new_result - result) - expected_increase) < 1e-10, (
                    f"Increase in result should equal weight * increase in metric: "
                    f"expected increase={expected_increase}, actual increase={new_result - result}"
                )

    def test_aggregation_edge_cases(self):
        """Test edge cases for aggregation."""
        # Test single metric
        weights = {"metric1": 1.0}
        values = {"metric1": 0.7}
        aggregator = MetricAggregator(weights)
        result = aggregator.aggregate(values)
        assert abs(result - 0.7) < 1e-10

        # Test all zeros
        weights = {"m1": 0.5, "m2": 0.5}
        values = {"m1": 0.0, "m2": 0.0}
        aggregator = MetricAggregator(weights)
        result = aggregator.aggregate(values)
        assert abs(result - 0.0) < 1e-10

        # Test all ones
        values = {"m1": 1.0, "m2": 1.0}
        result = aggregator.aggregate(values)
        assert abs(result - 1.0) < 1e-10

    def test_weight_validation(self):
        """Test weight validation in MetricAggregator."""
        # Test weights that don't sum to 1
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            MetricAggregator({"m1": 0.3, "m2": 0.4})  # Sum = 0.7

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            MetricAggregator({"m1": 0.6, "m2": 0.5})  # Sum = 1.1

        # Test negative weights
        with pytest.raises(ValueError, match="All weights must be non-negative"):
            MetricAggregator({"m1": 0.6, "m2": -0.1, "m3": 0.5})

        # Test valid weights
        aggregator = MetricAggregator({"m1": 0.3, "m2": 0.7})
        assert aggregator is not None


class TestConfidenceFormulaCorrectness:
    """Property-based tests for confidence formula correctness."""

    @given(
        st.lists(
            st.floats(min_value=0.001, max_value=1.0),  # Avoid exact zeros
            min_size=2,
            max_size=20,
        ).filter(lambda probs: len(probs) >= 2 and sum(probs) > 0)
    )
    @settings(max_examples=100, deadline=None)
    def test_confidence_formula_correctness_property(self, raw_probs: List[float]):
        """
        **Feature: udl-rating-framework, Property 5: Confidence Formula Correctness**
        **Validates: Requirements 1.6, 5.4**

        For any prediction distribution p, the confidence C must equal 1 - H(p)/H_max
        where H(p) = -Σ pᵢ log(pᵢ) and H_max = log(n).
        """
        # Normalize to create valid probability distribution
        prob_sum = sum(raw_probs)
        assume(prob_sum > 0)  # Ensure we can normalize

        probs = np.array([p / prob_sum for p in raw_probs])

        # Verify it's a valid probability distribution
        assert abs(np.sum(probs) - 1.0) < 1e-10, "Probabilities should sum to 1"
        assert np.all(probs >= 0), "All probabilities should be non-negative"

        calculator = ConfidenceCalculator()
        confidence = calculator.compute_confidence(probs)

        # Manually compute expected confidence using the formula
        epsilon = 1e-10
        safe_probs = probs + epsilon  # Add epsilon to avoid log(0)

        # H(p) = -Σ pᵢ log(pᵢ)
        entropy = -np.sum(probs * np.log(safe_probs))

        # H_max = log(n)
        max_entropy = np.log(len(probs))

        # C = 1 - H(p)/H_max
        if max_entropy == 0:  # Single element case
            expected_confidence = 1.0
        else:
            expected_confidence = 1.0 - (entropy / max_entropy)

        # Ensure expected is in bounds
        expected_confidence = np.clip(expected_confidence, 0.0, 1.0)

        # Compare with computed confidence
        assert abs(confidence - expected_confidence) < 1e-6, (
            f"Confidence formula incorrect: expected {expected_confidence}, got {confidence}"
        )

        # Verify boundedness: 0 ≤ C ≤ 1
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence {confidence} is outside [0,1] bounds"
        )

    def test_confidence_extreme_cases(self):
        """Test confidence calculation for extreme probability distributions."""
        calculator = ConfidenceCalculator()

        # Test uniform distribution (should have low confidence)
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        confidence = calculator.compute_confidence(uniform_probs)

        # For uniform distribution, entropy is maximized, so confidence should be 0
        assert abs(confidence - 0.0) < 1e-6, (
            f"Uniform distribution should have confidence ≈ 0, got {confidence}"
        )

        # Test delta distribution (should have high confidence)
        delta_probs = np.array([1.0, 0.0, 0.0, 0.0])
        confidence = calculator.compute_confidence(delta_probs)

        # For delta distribution, entropy is minimized, so confidence should be 1
        assert confidence > 0.99, (
            f"Delta distribution should have confidence ≈ 1, got {confidence}"
        )

        # Test binary distribution with different levels of certainty
        # High certainty: [0.9, 0.1]
        high_cert_probs = np.array([0.9, 0.1])
        high_confidence = calculator.compute_confidence(high_cert_probs)

        # Low certainty: [0.6, 0.4]
        low_cert_probs = np.array([0.6, 0.4])
        low_confidence = calculator.compute_confidence(low_cert_probs)

        # Higher certainty should lead to higher confidence
        assert high_confidence > low_confidence, (
            f"Higher certainty should lead to higher confidence: "
            f"high={high_confidence}, low={low_confidence}"
        )

    def test_confidence_monotonicity(self):
        """Test that confidence increases as distribution becomes more peaked."""
        calculator = ConfidenceCalculator()

        # Start with uniform distribution and gradually make it more peaked
        base_prob = 0.25
        confidences = []

        for peak_increase in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            # Create distribution where first element gets more probability
            probs = np.array(
                [
                    base_prob + peak_increase,
                    base_prob - peak_increase / 3,
                    base_prob - peak_increase / 3,
                    base_prob - peak_increase / 3,
                ]
            )

            # Ensure valid probabilities
            probs = np.clip(probs, 0.001, 1.0)
            probs = probs / np.sum(probs)  # Renormalize

            confidence = calculator.compute_confidence(probs)
            confidences.append(confidence)

        # Confidence should generally increase as distribution becomes more peaked
        # (allowing for some numerical noise)
        for i in range(1, len(confidences)):
            assert confidences[i] >= confidences[i - 1] - 1e-6, (
                f"Confidence should increase with peakedness: "
                f"step {i - 1}: {confidences[i - 1]}, step {i}: {confidences[i]}"
            )

    def test_confidence_input_validation(self):
        """Test confidence calculator with various input types."""
        calculator = ConfidenceCalculator()

        # Test with list input
        list_probs = [0.7, 0.3]
        confidence = calculator.compute_confidence(list_probs)
        assert 0.0 <= confidence <= 1.0

        # Test with numpy array input
        array_probs = np.array([0.7, 0.3])
        confidence2 = calculator.compute_confidence(array_probs)
        assert abs(confidence - confidence2) < 1e-10

        # Test with single element (edge case)
        single_prob = np.array([1.0])
        confidence = calculator.compute_confidence(single_prob)
        assert confidence == 1.0  # Single element should have maximum confidence

        # Test with very small probabilities (near zero)
        small_probs = np.array([0.999999, 0.000001])
        confidence = calculator.compute_confidence(small_probs)
        assert 0.0 <= confidence <= 1.0


class TestAggregationConfidenceIntegration:
    """Integration tests for aggregation and confidence working together."""

    def test_aggregation_confidence_workflow(self):
        """Test typical workflow of aggregation followed by confidence calculation."""
        # Simulate metric computation results
        metric_values = {
            "consistency": 0.8,
            "completeness": 0.6,
            "expressiveness": 0.9,
            "structural_coherence": 0.7,
        }

        # Define weights
        weights = {
            "consistency": 0.3,
            "completeness": 0.2,
            "expressiveness": 0.3,
            "structural_coherence": 0.2,
        }

        # Aggregate metrics
        aggregator = MetricAggregator(weights)
        overall_quality = aggregator.aggregate(metric_values)

        # Verify aggregation
        expected = 0.3 * 0.8 + 0.2 * 0.6 + 0.3 * 0.9 + 0.2 * 0.7
        assert abs(overall_quality - expected) < 1e-10

        # Simulate prediction distribution for confidence
        # (in practice, this would come from model uncertainty)
        prediction_probs = np.array([overall_quality, 1.0 - overall_quality])

        # Calculate confidence
        calculator = ConfidenceCalculator()
        confidence = calculator.compute_confidence(prediction_probs)

        # Verify confidence is reasonable
        assert 0.0 <= confidence <= 1.0

        # For this binary case with unequal probabilities, confidence should be > 0
        assert confidence > 0.0

    def test_multiple_aggregation_scenarios(self):
        """Test aggregation with different weight configurations."""
        metric_values = {"m1": 0.4, "m2": 0.8}

        # Equal weights
        equal_weights = {"m1": 0.5, "m2": 0.5}
        aggregator1 = MetricAggregator(equal_weights)
        result1 = aggregator1.aggregate(metric_values)
        expected1 = 0.5 * 0.4 + 0.5 * 0.8  # = 0.6
        assert abs(result1 - expected1) < 1e-10

        # Weighted toward first metric
        weighted1 = {"m1": 0.8, "m2": 0.2}
        aggregator2 = MetricAggregator(weighted1)
        result2 = aggregator2.aggregate(metric_values)
        expected2 = 0.8 * 0.4 + 0.2 * 0.8  # = 0.48
        assert abs(result2 - expected2) < 1e-10

        # Weighted toward second metric
        weighted2 = {"m1": 0.2, "m2": 0.8}
        aggregator3 = MetricAggregator(weighted2)
        result3 = aggregator3.aggregate(metric_values)
        expected3 = 0.2 * 0.4 + 0.8 * 0.8  # = 0.72
        assert abs(result3 - expected3) < 1e-10

        # Results should be ordered based on which metric is emphasized
        assert result2 < result1 < result3  # m1 < equal < m2 (since m2 > m1)
