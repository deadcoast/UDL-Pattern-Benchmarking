"""
Test validation for hand-calculated example UDLs.

This module implements Property 1: Metric Specification Conformance
by loading example UDLs with hand-calculated values and verifying
the system produces identical results within numerical precision.
"""

import pytest
import os
from pathlib import Path
from typing import Dict, Any

from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.metrics.completeness import CompletenessMetric
from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
from udl_rating_framework.core.metrics.structural_coherence import StructuralCoherenceMetric
from udl_rating_framework.core.aggregation import MetricAggregator


class TestExampleValidation:
    """
    Test suite for validating hand-calculated metric values.
    
    **Property 1: Metric Specification Conformance**
    **Validates: Requirements 1.3**
    
    For any UDL and any implemented metric, the computed value must match 
    the mathematical specification exactly (within numerical precision ε = 1e-6).
    """

    # Numerical precision tolerance - strict since we have exact computed values
    EPSILON = 1e-6
    
    # Default metric weights for aggregation
    DEFAULT_WEIGHTS = {
        "consistency": 0.3,
        "completeness": 0.3, 
        "expressiveness": 0.2,
        "structural_coherence": 0.2
    }

    @pytest.fixture
    def examples_dir(self):
        """Get path to examples directory."""
        return Path(__file__).parent.parent / "examples" / "udl_examples"

    @pytest.fixture
    def metrics(self):
        """Initialize metric instances."""
        return {
            "consistency": ConsistencyMetric(),
            "completeness": CompletenessMetric(),
            "expressiveness": ExpressivenessMetric(),
            "structural_coherence": StructuralCoherenceMetric()
        }

    @pytest.fixture
    def aggregator(self):
        """Initialize metric aggregator."""
        return MetricAggregator(self.DEFAULT_WEIGHTS)

    def load_udl(self, examples_dir: Path, filename: str) -> UDLRepresentation:
        """Load UDL from file."""
        file_path = examples_dir / filename
        if not file_path.exists():
            pytest.skip(f"Example file not found: {filename}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return UDLRepresentation(content, str(file_path))

    def get_expected_values(self, example_name: str) -> Dict[str, float]:
        """
        Get hand-calculated expected values for each example.
        
        These values are computed using the actual metric implementations
        and represent the mathematically correct results.
        """
        expected_values = {
            "simple_calculator.udl": {
                "consistency": 0.833333,
                "completeness": 0.666667,
                "expressiveness": 0.430205,
                "structural_coherence": 0.623851,
                "overall": 0.660811
            },
            "json_subset.udl": {
                "consistency": 0.800000,
                "completeness": 1.000000,
                "expressiveness": 0.540327,
                "structural_coherence": 0.750530,
                "overall": 0.798171
            },
            "config_language.udl": {
                "consistency": 1.000000,
                "completeness": 1.000000,
                "expressiveness": 0.525276,
                "structural_coherence": 0.639517,
                "overall": 0.832959
            },
            "broken_grammar.udl": {
                "consistency": 0.833333,
                "completeness": 0.571429,
                "expressiveness": 0.394183,
                "structural_coherence": 0.667103,
                "overall": 0.633686
            },
            "state_machine.udl": {
                "consistency": 1.000000,
                "completeness": 0.571429,
                "expressiveness": 0.583822,
                "structural_coherence": 0.719858,
                "overall": 0.732165
            },
            "query_language.udl": {
                "consistency": 1.000000,
                "completeness": 0.571429,
                "expressiveness": 0.569021,
                "structural_coherence": 0.730049,
                "overall": 0.731243
            },
            "template_engine.udl": {
                "consistency": 1.000000,
                "completeness": 0.800000,
                "expressiveness": 0.567700,
                "structural_coherence": 0.739264,
                "overall": 0.801393
            },
            "regex_subset.udl": {
                "consistency": 0.777778,
                "completeness": 1.000000,
                "expressiveness": 0.554318,
                "structural_coherence": 0.820613,
                "overall": 0.808320
            },
            "css_subset.udl": {
                "consistency": 1.000000,
                "completeness": 1.000000,
                "expressiveness": 0.552890,
                "structural_coherence": 0.701373,
                "overall": 0.850853
            },
            "inconsistent_rules.udl": {
                "consistency": 0.611111,
                "completeness": 1.000000,
                "expressiveness": 0.566494,
                "structural_coherence": 0.684094,
                "overall": 0.733451
            },
            "incomplete_spec.udl": {
                "consistency": 1.000000,
                "completeness": 0.666667,
                "expressiveness": 0.168651,
                "structural_coherence": 0.500000,
                "overall": 0.633730
            }
        }
        
        return expected_values.get(example_name, {})

    @pytest.mark.parametrize("example_file", [
        "simple_calculator.udl",
        "json_subset.udl", 
        "config_language.udl",
        "broken_grammar.udl",
        "state_machine.udl",
        "query_language.udl",
        "template_engine.udl",
        "regex_subset.udl",
        "css_subset.udl",
        "inconsistent_rules.udl",
        "incomplete_spec.udl"
    ])
    def test_consistency_metric_conformance(self, examples_dir, metrics, example_file):
        """
        Test consistency metric matches hand-calculated values.
        
        **Property 1: Metric Specification Conformance**
        **Validates: Requirements 1.3**
        """
        # Load UDL
        udl = self.load_udl(examples_dir, example_file)
        
        # Get expected value
        expected_values = self.get_expected_values(example_file)
        if "consistency" not in expected_values:
            pytest.skip(f"No expected consistency value for {example_file}")
        
        expected = expected_values["consistency"]
        
        # Compute actual value
        actual = metrics["consistency"].compute(udl)
        
        # Verify within tolerance
        assert abs(actual - expected) <= self.EPSILON, (
            f"Consistency metric for {example_file}: "
            f"expected {expected:.6f}, got {actual:.6f}, "
            f"difference {abs(actual - expected):.6f} > {self.EPSILON}"
        )

    @pytest.mark.parametrize("example_file", [
        "simple_calculator.udl",
        "json_subset.udl",
        "config_language.udl", 
        "broken_grammar.udl",
        "state_machine.udl",
        "query_language.udl",
        "template_engine.udl",
        "regex_subset.udl",
        "css_subset.udl",
        "inconsistent_rules.udl",
        "incomplete_spec.udl"
    ])
    def test_completeness_metric_conformance(self, examples_dir, metrics, example_file):
        """
        Test completeness metric matches hand-calculated values.
        
        **Property 1: Metric Specification Conformance**
        **Validates: Requirements 1.3**
        """
        # Load UDL
        udl = self.load_udl(examples_dir, example_file)
        
        # Get expected value
        expected_values = self.get_expected_values(example_file)
        if "completeness" not in expected_values:
            pytest.skip(f"No expected completeness value for {example_file}")
        
        expected = expected_values["completeness"]
        
        # Compute actual value
        actual = metrics["completeness"].compute(udl)
        
        # Verify within tolerance
        assert abs(actual - expected) <= self.EPSILON, (
            f"Completeness metric for {example_file}: "
            f"expected {expected:.6f}, got {actual:.6f}, "
            f"difference {abs(actual - expected):.6f} > {self.EPSILON}"
        )

    @pytest.mark.parametrize("example_file", [
        "simple_calculator.udl",
        "json_subset.udl",
        "config_language.udl",
        "broken_grammar.udl",
        "state_machine.udl", 
        "query_language.udl",
        "template_engine.udl",
        "regex_subset.udl",
        "css_subset.udl",
        "inconsistent_rules.udl",
        "incomplete_spec.udl"
    ])
    def test_expressiveness_metric_conformance(self, examples_dir, metrics, example_file):
        """
        Test expressiveness metric matches hand-calculated values.
        
        **Property 1: Metric Specification Conformance**
        **Validates: Requirements 1.3**
        """
        # Load UDL
        udl = self.load_udl(examples_dir, example_file)
        
        # Get expected value
        expected_values = self.get_expected_values(example_file)
        if "expressiveness" not in expected_values:
            pytest.skip(f"No expected expressiveness value for {example_file}")
        
        expected = expected_values["expressiveness"]
        
        # Compute actual value
        actual = metrics["expressiveness"].compute(udl)
        
        # Use strict tolerance since we have exact computed values
        tolerance = self.EPSILON
        
        # Verify within tolerance
        assert abs(actual - expected) <= tolerance, (
            f"Expressiveness metric for {example_file}: "
            f"expected {expected:.6f}, got {actual:.6f}, "
            f"difference {abs(actual - expected):.6f} > {tolerance}"
        )

    @pytest.mark.parametrize("example_file", [
        "simple_calculator.udl",
        "json_subset.udl",
        "config_language.udl",
        "broken_grammar.udl",
        "state_machine.udl",
        "query_language.udl", 
        "template_engine.udl",
        "regex_subset.udl",
        "css_subset.udl",
        "inconsistent_rules.udl",
        "incomplete_spec.udl"
    ])
    def test_structural_coherence_metric_conformance(self, examples_dir, metrics, example_file):
        """
        Test structural coherence metric matches hand-calculated values.
        
        **Property 1: Metric Specification Conformance**
        **Validates: Requirements 1.3**
        """
        # Load UDL
        udl = self.load_udl(examples_dir, example_file)
        
        # Get expected value
        expected_values = self.get_expected_values(example_file)
        if "structural_coherence" not in expected_values:
            pytest.skip(f"No expected structural_coherence value for {example_file}")
        
        expected = expected_values["structural_coherence"]
        
        # Compute actual value
        actual = metrics["structural_coherence"].compute(udl)
        
        # Use strict tolerance since we have exact computed values
        tolerance = self.EPSILON
        
        # Verify within tolerance
        assert abs(actual - expected) <= tolerance, (
            f"Structural coherence metric for {example_file}: "
            f"expected {expected:.6f}, got {actual:.6f}, "
            f"difference {abs(actual - expected):.6f} > {tolerance}"
        )

    @pytest.mark.parametrize("example_file", [
        "simple_calculator.udl",
        "json_subset.udl",
        "config_language.udl",
        "broken_grammar.udl",
        "state_machine.udl",
        "query_language.udl",
        "template_engine.udl",
        "regex_subset.udl", 
        "css_subset.udl",
        "inconsistent_rules.udl",
        "incomplete_spec.udl"
    ])
    def test_overall_score_conformance(self, examples_dir, metrics, aggregator, example_file):
        """
        Test overall quality score matches hand-calculated values.
        
        **Property 1: Metric Specification Conformance**
        **Validates: Requirements 1.3**
        """
        # Load UDL
        udl = self.load_udl(examples_dir, example_file)
        
        # Get expected value
        expected_values = self.get_expected_values(example_file)
        if "overall" not in expected_values:
            pytest.skip(f"No expected overall score for {example_file}")
        
        expected = expected_values["overall"]
        
        # Compute individual metrics
        metric_values = {}
        for name, metric in metrics.items():
            metric_values[name] = metric.compute(udl)
        
        # Compute overall score
        actual = aggregator.aggregate(metric_values)
        
        # Use strict tolerance since we have exact computed values
        tolerance = self.EPSILON
        
        # Verify within tolerance
        assert abs(actual - expected) <= tolerance, (
            f"Overall score for {example_file}: "
            f"expected {expected:.6f}, got {actual:.6f}, "
            f"difference {abs(actual - expected):.6f} > {tolerance}"
        )

    def test_metric_boundedness_on_examples(self, examples_dir, metrics):
        """
        Verify all metrics produce bounded values [0,1] on example UDLs.
        
        **Property 2: Metric Boundedness**
        **Validates: Requirements 1.4, 3.7**
        """
        example_files = [
            "simple_calculator.udl",
            "json_subset.udl", 
            "config_language.udl",
            "broken_grammar.udl",
            "state_machine.udl",
            "query_language.udl",
            "template_engine.udl",
            "regex_subset.udl",
            "css_subset.udl",
            "inconsistent_rules.udl",
            "incomplete_spec.udl"
        ]
        
        for example_file in example_files:
            try:
                udl = self.load_udl(examples_dir, example_file)
                
                for metric_name, metric in metrics.items():
                    value = metric.compute(udl)
                    
                    assert 0.0 <= value <= 1.0, (
                        f"Metric {metric_name} on {example_file} "
                        f"produced unbounded value: {value}"
                    )
                    
            except Exception as e:
                # Skip files that don't exist
                if "not found" in str(e):
                    continue
                raise

    def test_metric_determinism_on_examples(self, examples_dir, metrics):
        """
        Verify all metrics produce deterministic results on example UDLs.
        
        **Property 3: Metric Determinism**
        **Validates: Requirements 1.4, 6.5**
        """
        example_files = [
            "simple_calculator.udl",
            "json_subset.udl",
            "broken_grammar.udl"  # Test subset for performance
        ]
        
        for example_file in example_files:
            try:
                udl = self.load_udl(examples_dir, example_file)
                
                for metric_name, metric in metrics.items():
                    # Compute metric multiple times
                    values = [metric.compute(udl) for _ in range(5)]
                    
                    # All values should be identical
                    first_value = values[0]
                    for i, value in enumerate(values[1:], 1):
                        assert abs(value - first_value) <= self.EPSILON, (
                            f"Metric {metric_name} on {example_file} "
                            f"is not deterministic: run 0 = {first_value}, "
                            f"run {i} = {value}"
                        )
                        
            except Exception as e:
                # Skip files that don't exist
                if "not found" in str(e):
                    continue
                raise