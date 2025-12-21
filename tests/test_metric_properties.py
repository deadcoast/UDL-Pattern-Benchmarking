"""
Property-based tests for quality metric properties.

Tests the mathematical properties that all metrics must satisfy:
- Boundedness: All metrics must produce values in [0,1]
- Determinism: Same input must always produce same output
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from udl_rating_framework.core.metrics.base import MetricRegistry, QualityMetric
from udl_rating_framework.core.representation import UDLRepresentation


# Test metric implementations for property testing
class TestConsistencyMetric(QualityMetric):
    """Test implementation of consistency metric for property testing."""

    def compute(self, udl: UDLRepresentation) -> float:
        """Simple consistency computation for testing."""
        rules = udl.get_grammar_rules()
        if not rules:
            return 1.0  # Empty grammar is perfectly consistent

        # Simple consistency: 1 - (number of rules with cycles) / total rules
        graph = udl.get_grammar_graph()
        try:
            import networkx as nx

            cycles = list(nx.simple_cycles(graph))
            cycle_count = len(cycles)
            return max(0.0, min(1.0, 1.0 - (cycle_count / (len(rules) + 1))))
        except Exception:
            return 0.5  # Default if cycle detection fails

    def get_formula(self) -> str:
        return r"C = 1 - \frac{|Cycles|}{|Rules| + 1}"

    def get_properties(self) -> dict:
        return {
            "bounded": True,
            "monotonic": False,
            "additive": False,
            "continuous": False,
        }


class TestCompletenessMetric(QualityMetric):
    """Test implementation of completeness metric for property testing."""

    def compute(self, udl: UDLRepresentation) -> float:
        """Simple completeness computation for testing."""
        tokens = udl.get_tokens()
        if not tokens:
            return 0.0

        # Simple completeness: ratio of unique identifiers to total tokens
        from udl_rating_framework.core.representation import TokenType

        identifier_tokens = [
            t for t in tokens if t.type == TokenType.IDENTIFIER]
        unique_identifiers = len(set(t.text for t in identifier_tokens))
        total_meaningful_tokens = len(
            [
                t
                for t in tokens
                if t.type
                in [TokenType.IDENTIFIER, TokenType.OPERATOR, TokenType.LITERAL]
            ]
        )

        if total_meaningful_tokens == 0:
            return 0.0

        return min(1.0, unique_identifiers / max(1, total_meaningful_tokens))

    def get_formula(self) -> str:
        return r"Comp = \frac{|Unique\_Identifiers|}{|Total\_Tokens|}"

    def get_properties(self) -> dict:
        return {
            "bounded": True,
            "monotonic": True,
            "additive": False,
            "continuous": True,
        }


class TestBrokenBoundednessMetric(QualityMetric):
    """Intentionally broken metric that violates boundedness for testing."""

    def compute(self, udl: UDLRepresentation) -> float:
        """Returns values outside [0,1] to test boundedness validation."""
        tokens = udl.get_tokens()
        return len(tokens) * 2.0  # Intentionally unbounded

    def get_formula(self) -> str:
        return r"Broken = 2 \cdot |Tokens|"

    def get_properties(self) -> dict:
        return {
            "bounded": False,  # Honestly reports it's not bounded
            "monotonic": True,
            "additive": True,
            "continuous": True,
        }


class TestNonDeterministicMetric(QualityMetric):
    """Intentionally non-deterministic metric for testing."""

    def compute(self, udl: UDLRepresentation) -> float:
        """Returns random values to test determinism validation."""
        import random

        return random.random()  # Intentionally non-deterministic

    def get_formula(self) -> str:
        return r"Random = \text{random}()"

    def get_properties(self) -> dict:
        return {
            "bounded": True,
            "monotonic": False,
            "additive": False,
            "continuous": False,
        }


# Strategy for generating UDL text
def udl_text_strategy():
    """Generate realistic UDL text for testing."""
    # Simple grammar patterns
    simple_grammars = [
        "expr ::= term",
        "expr ::= term '+' expr | term",
        "term ::= factor '*' term | factor",
        "factor ::= '(' expr ')' | number",
        "number ::= digit+",
        "digit ::= '0' | '1' | '2'",
        "",  # Empty grammar
        "# Just a comment",
        "stmt := assignment | expression",
        "assignment = id '=' expr",
    ]

    return st.one_of(
        [
            st.sampled_from(simple_grammars),
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz ::=|()'+*0123456789\n# ",
                min_size=0,
                max_size=200,
            ),
        ]
    )


class TestMetricBoundedness:
    """Property-based tests for metric boundedness."""

    def setup_method(self):
        """Set up test metrics."""
        MetricRegistry.clear()
        MetricRegistry.register("test_consistency", TestConsistencyMetric)
        MetricRegistry.register("test_completeness", TestCompletenessMetric)

    def teardown_method(self):
        """Clean up after tests and re-register default metrics."""
        MetricRegistry.clear()
        # Re-register default metrics for other tests
        from udl_rating_framework.core.metrics import _register_default_metrics

        _register_default_metrics()

    @given(udl_text_strategy())
    @settings(max_examples=100, deadline=None)
    def test_metric_boundedness_property(self, udl_text: str):
        """
        **Feature: udl-rating-framework, Property 2: Metric Boundedness**
        **Validates: Requirements 1.4, 3.7**

        For any UDL and any quality metric m, the computed value must satisfy 0 ≤ m(UDL) ≤ 1.
        """
        # Skip empty or whitespace-only strings that might cause parsing issues
        assume(udl_text is not None)

        try:
            udl = UDLRepresentation(udl_text, "test.udl")
        except Exception:
            # Skip UDLs that can't be parsed
            assume(False)

        # Test all registered metrics that claim to be bounded
        for metric_name, metric_class in MetricRegistry.list_metrics().items():
            metric = metric_class()

            # Only test metrics that claim to be bounded
            properties = metric.get_properties()
            if properties.get("bounded", False):
                try:
                    value = metric.compute(udl)

                    assert 0.0 <= value <= 1.0, (
                        f"Metric {metric_name} produced value {value} outside [0,1] "
                        f"for UDL: {repr(udl_text[:100])}"
                    )

                    # Also test the verification method
                    assert metric.verify_boundedness(udl), (
                        f"Metric {metric_name} failed boundedness verification "
                        f"for UDL: {repr(udl_text[:100])}"
                    )

                except Exception as e:
                    # If computation fails, that's also a test failure for bounded metrics
                    pytest.fail(
                        f"Bounded metric {metric_name} failed to compute: {e}")


class TestMetricDeterminism:
    """Property-based tests for metric determinism."""

    def setup_method(self):
        """Set up test metrics."""
        MetricRegistry.clear()
        MetricRegistry.register("test_consistency", TestConsistencyMetric)
        MetricRegistry.register("test_completeness", TestCompletenessMetric)

    def teardown_method(self):
        """Clean up after tests and re-register default metrics."""
        MetricRegistry.clear()
        # Re-register default metrics for other tests
        from udl_rating_framework.core.metrics import _register_default_metrics

        _register_default_metrics()

    @given(udl_text_strategy())
    @settings(max_examples=100, deadline=None)
    def test_metric_determinism_property(self, udl_text: str):
        """
        **Feature: udl-rating-framework, Property 3: Metric Determinism**
        **Validates: Requirements 1.4, 6.5**

        For any UDL, computing the same metric multiple times must produce identical results.
        """
        # Skip empty or whitespace-only strings that might cause parsing issues
        assume(udl_text is not None)

        try:
            udl = UDLRepresentation(udl_text, "test.udl")
        except Exception:
            # Skip UDLs that can't be parsed
            assume(False)

        # Test all registered metrics
        for metric_name, metric_class in MetricRegistry.list_metrics().items():
            metric = metric_class()

            try:
                # Compute metric multiple times
                values = []
                for _ in range(5):  # Test with 5 computations
                    value = metric.compute(udl)
                    values.append(value)

                # All values should be identical
                first_value = values[0]
                for i, value in enumerate(values[1:], 1):
                    assert value == first_value, (
                        f"Metric {metric_name} is not deterministic: "
                        f"computation 0 gave {first_value}, computation {i} gave {value} "
                        f"for UDL: {repr(udl_text[:100])}"
                    )

                # Also test the verification method
                assert metric.verify_determinism(udl, trials=5), (
                    f"Metric {metric_name} failed determinism verification "
                    f"for UDL: {repr(udl_text[:100])}"
                )

            except Exception as e:
                # If computation fails consistently, that's still deterministic
                # But if it fails inconsistently, that's non-deterministic
                try:
                    # Try again to see if failure is consistent
                    metric.compute(udl)
                    # If this succeeds but previous failed, it's non-deterministic
                    pytest.fail(
                        f"Metric {metric_name} has inconsistent failures: {e}")
                except Exception:
                    # Consistent failure is acceptable (though not ideal)
                    pass


class TestMetricRegistry:
    """Unit tests for the metric registry."""

    def setup_method(self):
        """Clean registry before each test."""
        MetricRegistry.clear()

    def teardown_method(self):
        """Clean registry after each test and re-register default metrics."""
        MetricRegistry.clear()
        # Re-register default metrics for other tests
        from udl_rating_framework.core.metrics import _register_default_metrics

        _register_default_metrics()

    def test_metric_registration(self):
        """Test basic metric registration."""
        MetricRegistry.register("test_metric", TestConsistencyMetric)

        assert "test_metric" in MetricRegistry.list_metrics()
        retrieved_class = MetricRegistry.get_metric("test_metric")
        assert retrieved_class == TestConsistencyMetric

    def test_metric_registration_validation(self):
        """Test that registration validates metric classes."""
        # Should raise TypeError for non-QualityMetric classes
        with pytest.raises(TypeError):
            MetricRegistry.register("invalid", str)

    def test_metric_unregistration(self):
        """Test metric unregistration."""
        MetricRegistry.register("test_metric", TestConsistencyMetric)
        assert "test_metric" in MetricRegistry.list_metrics()

        MetricRegistry.unregister("test_metric")
        assert "test_metric" not in MetricRegistry.list_metrics()

    def test_get_nonexistent_metric(self):
        """Test getting a metric that doesn't exist."""
        with pytest.raises(KeyError):
            MetricRegistry.get_metric("nonexistent")

    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        registry1 = MetricRegistry()
        registry2 = MetricRegistry()
        assert registry1 is registry2

    def test_metric_property_validation(self):
        """Test validation of metric properties."""
        # Create a UDL for testing
        udl = UDLRepresentation("expr ::= term", "test.udl")

        # Test well-behaved metric
        good_metric = TestConsistencyMetric()
        results = good_metric.validate_properties(udl)
        assert results["bounded"]
        assert results["deterministic"]

        # Test broken boundedness metric
        broken_metric = TestBrokenBoundednessMetric()
        results = broken_metric.validate_properties(udl)
        assert not results["bounded"]  # Should fail boundedness

        # Test non-deterministic metric
        random_metric = TestNonDeterministicMetric()
        results = random_metric.validate_properties(udl)
        assert not results["deterministic"]  # Should fail determinism

    def test_class_registration_method(self):
        """Test the class method for self-registration."""
        TestConsistencyMetric.register_metric("self_registered")

        assert "self_registered" in MetricRegistry.list_metrics()
        retrieved_class = MetricRegistry.get_metric("self_registered")
        assert retrieved_class == TestConsistencyMetric


class TestMetricValidation:
    """Tests for metric validation methods."""

    def test_boundedness_validation_with_broken_metric(self):
        """Test boundedness validation catches violations."""
        udl = UDLRepresentation("expr ::= term '+' expr", "test.udl")
        broken_metric = TestBrokenBoundednessMetric()

        # Should return False for boundedness
        assert not broken_metric.verify_boundedness(udl)

    def test_determinism_validation_with_random_metric(self):
        """Test determinism validation catches violations."""
        udl = UDLRepresentation("expr ::= term", "test.udl")
        random_metric = TestNonDeterministicMetric()

        # Should return False for determinism (with high probability)
        # Run multiple times to be sure
        failures = 0
        for _ in range(10):
            if not random_metric.verify_determinism(udl, trials=3):
                failures += 1

        # Should fail at least some of the time due to randomness
        assert failures > 0, "Random metric should fail determinism validation"
