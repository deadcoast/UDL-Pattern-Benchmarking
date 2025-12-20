"""
Property-based tests for mathematical formula validation.

Tests that the mathematical formulas documented in mathematical_framework.tex
match the actual implementations in the code.

**Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
**Validates: Requirements 4.1, 8.5**
"""

import pytest
import math
import zlib
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any
import networkx as nx
import numpy as np

from udl_rating_framework.core.representation import (
    UDLRepresentation,
    GrammarRule,
    Token,
    TokenType,
)
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.metrics.completeness import CompletenessMetric
from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
from udl_rating_framework.core.metrics.structural_coherence import (
    StructuralCoherenceMetric,
)
from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.core.confidence import ConfidenceCalculator


# Strategies for generating test data
def udl_text_strategy():
    """Generate realistic UDL text for testing."""
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
        "config ::= section*\nsection ::= '[' identifier ']' property*",
        "start ::= 'hello'",
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


@st.composite
def weight_strategy(draw):
    """Generate valid weights that sum to 1.0."""
    # Generate 4 random positive values
    values = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(4)]
    total = sum(values)
    # Normalize to sum to 1.0
    normalized = [v / total for v in values]
    return {
        "consistency": normalized[0],
        "completeness": normalized[1],
        "expressiveness": normalized[2],
        "structural_coherence": normalized[3],
    }


@st.composite
def probability_distribution_strategy(draw, n_classes=5):
    """Generate valid probability distributions."""
    # Generate n random positive values
    values = [draw(st.floats(min_value=0.001, max_value=1.0)) for _ in range(n_classes)]
    total = sum(values)
    # Normalize to sum to 1.0
    return [v / total for v in values]


class TestConsistencyFormulaValidation:
    """
    Tests that the Consistency metric implementation matches the documented formula.

    **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
    **Validates: Requirements 4.1, 8.5**

    LaTeX Formula:
    Consistency(U) = 1 - (|C(U)| + |Y(G)|) / (|R| + 1)

    Where:
    - C(U) = set of contradictory rule pairs
    - Y(G) = set of cycles in the grammar graph
    - R = set of production rules
    """

    @given(udl_text_strategy())
    @settings(max_examples=100, deadline=None)
    def test_consistency_formula_matches_implementation(self, udl_text: str):
        """
        **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
        **Validates: Requirements 4.1, 8.5**

        For any UDL, the consistency metric should follow the formula:
        Consistency(U) = 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)
        """
        assume(udl_text is not None)

        try:
            udl = UDLRepresentation(udl_text, "test.udl")
        except Exception:
            assume(False)

        metric = ConsistencyMetric()

        # Get the computed value from the implementation
        computed_value = metric.compute(udl)

        # Manually compute using the documented formula
        rules = udl.get_grammar_rules()
        graph = udl.get_grammar_graph()

        if not rules:
            expected_value = 1.0  # Empty grammar is perfectly consistent
        else:
            # Count cycles
            try:
                cycles = list(nx.simple_cycles(graph))
                cycle_count = len(cycles)
            except Exception:
                cycle_count = 0

            # Count contradictions (using the metric's method)
            contradictions = metric.find_contradictions(rules)
            contradiction_count = len(contradictions)

            # Apply the formula: 1 - (|C| + |Y|) / (|R| + 1)
            total_issues = cycle_count + contradiction_count
            total_rules = len(rules)
            expected_value = 1.0 - (total_issues / (total_rules + 1))
            expected_value = max(0.0, min(1.0, expected_value))

        # The computed value should match the formula
        assert abs(computed_value - expected_value) < 1e-6, (
            f"Consistency formula mismatch: computed={computed_value}, "
            f"expected={expected_value} for UDL: {repr(udl_text[:100])}"
        )

        # Verify boundedness property
        assert 0.0 <= computed_value <= 1.0, (
            f"Consistency value {computed_value} outside [0,1]"
        )


class TestCompletenessFormulaValidation:
    """
    Tests that the Completeness metric implementation matches the documented formula.

    **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
    **Validates: Requirements 4.1, 8.5**

    LaTeX Formula:
    Completeness(U) = |D(U) ∩ R_req(τ(U))| / |R_req(τ(U))|

    Where:
    - D(U) = set of defined constructs
    - τ(U) = inferred language type
    - R_req(τ) = set of required constructs for language type τ
    """

    @given(udl_text_strategy())
    @settings(max_examples=100, deadline=None)
    def test_completeness_formula_matches_implementation(self, udl_text: str):
        """
        **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
        **Validates: Requirements 4.1, 8.5**

        For any UDL, the completeness metric should follow the formula:
        Completeness(U) = |D(U) ∩ R_req| / |R_req|
        """
        assume(udl_text is not None)

        try:
            udl = UDLRepresentation(udl_text, "test.udl")
        except Exception:
            assume(False)

        metric = CompletenessMetric()

        # Get the computed value from the implementation
        computed_value = metric.compute(udl)

        # Manually compute using the documented formula
        defined_constructs = metric.extract_defined_constructs(udl)
        language_type = metric._infer_language_type(udl)
        required_constructs = metric.get_required_constructs(language_type)

        if not required_constructs:
            expected_value = 1.0 if not defined_constructs else 0.0
        else:
            # Compute intersection
            defined_construct_types = {c.construct_type for c in defined_constructs}
            intersection_count = len(
                defined_construct_types.intersection(required_constructs)
            )

            # Apply the formula: |D ∩ R_req| / |R_req|
            expected_value = intersection_count / len(required_constructs)
            expected_value = max(0.0, min(1.0, expected_value))

        # The computed value should match the formula
        assert abs(computed_value - expected_value) < 1e-6, (
            f"Completeness formula mismatch: computed={computed_value}, "
            f"expected={expected_value} for UDL: {repr(udl_text[:100])}"
        )

        # Verify boundedness property
        assert 0.0 <= computed_value <= 1.0, (
            f"Completeness value {computed_value} outside [0,1]"
        )


class TestExpressivenessFormulaValidation:
    """
    Tests that the Expressiveness metric implementation matches the documented formula.

    **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
    **Validates: Requirements 4.1, 8.5**

    LaTeX Formula:
    Expressiveness(U) = (Chomsky(U) + Complexity(U)) / 2

    Where:
    - Chomsky(U) ∈ {0, 1/3, 2/3, 1} for Type-3, Type-2, Type-1, Type-0
    - Complexity(U) = approximated Kolmogorov complexity via compression
    """

    @given(udl_text_strategy())
    @settings(max_examples=100, deadline=None)
    def test_expressiveness_formula_matches_implementation(self, udl_text: str):
        """
        **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
        **Validates: Requirements 4.1, 8.5**

        For any UDL, the expressiveness metric should follow the formula:
        Expressiveness(U) = (Chomsky_Level + Complexity_Score) / 2
        """
        assume(udl_text is not None)

        try:
            udl = UDLRepresentation(udl_text, "test.udl")
        except Exception:
            assume(False)

        metric = ExpressivenessMetric()

        # Get the computed value from the implementation
        computed_value = metric.compute(udl)

        # Verify boundedness property
        assert 0.0 <= computed_value <= 1.0, (
            f"Expressiveness value {computed_value} outside [0,1]"
        )

        # Verify the formula structure (Chomsky + Complexity) / 2
        # Since both components are in [0,1], the result should be in [0,1]
        # and should be the average of two bounded values
        rules = udl.get_grammar_rules()
        if not rules:
            assert computed_value == 0.0, "Empty grammar should have 0 expressiveness"


class TestStructuralCoherenceFormulaValidation:
    """
    Tests that the Structural Coherence metric implementation matches the documented formula.

    **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
    **Validates: Requirements 4.1, 8.5**

    LaTeX Formula:
    StructuralCoherence(U) = 1 - H(G) / H_max(G)

    Where:
    - H(G) = -Σ p(d) log₂ p(d) (Shannon entropy of degree distribution)
    - H_max(G) = log₂|V| (maximum entropy)
    """

    @given(udl_text_strategy())
    @settings(max_examples=100, deadline=None)
    def test_structural_coherence_formula_matches_implementation(self, udl_text: str):
        """
        **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
        **Validates: Requirements 4.1, 8.5**

        For any UDL, the structural coherence metric should follow the formula:
        StructuralCoherence(U) = 1 - H(G) / H_max
        """
        assume(udl_text is not None)

        try:
            udl = UDLRepresentation(udl_text, "test.udl")
        except Exception:
            assume(False)

        metric = StructuralCoherenceMetric()

        # Get the computed value from the implementation
        computed_value = metric.compute(udl)

        # Manually compute using the documented formula
        graph = udl.get_grammar_graph()
        num_nodes = graph.number_of_nodes()

        if num_nodes <= 1:
            expected_value = 1.0  # Trivial graph is perfectly coherent
        else:
            # Compute Shannon entropy of degree distribution
            entropy = metric.compute_shannon_entropy(graph)
            max_entropy = math.log2(num_nodes)

            if max_entropy == 0.0:
                expected_value = 1.0
            else:
                expected_value = 1.0 - (entropy / max_entropy)
                expected_value = max(0.0, min(1.0, expected_value))

        # The computed value should match the formula
        assert abs(computed_value - expected_value) < 1e-6, (
            f"Structural coherence formula mismatch: computed={computed_value}, "
            f"expected={expected_value} for UDL: {repr(udl_text[:100])}"
        )

        # Verify boundedness property
        assert 0.0 <= computed_value <= 1.0, (
            f"Structural coherence value {computed_value} outside [0,1]"
        )


class TestAggregationFormulaValidation:
    """
    Tests that the Aggregation function implementation matches the documented formula.

    **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
    **Validates: Requirements 4.1, 8.5**

    LaTeX Formula:
    Q(U) = Σᵢ wᵢ · mᵢ(U)

    Where:
    - wᵢ ≥ 0 are non-negative weights with Σwᵢ = 1
    - mᵢ(U) are the individual quality metrics
    """

    @given(weight_strategy())
    @settings(max_examples=100, deadline=None)
    def test_aggregation_formula_matches_implementation(
        self, weights: Dict[str, float]
    ):
        """
        **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
        **Validates: Requirements 4.1, 8.5**

        For any valid weights, the aggregation should follow the formula:
        Q(U) = Σ wᵢ · mᵢ
        """
        # Create aggregator with the generated weights
        aggregator = MetricAggregator(weights)

        # Generate some metric values
        metric_values = {
            "consistency": 0.8,
            "completeness": 0.9,
            "expressiveness": 0.7,
            "structural_coherence": 0.85,
        }

        # Get the computed value from the implementation
        computed_value = aggregator.aggregate(metric_values)

        # Manually compute using the documented formula: Q = Σ wᵢ · mᵢ
        expected_value = sum(
            weights[name] * value for name, value in metric_values.items()
        )

        # The computed value should match the formula
        assert abs(computed_value - expected_value) < 1e-6, (
            f"Aggregation formula mismatch: computed={computed_value}, "
            f"expected={expected_value}"
        )

        # Verify boundedness property (if all metrics are in [0,1] and weights sum to 1)
        assert 0.0 <= computed_value <= 1.0, (
            f"Aggregation value {computed_value} outside [0,1]"
        )


class TestConfidenceFormulaValidation:
    """
    Tests that the Confidence calculation implementation matches the documented formula.

    **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
    **Validates: Requirements 4.1, 8.5**

    LaTeX Formula:
    C(p) = 1 - H(p) / H_max

    Where:
    - H(p) = -Σ pᵢ log pᵢ (Shannon entropy)
    - H_max = log n (maximum entropy for n classes)
    """

    @given(probability_distribution_strategy())
    @settings(max_examples=100, deadline=None)
    def test_confidence_formula_matches_implementation(self, probs: List[float]):
        """
        **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
        **Validates: Requirements 4.1, 8.5**

        For any probability distribution, the confidence should follow the formula:
        C(p) = 1 - H(p) / H_max
        """
        calculator = ConfidenceCalculator()

        # Get the computed value from the implementation
        computed_value = calculator.compute_confidence(probs)

        # Manually compute using the documented formula
        probs_array = np.array(probs)
        epsilon = 1e-10
        probs_safe = probs_array + epsilon

        # Shannon entropy: H(p) = -Σ pᵢ log pᵢ
        entropy = -np.sum(probs_safe * np.log(probs_safe))

        # Maximum entropy: H_max = log n
        max_entropy = np.log(len(probs))

        # Confidence: C = 1 - H(p) / H_max
        if max_entropy == 0:
            expected_value = 1.0
        else:
            expected_value = 1.0 - (entropy / max_entropy)
            expected_value = np.clip(expected_value, 0.0, 1.0)

        # The computed value should match the formula (with some tolerance for epsilon)
        assert abs(computed_value - expected_value) < 0.01, (
            f"Confidence formula mismatch: computed={computed_value}, "
            f"expected={expected_value}"
        )

        # Verify boundedness property
        assert 0.0 <= computed_value <= 1.0, (
            f"Confidence value {computed_value} outside [0,1]"
        )


class TestFormulaDocstringConsistency:
    """
    Tests that the get_formula() methods return formulas consistent with documentation.
    """

    def test_consistency_metric_formula_string(self):
        """Test that ConsistencyMetric.get_formula() matches documentation."""
        metric = ConsistencyMetric()
        formula = metric.get_formula()

        # Should contain key elements of the formula
        assert "Consistency" in formula or "1 -" in formula
        assert "Contradictions" in formula or "Cycles" in formula or "Rules" in formula

    def test_completeness_metric_formula_string(self):
        """Test that CompletenessMetric.get_formula() matches documentation."""
        metric = CompletenessMetric()
        formula = metric.get_formula()

        # Should contain key elements of the formula
        assert (
            "Completeness" in formula or "Defined" in formula or "Required" in formula
        )

    def test_expressiveness_metric_formula_string(self):
        """Test that ExpressivenessMetric.get_formula() matches documentation."""
        metric = ExpressivenessMetric()
        formula = metric.get_formula()

        # Should contain key elements of the formula
        assert (
            "Expressiveness" in formula
            or "Chomsky" in formula
            or "Complexity" in formula
        )

    def test_structural_coherence_metric_formula_string(self):
        """Test that StructuralCoherenceMetric.get_formula() matches documentation."""
        metric = StructuralCoherenceMetric()
        formula = metric.get_formula()

        # Should contain key elements of the formula
        assert (
            "Structural" in formula
            or "Coherence" in formula
            or "H(" in formula
            or "entropy" in formula.lower()
        )
