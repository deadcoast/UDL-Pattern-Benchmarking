"""
Property-based and unit tests for ConsistencyMetric.

Tests the mathematical correctness of the consistency metric implementation.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import networkx as nx
from typing import List
from udl_rating_framework.core.representation import (
    UDLRepresentation,
    GrammarRule,
    Constraint,
)
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric


def create_udl_with_known_cycles_and_contradictions(
    num_rules: int, num_cycles: int, num_contradictions: int
) -> UDLRepresentation:
    """
    Create a UDL with known numbers of cycles and contradictions for testing.

    Args:
        num_rules: Total number of rules to create
        num_cycles: Number of cycles to introduce
        num_contradictions: Number of contradictions to introduce

    Returns:
        UDLRepresentation with specified characteristics
    """
    rules_text = []

    # Create basic rules without cycles or contradictions first
    base_rules = max(1, num_rules - num_cycles - num_contradictions)
    for i in range(base_rules):
        rules_text.append(f"rule{i} ::= 'terminal{i}'")

    # Add cycles by creating circular dependencies
    cycle_start_idx = base_rules
    for i in range(num_cycles):
        if i == 0:
            # Create a simple 2-node cycle: A -> B -> A
            rules_text.append(f"cycleA{i} ::= cycleB{i}")
            rules_text.append(f"cycleB{i} ::= cycleA{i}")
        else:
            # Create longer cycles
            rules_text.append(f"cycleA{i} ::= cycleB{i}")
            rules_text.append(f"cycleB{i} ::= cycleC{i}")
            rules_text.append(f"cycleC{i} ::= cycleA{i}")

    # Add contradictions by creating rules with same LHS but conflicting RHS
    contradiction_start_idx = cycle_start_idx + num_cycles * 2
    for i in range(num_contradictions):
        # Create contradictory rules with same LHS
        rules_text.append(f"conflict{i} ::= 'true'")
        rules_text.append(f"conflict{i} ::= 'false'")

    # Join all rules into UDL text
    udl_text = "\n".join(rules_text)
    return UDLRepresentation(udl_text, "test.udl")


class TestConsistencyMetricProperties:
    """Property-based tests for ConsistencyMetric."""

    def test_consistency_metric_correctness_property(self):
        """
        **Feature: udl-rating-framework, Property 9: Consistency Metric Correctness**
        **Validates: Requirements 3.2**

        For any UDL with grammar graph G, the consistency score must be computed as
        1 - (|Contradictions| + |Cycles|) / (|Rules| + 1).
        """
        # Test with several known configurations
        test_cases = [
            (3, 0, 0),  # 3 rules, no cycles, no contradictions
            (4, 1, 0),  # 4 rules, 1 cycle, no contradictions
            (4, 0, 1),  # 4 rules, no cycles, 1 contradiction
            (5, 1, 1),  # 5 rules, 1 cycle, 1 contradiction
            (2, 1, 0),  # 2 rules, 1 cycle, no contradictions
            (2, 0, 1),  # 2 rules, no cycles, 1 contradiction
        ]

        for num_rules, num_cycles, num_contradictions in test_cases:
            # Ensure we don't create more issues than we have capacity for
            if num_cycles + num_contradictions > num_rules:
                continue

            # Create UDL with known characteristics
            udl = create_udl_with_known_cycles_and_contradictions(
                num_rules, num_cycles, num_contradictions
            )

            # Create consistency metric
            metric = ConsistencyMetric()

            # Compute the consistency score
            computed_score = metric.compute(udl)

            # Manually verify the cycle and contradiction counts
            graph = udl.get_grammar_graph()
            detected_cycles = metric.detect_cycles(graph)
            detected_contradictions = metric.find_contradictions(
                udl.get_grammar_rules()
            )

            actual_rules = len(udl.get_grammar_rules())
            actual_cycles = len(detected_cycles)
            actual_contradictions = len(detected_contradictions)

            # Compute expected score using the formula
            expected_score = 1.0 - (actual_contradictions + actual_cycles) / (
                actual_rules + 1
            )
            expected_score = max(0.0, min(1.0, expected_score))  # Ensure bounded

            # Verify the formula is correctly implemented
            assert abs(computed_score - expected_score) < 1e-6, (
                f"Consistency metric formula incorrect for case ({num_rules}, {num_cycles}, {num_contradictions}). "
                f"Expected: {expected_score}, Got: {computed_score}. "
                f"Rules: {actual_rules}, Cycles: {actual_cycles}, "
                f"Contradictions: {actual_contradictions}"
            )

            # Verify boundedness
            assert (
                0.0 <= computed_score <= 1.0
            ), f"Consistency score {computed_score} not in [0,1] for case ({num_rules}, {num_cycles}, {num_contradictions})"


class TestConsistencyMetricUnits:
    """Unit tests for ConsistencyMetric."""

    def test_empty_grammar_perfect_consistency(self):
        """Test on UDL with no contradictions/cycles (expect 1.0)."""
        # Empty grammar should be perfectly consistent
        udl = UDLRepresentation("", "empty.udl")
        metric = ConsistencyMetric()

        score = metric.compute(udl)
        assert score == 1.0, f"Empty grammar should have consistency 1.0, got {score}"

    def test_simple_grammar_no_issues(self):
        """Test on simple grammar with no cycles or contradictions."""
        udl_text = """
        expr ::= term
        term ::= factor
        factor ::= 'number'
        """
        udl = UDLRepresentation(udl_text, "simple.udl")
        metric = ConsistencyMetric()

        score = metric.compute(udl)

        # Should be perfect consistency (no cycles or contradictions)
        # Formula: 1 - (0 + 0) / (3 + 1) = 1.0
        assert score == 1.0, f"Simple grammar should have consistency 1.0, got {score}"

    def test_grammar_with_known_contradictions(self):
        """Test on UDL with known contradictions."""
        udl_text = """
        boolean ::= 'true'
        boolean ::= 'false'
        """
        udl = UDLRepresentation(udl_text, "contradictory.udl")
        metric = ConsistencyMetric()

        score = metric.compute(udl)

        # Should detect the contradiction between 'true' and 'false'
        # Formula: 1 - (1 + 0) / (2 + 1) = 1 - 1/3 = 2/3 ≈ 0.667
        expected_score = 1.0 - (1 + 0) / (2 + 1)
        assert (
            abs(score - expected_score) < 1e-6
        ), f"Expected consistency {expected_score}, got {score}"

    def test_grammar_with_known_cycles(self):
        """Test on UDL with known cycles."""
        udl_text = """
        expr ::= term
        term ::= expr
        """
        udl = UDLRepresentation(udl_text, "cyclic.udl")
        metric = ConsistencyMetric()

        score = metric.compute(udl)

        # Should detect the cycle between expr and term
        # Formula: 1 - (0 + 1) / (2 + 1) = 1 - 1/3 = 2/3 ≈ 0.667
        expected_score = 1.0 - (0 + 1) / (2 + 1)
        assert (
            abs(score - expected_score) < 1e-6
        ), f"Expected consistency {expected_score}, got {score}"

    def test_grammar_with_cycles_and_contradictions(self):
        """Test on UDL with both cycles and contradictions."""
        udl_text = """
        expr ::= term
        term ::= expr
        boolean ::= 'true'
        boolean ::= 'false'
        """
        udl = UDLRepresentation(udl_text, "complex.udl")
        metric = ConsistencyMetric()

        score = metric.compute(udl)

        # Should detect 1 cycle and 1 contradiction
        # Formula: 1 - (1 + 1) / (4 + 1) = 1 - 2/5 = 3/5 = 0.6
        expected_score = 1.0 - (1 + 1) / (4 + 1)
        assert (
            abs(score - expected_score) < 1e-6
        ), f"Expected consistency {expected_score}, got {score}"

    def test_cycle_detection_method(self):
        """Test the cycle detection method directly."""
        udl_text = """
        A ::= B
        B ::= C  
        C ::= A
        D ::= E
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = ConsistencyMetric()

        graph = udl.get_grammar_graph()
        cycles = metric.detect_cycles(graph)

        # Should find one cycle: A -> B -> C -> A
        assert len(cycles) >= 1, f"Expected at least 1 cycle, found {len(cycles)}"

        # Check that the cycle contains the expected nodes
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)

        expected_cycle_nodes = {"A", "B", "C"}
        assert expected_cycle_nodes.issubset(
            cycle_nodes
        ), f"Expected cycle nodes {expected_cycle_nodes} not found in {cycle_nodes}"

    def test_contradiction_detection_method(self):
        """Test the contradiction detection method directly."""
        udl_text = """
        value ::= 'true'
        value ::= 'false'
        other ::= 'something'
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = ConsistencyMetric()

        rules = udl.get_grammar_rules()
        contradictions = metric.find_contradictions(rules)

        # Should find contradiction between 'true' and 'false' rules
        assert (
            len(contradictions) >= 1
        ), f"Expected at least 1 contradiction, found {len(contradictions)}"

        # Verify the contradiction involves the expected rules
        contradiction_lhs = set()
        for rule1, rule2 in contradictions:
            contradiction_lhs.add(rule1.lhs)
            contradiction_lhs.add(rule2.lhs)

        assert (
            "value" in contradiction_lhs
        ), f"Expected 'value' in contradiction LHS, got {contradiction_lhs}"

    def test_metric_properties(self):
        """Test that the metric reports correct mathematical properties."""
        metric = ConsistencyMetric()
        properties = metric.get_properties()

        assert properties["bounded"] == True
        assert properties["monotonic"] == False
        assert properties["additive"] == False
        assert properties["continuous"] == False

    def test_metric_formula(self):
        """Test that the metric returns the correct LaTeX formula."""
        metric = ConsistencyMetric()
        formula = metric.get_formula()

        expected_formula = (
            r"Consistency(U) = 1 - \frac{|Contradictions| + |Cycles|}{|Rules| + 1}"
        )
        assert (
            formula == expected_formula
        ), f"Expected formula {expected_formula}, got {formula}"

    def test_boundedness_verification(self):
        """Test that the metric satisfies boundedness property."""
        test_cases = [
            "",  # Empty
            "A ::= 'a'",  # Simple
            "A ::= B\nB ::= A",  # Cycle
            "X ::= 'true'\nX ::= 'false'",  # Contradiction
        ]

        metric = ConsistencyMetric()

        for udl_text in test_cases:
            udl = UDLRepresentation(udl_text, "test.udl")
            score = metric.compute(udl)

            assert (
                0.0 <= score <= 1.0
            ), f"Score {score} not in [0,1] for UDL: {repr(udl_text)}"

            # Also test the verification method
            assert metric.verify_boundedness(
                udl
            ), f"Boundedness verification failed for UDL: {repr(udl_text)}"

    def test_determinism_verification(self):
        """Test that the metric satisfies determinism property."""
        udl_text = """
        expr ::= term '+' expr
        term ::= factor '*' term
        factor ::= '(' expr ')' | 'number'
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = ConsistencyMetric()

        # Compute multiple times
        scores = [metric.compute(udl) for _ in range(5)]

        # All should be identical
        first_score = scores[0]
        for score in scores[1:]:
            assert score == first_score, f"Non-deterministic behavior: got {scores}"

        # Also test the verification method
        assert metric.verify_determinism(udl), "Determinism verification failed"
