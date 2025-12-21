"""
Tests for advanced metrics implementation (Task 29).

Tests semantic similarity, readability, maintainability, cross-language compatibility,
and evolution tracking metrics.
"""

from udl_rating_framework.core.metrics.cross_language_compatibility import (
    CrossLanguageCompatibilityMetric,
)
from udl_rating_framework.core.metrics.evolution_tracking import EvolutionTrackingMetric
from udl_rating_framework.core.metrics.maintainability import MaintainabilityMetric
from udl_rating_framework.core.metrics.readability import ReadabilityMetric
from udl_rating_framework.core.metrics.semantic_similarity import (
    SemanticSimilarityMetric,
)
from udl_rating_framework.core.representation import UDLRepresentation


class TestSemanticSimilarityMetric:
    """Test semantic similarity metric."""

    def test_compute_basic(self):
        """Test basic semantic similarity computation."""
        udl_text = """
        expression ::= term '+' term
        term ::= factor '*' factor
        factor ::= number | identifier
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = SemanticSimilarityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_empty_udl(self):
        """Test semantic similarity with empty UDL."""
        udl = UDLRepresentation("", "empty.udl")
        metric = SemanticSimilarityMetric()

        score = metric.compute(udl)
        assert score == 0.0

    def test_single_construct(self):
        """Test semantic similarity with single construct."""
        udl_text = "identifier ::= letter+"
        udl = UDLRepresentation(udl_text, "single.udl")
        metric = SemanticSimilarityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0  # Single construct should produce valid score

    def test_properties(self):
        """Test metric properties."""
        metric = SemanticSimilarityMetric()
        properties = metric.get_properties()

        assert properties["bounded"] is True
        assert properties["continuous"] is True

    def test_formula(self):
        """Test formula string."""
        metric = SemanticSimilarityMetric()
        formula = metric.get_formula()
        assert "SemanticSimilarity" in formula
        assert "sim" in formula


class TestReadabilityMetric:
    """Test readability metric."""

    def test_compute_basic(self):
        """Test basic readability computation."""
        udl_text = """
        // Simple expression grammar
        expression ::= term ('+' term)*
        term ::= factor ('*' factor)*
        factor ::= number | identifier | '(' expression ')'
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = ReadabilityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_empty_udl(self):
        """Test readability with empty UDL."""
        udl = UDLRepresentation("", "empty.udl")
        metric = ReadabilityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0  # Empty UDL should produce valid score

    def test_complex_udl(self):
        """Test readability with complex UDL."""
        udl_text = """
        // Complex nested grammar
        program ::= ((((statement)*)*)*)*
        statement ::= if_stmt | while_stmt | assignment
        if_stmt ::= 'if' '(' expression ')' '{' statement* '}' ('else' '{' statement* '}')?
        """
        udl = UDLRepresentation(udl_text, "complex.udl")
        metric = ReadabilityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_properties(self):
        """Test metric properties."""
        metric = ReadabilityMetric()
        properties = metric.get_properties()

        assert properties["bounded"] is True
        assert properties["continuous"] is True

    def test_formula(self):
        """Test formula string."""
        metric = ReadabilityMetric()
        formula = metric.get_formula()
        assert "Readability" in formula
        assert "Flesch" in formula


class TestMaintainabilityMetric:
    """Test maintainability metric."""

    def test_compute_basic(self):
        """Test basic maintainability computation."""
        udl_text = """
        // Well-structured grammar
        program ::= statement*
        statement ::= assignment | if_statement
        assignment ::= identifier '=' expression
        expression ::= term ('+' term)*
        term ::= factor ('*' factor)*
        factor ::= number | identifier
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = MaintainabilityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_empty_udl(self):
        """Test maintainability with empty UDL."""
        udl = UDLRepresentation("", "empty.udl")
        metric = MaintainabilityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0  # Empty UDL should produce valid score

    def test_detailed_metrics(self):
        """Test detailed maintainability metrics."""
        udl_text = """
        // Simple grammar with comments
        expression ::= term '+' term  // Addition
        term ::= number | identifier  // Basic terms
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = MaintainabilityMetric()

        detailed = metric.get_detailed_metrics(udl)

        assert "halstead_volume" in detailed
        assert "cyclomatic_complexity" in detailed
        assert "lines_of_code" in detailed
        assert "comment_ratio" in detailed
        assert "maintainability_index" in detailed

        for value in detailed.values():
            assert isinstance(value, (int, float))
            assert value >= 0

    def test_properties(self):
        """Test metric properties."""
        metric = MaintainabilityMetric()
        properties = metric.get_properties()

        assert properties["bounded"] is True
        assert properties["continuous"] is True

    def test_formula(self):
        """Test formula string."""
        metric = MaintainabilityMetric()
        formula = metric.get_formula()
        assert "MI" in formula
        assert "HV" in formula


class TestCrossLanguageCompatibilityMetric:
    """Test cross-language compatibility metric."""

    def test_compute_basic(self):
        """Test basic compatibility computation."""
        udl_text = """
        expression ::= term | term '+' expression
        term ::= factor | factor '*' term
        factor ::= number | '(' expression ')'
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = CrossLanguageCompatibilityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_antlr_format(self):
        """Test compatibility with ANTLR format."""
        udl_text = """
        grammar TestGrammar;
        
        expression : term ('+' term)* ;
        term : factor ('*' factor)* ;
        factor : NUMBER | '(' expression ')' ;
        
        NUMBER : [0-9]+ ;
        """
        udl = UDLRepresentation(udl_text, "test.g4")
        metric = CrossLanguageCompatibilityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_peg_format(self):
        """Test compatibility with PEG format."""
        udl_text = """
        Expression <- Term ('+' Term)*
        Term <- Factor ('*' Factor)*
        Factor <- Number / '(' Expression ')'
        Number <- [0-9]+
        """
        udl = UDLRepresentation(udl_text, "test.peg")
        metric = CrossLanguageCompatibilityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_empty_udl(self):
        """Test compatibility with empty UDL."""
        udl = UDLRepresentation("", "empty.udl")
        metric = CrossLanguageCompatibilityMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_properties(self):
        """Test metric properties."""
        metric = CrossLanguageCompatibilityMetric()
        properties = metric.get_properties()

        assert properties["bounded"] is True
        assert properties["continuous"] is True

    def test_formula(self):
        """Test formula string."""
        metric = CrossLanguageCompatibilityMetric()
        formula = metric.get_formula()
        assert "Compatibility" in formula
        assert "Portability" in formula


class TestEvolutionTrackingMetric:
    """Test evolution tracking metric."""

    def test_compute_basic(self):
        """Test basic evolution tracking computation."""
        udl_text = """
        // Modular grammar design
        program ::= module*
        module ::= declaration*
        declaration ::= function_decl | variable_decl
        function_decl ::= 'function' identifier '(' parameter_list ')' block
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = EvolutionTrackingMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_version_comparison(self):
        """Test version comparison functionality."""
        udl_v1_text = """
        expression ::= term
        term ::= number
        """
        udl_v2_text = """
        expression ::= term | term '+' expression
        term ::= number | identifier
        identifier ::= letter+
        """

        udl_v1 = UDLRepresentation(udl_v1_text, "v1.udl")
        udl_v2 = UDLRepresentation(udl_v2_text, "v2.udl")

        metric = EvolutionTrackingMetric()
        comparison = metric.compare_versions(udl_v1, udl_v2)

        assert "structural_similarity" in comparison
        assert "added_constructs" in comparison
        assert "removed_constructs" in comparison
        assert "modified_rules" in comparison
        assert "compatibility_impact" in comparison
        assert "change_complexity" in comparison

        assert 0.0 <= comparison["structural_similarity"] <= 1.0
        assert isinstance(comparison["added_constructs"], set)
        assert isinstance(comparison["removed_constructs"], set)

    def test_version_history(self):
        """Test version history tracking."""
        udl_v1_text = "expression ::= number"
        udl_v2_text = "expression ::= number | identifier"

        udl_v1 = UDLRepresentation(udl_v1_text, "v1.udl")
        udl_v2 = UDLRepresentation(udl_v2_text, "v2.udl")

        metric = EvolutionTrackingMetric()
        metric.add_version("v1.0", udl_v1)

        # Compute score for v2 with v1 in history
        score = metric.compute(udl_v2)
        assert 0.0 <= score <= 1.0

    def test_empty_udl(self):
        """Test evolution tracking with empty UDL."""
        udl = UDLRepresentation("", "empty.udl")
        metric = EvolutionTrackingMetric()

        score = metric.compute(udl)
        assert 0.0 <= score <= 1.0

    def test_properties(self):
        """Test metric properties."""
        metric = EvolutionTrackingMetric()
        properties = metric.get_properties()

        assert properties["bounded"] is True
        assert properties["continuous"] is True

    def test_formula(self):
        """Test formula string."""
        metric = EvolutionTrackingMetric()
        formula = metric.get_formula()
        assert "Evolution" in formula
        assert "Stability" in formula


class TestAdvancedMetricsIntegration:
    """Test integration of all advanced metrics."""

    def test_all_metrics_bounded(self):
        """Test that all advanced metrics produce bounded results."""
        udl_text = """
        // Test grammar for all metrics
        program ::= statement*
        statement ::= assignment | if_statement
        assignment ::= identifier '=' expression
        if_statement ::= 'if' '(' expression ')' '{' statement* '}'
        expression ::= term ('+' term)*
        term ::= factor ('*' factor)*
        factor ::= number | identifier | '(' expression ')'
        """
        udl = UDLRepresentation(udl_text, "test.udl")

        metrics = [
            SemanticSimilarityMetric(),
            ReadabilityMetric(),
            MaintainabilityMetric(),
            CrossLanguageCompatibilityMetric(),
            EvolutionTrackingMetric(),
        ]

        for metric in metrics:
            score = metric.compute(udl)
            assert 0.0 <= score <= 1.0, (
                f"{metric.__class__.__name__} produced out-of-bounds score: {score}"
            )

    def test_all_metrics_deterministic(self):
        """Test that all advanced metrics are deterministic."""
        udl_text = """
        expression ::= term ('+' term)*
        term ::= factor ('*' factor)*
        factor ::= number | identifier
        """
        udl = UDLRepresentation(udl_text, "test.udl")

        metrics = [
            SemanticSimilarityMetric(),
            ReadabilityMetric(),
            MaintainabilityMetric(),
            CrossLanguageCompatibilityMetric(),
            EvolutionTrackingMetric(),
        ]

        for metric in metrics:
            scores = [metric.compute(udl) for _ in range(3)]
            assert all(score == scores[0] for score in scores), (
                f"{metric.__class__.__name__} is not deterministic: {scores}"
            )

    def test_metric_registration(self):
        """Test that all advanced metrics are properly registered."""
        from udl_rating_framework.core.metrics.base import MetricRegistry

        registry = MetricRegistry()
        registered_metrics = registry.list_metrics()

        expected_metrics = [
            "semantic_similarity",
            "readability",
            "maintainability",
            "cross_language_compatibility",
            "evolution_tracking",
        ]

        for metric_name in expected_metrics:
            assert metric_name in registered_metrics, (
                f"Metric {metric_name} not registered"
            )

    def test_all_metrics_have_formulas(self):
        """Test that all advanced metrics provide LaTeX formulas."""
        metrics = [
            SemanticSimilarityMetric(),
            ReadabilityMetric(),
            MaintainabilityMetric(),
            CrossLanguageCompatibilityMetric(),
            EvolutionTrackingMetric(),
        ]

        for metric in metrics:
            formula = metric.get_formula()
            assert isinstance(formula, str)
            assert len(formula) > 0
            # Should contain LaTeX-style notation
            assert any(char in formula for char in ["\\", "_", "^", "{", "}"])

    def test_all_metrics_have_properties(self):
        """Test that all advanced metrics define mathematical properties."""
        metrics = [
            SemanticSimilarityMetric(),
            ReadabilityMetric(),
            MaintainabilityMetric(),
            CrossLanguageCompatibilityMetric(),
            EvolutionTrackingMetric(),
        ]

        expected_properties = ["bounded",
                               "monotonic", "additive", "continuous"]

        for metric in metrics:
            properties = metric.get_properties()
            assert isinstance(properties, dict)

            for prop in expected_properties:
                assert prop in properties
                assert isinstance(properties[prop], bool)
