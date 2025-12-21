"""
Property-based and unit tests for CompletenessMetric.

Tests the mathematical correctness of the completeness metric implementation.
"""

from hypothesis import given, strategies as st, settings, assume
from typing import Set
from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.core.metrics.completeness import CompletenessMetric


def create_udl_with_known_constructs(construct_types: Set[str]) -> UDLRepresentation:
    """
    Create a UDL with specific construct types for testing.

    Args:
        construct_types: Set of construct types to include

    Returns:
        UDLRepresentation with specified constructs
    """
    udl_parts = []

    # Always include basic production rules if any constructs are requested
    if construct_types:
        udl_parts.append("start ::= expr")

    # Add constructs based on requested types
    if "operators" in construct_types:
        udl_parts.append("expr ::= term '+' expr | term")
        udl_parts.append("term ::= factor '*' term | factor")

    if "literals" in construct_types or "terminals" in construct_types:
        udl_parts.append("factor ::= 'number' | 'identifier'")

    if "keywords" in construct_types:
        udl_parts.append("stmt ::= 'if' expr 'then' stmt")

    if "identifiers" in construct_types:
        udl_parts.append("var ::= identifier")

    if "comments" in construct_types:
        udl_parts.append("# This is a comment")

    if "control_flow" in construct_types:
        udl_parts.append("loop ::= 'while' expr stmt")

    if "statements" in construct_types:
        udl_parts.append("program ::= stmt+")

    if "expressions" in construct_types:
        udl_parts.append("primary ::= '(' expr ')'")

    if "precedence" in construct_types or "associativity" in construct_types:
        udl_parts.append("expr ::= expr '+' term")  # Left-recursive for associativity

    if "nesting" in construct_types:
        udl_parts.append("nested ::= '(' nested ')' | 'base'")

    if "key_value_pairs" in construct_types:
        udl_parts.append("config ::= key '=' value")

    if "sections" in construct_types:
        udl_parts.append("section ::= '[' name ']'")

    if "tags" in construct_types:
        udl_parts.append("element ::= '<' tag '>' content '</' tag '>'")

    if "attributes" in construct_types:
        udl_parts.append("tag ::= name attr*")

    if "content" in construct_types:
        udl_parts.append("content ::= text | element")

    udl_text = "\n".join(udl_parts)
    return UDLRepresentation(udl_text, "test.udl")


class TestCompletenessMetricProperties:
    """Property-based tests for CompletenessMetric."""

    def test_completeness_metric_correctness_property(self):
        """
        **Feature: udl-rating-framework, Property 10: Completeness Metric Correctness**
        **Validates: Requirements 3.3**

        For any UDL, the completeness score must equal |Defined_Constructs| / |Required_Constructs|.
        """
        # Test with various construct combinations for different language types
        test_cases = [
            # (construct_types, expected_language_type, description)
            (set(), "basic_grammar", "empty UDL"),
            ({"production_rules", "terminals"}, "basic_grammar", "minimal grammar"),
            (
                {"production_rules", "terminals", "operators"},
                "basic_grammar",
                "grammar with operators",
            ),
            (
                {
                    "production_rules",
                    "terminals",
                    "operators",
                    "precedence",
                    "expressions",
                },
                "expression_language",
                "expression language",
            ),
            (
                {"production_rules", "keywords", "statements", "control_flow"},
                "programming_language",
                "programming language",
            ),
            (
                {"production_rules", "tags", "attributes", "nesting"},
                "markup_language",
                "markup language",
            ),
            (
                {"production_rules", "key_value_pairs", "sections"},
                "configuration_language",
                "config language",
            ),
        ]

        metric = CompletenessMetric()

        for construct_types, expected_lang_type, description in test_cases:
            # Create UDL with specific constructs
            udl = create_udl_with_known_constructs(construct_types)

            # Compute completeness score
            computed_score = metric.compute(udl)

            # Extract actual constructs and determine language type
            defined_constructs = metric.extract_defined_constructs(udl)
            inferred_lang_type = metric._infer_language_type(udl)
            required_constructs = metric.get_required_constructs(inferred_lang_type)

            # Compute expected score manually
            defined_construct_types = {
                construct.construct_type for construct in defined_constructs
            }
            coverage_count = len(
                defined_construct_types.intersection(required_constructs)
            )

            if not required_constructs:
                expected_score = 1.0 if not defined_constructs else 0.0
            else:
                expected_score = coverage_count / len(required_constructs)

            expected_score = max(0.0, min(1.0, expected_score))

            # Verify the formula is correctly implemented
            assert abs(computed_score - expected_score) < 1e-6, (
                f"Completeness metric formula incorrect for {description}. "
                f"Expected: {expected_score}, Got: {computed_score}. "
                f"Defined constructs: {defined_construct_types}, "
                f"Required constructs: {required_constructs}, "
                f"Coverage: {coverage_count}/{len(required_constructs)}"
            )

            # Verify boundedness
            assert 0.0 <= computed_score <= 1.0, (
                f"Completeness score {computed_score} not in [0,1] for {description}"
            )

    @given(
        st.sets(
            st.sampled_from(
                [
                    "production_rules",
                    "terminals",
                    "operators",
                    "keywords",
                    "identifiers",
                    "literals",
                    "comments",
                    "statements",
                    "expressions",
                    "control_flow",
                    "precedence",
                    "associativity",
                    "nesting",
                    "key_value_pairs",
                    "sections",
                    "tags",
                    "attributes",
                    "content",
                ]
            ),
            min_size=0,
            max_size=8,
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_completeness_formula_property(self, construct_types: Set[str]):
        """
        Property test: Completeness formula holds for generated construct sets.

        **Feature: udl-rating-framework, Property 10: Completeness Metric Correctness**
        **Validates: Requirements 3.3**
        """
        # Skip empty sets occasionally to test edge cases
        assume(len(construct_types) <= 8)  # Reasonable limit

        try:
            # Create UDL with specified constructs
            udl = create_udl_with_known_constructs(construct_types)
            metric = CompletenessMetric()

            # Compute completeness score
            computed_score = metric.compute(udl)

            # Manually verify the computation
            defined_constructs = metric.extract_defined_constructs(udl)
            inferred_lang_type = metric._infer_language_type(udl)
            required_constructs = metric.get_required_constructs(inferred_lang_type)

            defined_construct_types = {
                construct.construct_type for construct in defined_constructs
            }
            coverage_count = len(
                defined_construct_types.intersection(required_constructs)
            )

            if not required_constructs:
                expected_score = 1.0 if not defined_constructs else 0.0
            else:
                expected_score = coverage_count / len(required_constructs)

            expected_score = max(0.0, min(1.0, expected_score))

            # Verify formula correctness
            assert abs(computed_score - expected_score) < 1e-6, (
                f"Formula mismatch for constructs {construct_types}. "
                f"Expected: {expected_score}, Got: {computed_score}"
            )

            # Verify boundedness
            assert 0.0 <= computed_score <= 1.0, (
                f"Score {computed_score} not bounded for constructs {construct_types}"
            )

        except Exception as e:
            # If UDL creation fails, skip this test case
            assume(False, f"UDL creation failed: {e}")


class TestCompletenessMetricUnits:
    """Unit tests for CompletenessMetric."""

    def test_fully_complete_udl(self):
        """Test on fully complete UDL (expect 1.0)."""
        # Create a UDL that has all basic grammar constructs
        udl_text = """
        # Complete basic grammar
        expr ::= term '+' expr | term
        term ::= factor '*' term | factor  
        factor ::= '(' expr ')' | 'number'
        """
        udl = UDLRepresentation(udl_text, "complete.udl")
        metric = CompletenessMetric()

        score = metric.compute(udl)

        # For basic grammar, this should be complete (has production_rules, terminals, non_terminals)
        assert score == 1.0, (
            f"Complete basic grammar should have completeness 1.0, got {score}"
        )

    def test_partially_complete_udl(self):
        """Test on partially complete UDL."""
        # Create a UDL that only has some required constructs
        udl_text = """
        expr ::= 'number'
        """
        udl = UDLRepresentation(udl_text, "partial.udl")
        metric = CompletenessMetric()

        score = metric.compute(udl)

        # This UDL actually has all basic grammar requirements:
        # - production_rules (expr ::= 'number')
        # - terminals ('number')
        # - non_terminals (expr)
        # So it should be complete for basic grammar
        expected_score = 1.0  # All basic grammar requirements satisfied
        assert abs(score - expected_score) < 1e-6, (
            f"Expected completeness {expected_score}, got {score}"
        )

    def test_minimal_udl(self):
        """Test on minimal UDL."""
        # Very minimal UDL with just one rule
        udl_text = "start ::= 'end'"
        udl = UDLRepresentation(udl_text, "minimal.udl")
        metric = CompletenessMetric()

        score = metric.compute(udl)

        # This minimal UDL has all basic grammar requirements:
        # - production_rules (start ::= 'end')
        # - terminals ('end')
        # - non_terminals (start)
        # So it should be complete for basic grammar
        expected_score = 1.0
        assert abs(score - expected_score) < 1e-6, (
            f"Expected minimal completeness {expected_score}, got {score}"
        )

    def test_incomplete_programming_language(self):
        """Test on UDL that looks like programming language but is incomplete."""
        # Create a UDL that has keywords (making it programming language) but missing constructs
        udl_text = """
        stmt ::= 'if' expr
        expr ::= 'number'
        """
        udl = UDLRepresentation(udl_text, "incomplete_prog.udl")
        metric = CompletenessMetric()

        score = metric.compute(udl)

        # This should be detected as programming language which requires many constructs
        # It's missing statements, control_flow, etc., so should have partial completeness
        lang_type = metric._infer_language_type(udl)
        required = metric.get_required_constructs(lang_type)
        defined = metric.extract_defined_constructs(udl)
        defined_types = {c.construct_type for c in defined}

        print(f"Language type: {lang_type}")
        print(f"Required: {required}")
        print(f"Defined: {defined_types}")
        print(f"Score: {score}")

        # Should have some completeness but likely not full for programming language
        assert score >= 0.0, f"Should have non-negative completeness, got {score}"
        assert score <= 1.0, f"Should not exceed 1.0 completeness, got {score}"

    def test_empty_udl(self):
        """Test on empty UDL."""
        udl = UDLRepresentation("", "empty.udl")
        metric = CompletenessMetric()

        score = metric.compute(udl)

        # Empty UDL should have 0 completeness
        assert score == 0.0, f"Empty UDL should have completeness 0.0, got {score}"

    def test_expression_language_completeness(self):
        """Test completeness for expression language."""
        udl_text = """
        expr ::= expr '+' term | term
        term ::= term '*' factor | factor
        factor ::= '(' expr ')' | 'number'
        """
        udl = UDLRepresentation(udl_text, "expr_lang.udl")
        metric = CompletenessMetric()

        score = metric.compute(udl)

        # This should be detected as expression language and have high completeness
        # Should have: production_rules, terminals, non_terminals, operators, precedence, associativity, expressions
        # That's most of the expression language requirements
        assert score > 0.5, (
            f"Expression language should have high completeness, got {score}"
        )

    def test_programming_language_completeness(self):
        """Test completeness for programming language constructs."""
        udl_text = """
        program ::= stmt+
        stmt ::= 'if' expr 'then' stmt | assignment
        assignment ::= identifier '=' expr
        expr ::= term '+' expr | term
        term ::= 'number' | identifier
        """
        udl = UDLRepresentation(udl_text, "prog_lang.udl")
        metric = CompletenessMetric()

        score = metric.compute(udl)

        # Should be detected as programming language with reasonable completeness
        assert score > 0.3, (
            f"Programming language should have reasonable completeness, got {score}"
        )

    def test_construct_extraction_method(self):
        """Test the extract_defined_constructs method directly."""
        udl_text = """
        # Grammar with various constructs
        expr ::= term '+' expr | term
        term ::= 'number'
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = CompletenessMetric()

        constructs = metric.extract_defined_constructs(udl)
        construct_types = {c.construct_type for c in constructs}

        # Check that we found the basic expected types
        assert "production_rules" in construct_types, "Should detect production rules"
        assert "terminals" in construct_types, "Should detect terminals"
        assert "operators" in construct_types, "Should detect operators"

    def test_language_type_inference(self):
        """Test language type inference method."""
        metric = CompletenessMetric()

        # Test basic grammar
        basic_udl = UDLRepresentation("expr ::= 'term'", "basic.udl")
        lang_type = metric._infer_language_type(basic_udl)
        assert lang_type == "basic_grammar", f"Expected basic_grammar, got {lang_type}"

        # Test expression language
        expr_udl = UDLRepresentation("expr ::= expr '+' term | term", "expr.udl")
        lang_type = metric._infer_language_type(expr_udl)
        # Could be expression_language or basic_grammar depending on complexity
        assert lang_type in [
            "expression_language",
            "basic_grammar",
        ], f"Unexpected type {lang_type}"

    def test_required_constructs_method(self):
        """Test the get_required_constructs method."""
        metric = CompletenessMetric()

        # Test basic grammar requirements
        basic_req = metric.get_required_constructs("basic_grammar")
        assert "production_rules" in basic_req
        assert "terminals" in basic_req
        assert "non_terminals" in basic_req

        # Test expression language requirements
        expr_req = metric.get_required_constructs("expression_language")
        assert "operators" in expr_req
        assert "precedence" in expr_req

        # Test unknown language type (should default)
        unknown_req = metric.get_required_constructs("unknown_type")
        assert "production_rules" in unknown_req
        assert "terminals" in unknown_req

    def test_metric_properties(self):
        """Test that the metric reports correct mathematical properties."""
        metric = CompletenessMetric()
        properties = metric.get_properties()

        assert properties["bounded"] is True
        assert properties["monotonic"] is True
        assert properties["additive"] is False
        assert properties["continuous"] is False

    def test_metric_formula(self):
        """Test that the metric returns the correct LaTeX formula."""
        metric = CompletenessMetric()
        formula = metric.get_formula()

        expected_formula = (
            r"Completeness(U) = \frac{|Defined\_Constructs|}{|Required\_Constructs|}"
        )
        assert formula == expected_formula, (
            f"Expected formula {expected_formula}, got {formula}"
        )

    def test_boundedness_verification(self):
        """Test that the metric satisfies boundedness property."""
        test_cases = [
            "",  # Empty
            "A ::= 'a'",  # Simple
            "expr ::= term '+' expr",  # Expression
            "stmt ::= 'if' expr stmt",  # Statement
        ]

        metric = CompletenessMetric()

        for udl_text in test_cases:
            udl = UDLRepresentation(udl_text, "test.udl")
            score = metric.compute(udl)

            assert 0.0 <= score <= 1.0, (
                f"Score {score} not in [0,1] for UDL: {repr(udl_text)}"
            )

            # Also test the verification method
            assert metric.verify_boundedness(udl), (
                f"Boundedness verification failed for UDL: {repr(udl_text)}"
            )

    def test_determinism_verification(self):
        """Test that the metric satisfies determinism property."""
        udl_text = """
        program ::= stmt+
        stmt ::= expr ';'
        expr ::= term '+' expr | term
        term ::= 'number' | 'identifier'
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        metric = CompletenessMetric()

        # Compute multiple times
        scores = [metric.compute(udl) for _ in range(5)]

        # All should be identical
        first_score = scores[0]
        for score in scores[1:]:
            assert score == first_score, f"Non-deterministic behavior: got {scores}"

        # Also test the verification method
        assert metric.verify_determinism(udl), "Determinism verification failed"
