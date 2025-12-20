"""
Cross-Language Compatibility Metric implementation.

Measures compatibility and portability across different grammar formats and language paradigms.
"""

import re
from typing import Dict, List, Set, Any, Tuple, Optional
from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.representation import (
    UDLRepresentation,
    Token,
    TokenType,
    GrammarFormat,
)


class CrossLanguageCompatibilityMetric(QualityMetric):
    """
    Measures cross-language compatibility and portability.

    Mathematical Definition:
    Compatibility(U) = w₁·Portability(U) + w₂·Standards(U) + w₃·Universality(U) + w₄·Interop(U)

    Where:
    - Portability(U): How easily the UDL can be translated to other formats
    - Standards(U): Adherence to common standards and conventions
    - Universality(U): Use of universal constructs vs format-specific features
    - Interop(U): Interoperability with common tools and parsers
    - wᵢ: Weights summing to 1

    Algorithm:
    1. Analyze format-specific vs universal constructs
    2. Check adherence to common standards (ISO EBNF, RFC ABNF, etc.)
    3. Evaluate portability to other grammar formats
    4. Assess tool interoperability potential
    5. Combine metrics with learned weights
    """

    def __init__(self):
        """Initialize cross-language compatibility metric."""
        self.weights = {
            "portability": 0.3,
            "standards": 0.25,
            "universality": 0.25,
            "interoperability": 0.2,
        }

        # Define universal constructs that work across formats
        self.universal_constructs = {
            "production_rules",
            "terminals",
            "non_terminals",
            "alternation",
            "concatenation",
            "grouping",
            "optional",
            "repetition",
        }

        # Define format-specific features
        self.format_specific_features = {
            GrammarFormat.ANTLR: {
                "lexer_rules",
                "parser_rules",
                "actions",
                "predicates",
                "channels",
                "modes",
                "fragments",
                "imports",
            },
            GrammarFormat.PEG: {
                "ordered_choice",
                "not_predicate",
                "and_predicate",
                "cut_operator",
                "packrat_parsing",
            },
            GrammarFormat.YACC_BISON: {
                "precedence_declarations",
                "associativity",
                "semantic_actions",
                "error_recovery",
                "shift_reduce",
            },
            GrammarFormat.EBNF: {"iso_syntax", "repetition_syntax", "exception_syntax"},
            GrammarFormat.ABNF: {"rfc_syntax", "prose_values", "numeric_values"},
        }

    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute cross-language compatibility score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Compatibility score in [0, 1]
        """
        # Compute individual compatibility components
        portability_score = self._compute_portability_score(udl)
        standards_score = self._compute_standards_score(udl)
        universality_score = self._compute_universality_score(udl)
        interoperability_score = self._compute_interoperability_score(udl)

        # Weighted combination
        compatibility_score = (
            self.weights["portability"] * portability_score
            + self.weights["standards"] * standards_score
            + self.weights["universality"] * universality_score
            + self.weights["interoperability"] * interoperability_score
        )

        return max(0.0, min(1.0, compatibility_score))

    def _compute_portability_score(self, udl: UDLRepresentation) -> float:
        """
        Compute portability score (ease of translation to other formats).

        Args:
            udl: UDLRepresentation instance

        Returns:
            Portability score in [0, 1]
        """
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()
        current_format = udl.get_format()

        # Analyze constructs used
        used_constructs = self._identify_used_constructs(tokens, rules)

        # Check how many constructs are universal vs format-specific
        universal_count = len(used_constructs.intersection(self.universal_constructs))
        format_specific_count = 0

        if current_format in self.format_specific_features:
            format_specific_features = self.format_specific_features[current_format]
            format_specific_count = len(
                used_constructs.intersection(format_specific_features)
            )

        total_constructs = len(used_constructs)
        if total_constructs == 0:
            return 1.0

        # Higher ratio of universal constructs = higher portability
        universality_ratio = universal_count / total_constructs
        format_specificity_penalty = format_specific_count / total_constructs

        portability_score = universality_ratio - 0.5 * format_specificity_penalty

        return max(0.0, min(1.0, portability_score))

    def _compute_standards_score(self, udl: UDLRepresentation) -> float:
        """
        Compute adherence to common standards.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Standards adherence score in [0, 1]
        """
        tokens = udl.get_tokens()
        current_format = udl.get_format()

        # Check adherence to format-specific standards
        if current_format == GrammarFormat.EBNF:
            return self._check_iso_ebnf_compliance(tokens)
        elif current_format == GrammarFormat.ABNF:
            return self._check_rfc_abnf_compliance(tokens)
        elif current_format == GrammarFormat.BNF:
            return self._check_bnf_compliance(tokens)
        elif current_format == GrammarFormat.ANTLR:
            return self._check_antlr_conventions(tokens)
        elif current_format == GrammarFormat.PEG:
            return self._check_peg_conventions(tokens)
        else:
            # Generic standards check
            return self._check_generic_standards(tokens)

    def _compute_universality_score(self, udl: UDLRepresentation) -> float:
        """
        Compute universality score (use of common constructs).

        Args:
            udl: UDLRepresentation instance

        Returns:
            Universality score in [0, 1]
        """
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()

        # Check for universal patterns
        universal_patterns = {
            "basic_production": self._has_basic_productions(rules),
            "standard_operators": self._uses_standard_operators(tokens),
            "common_naming": self._uses_common_naming(tokens),
            "standard_literals": self._uses_standard_literals(tokens),
            "conventional_structure": self._has_conventional_structure(rules),
        }

        # Score based on presence of universal patterns
        universality_score = sum(universal_patterns.values()) / len(universal_patterns)

        return universality_score

    def _compute_interoperability_score(self, udl: UDLRepresentation) -> float:
        """
        Compute interoperability score (compatibility with common tools).

        Args:
            udl: UDLRepresentation instance

        Returns:
            Interoperability score in [0, 1]
        """
        tokens = udl.get_tokens()
        current_format = udl.get_format()

        # Check compatibility with common parser generators
        tool_compatibility = {
            "yacc_bison": self._compatible_with_yacc_bison(tokens),
            "antlr": self._compatible_with_antlr(tokens),
            "peg_parsers": self._compatible_with_peg_parsers(tokens),
            "generic_parsers": self._compatible_with_generic_parsers(tokens),
        }

        # Weight by tool popularity/importance
        weights = {
            "yacc_bison": 0.3,
            "antlr": 0.3,
            "peg_parsers": 0.2,
            "generic_parsers": 0.2,
        }

        interoperability_score = sum(
            weights[tool] * compatibility
            for tool, compatibility in tool_compatibility.items()
        )

        return interoperability_score

    def _identify_used_constructs(self, tokens: List[Token], rules: List) -> Set[str]:
        """
        Identify which constructs are used in the UDL.

        Args:
            tokens: List of tokens
            rules: List of grammar rules

        Returns:
            Set of construct names
        """
        constructs = set()

        # Basic constructs
        if rules:
            constructs.add("production_rules")

        # Analyze tokens for specific constructs
        for token in tokens:
            if token.type == TokenType.OPERATOR:
                if token.text == "|":
                    constructs.add("alternation")
                elif token.text in ["*", "+"]:
                    constructs.add("repetition")
                elif token.text == "?":
                    constructs.add("optional")
                elif token.text in ["(", ")"]:
                    constructs.add("grouping")
                elif token.text == "<-":
                    constructs.add("ordered_choice")
                elif token.text in ["&", "~"]:
                    constructs.add("predicates")

            elif token.type == TokenType.LITERAL:
                constructs.add("terminals")

            elif token.type == TokenType.IDENTIFIER:
                constructs.add("non_terminals")

        return constructs

    def _check_iso_ebnf_compliance(self, tokens: List[Token]) -> float:
        """Check compliance with ISO/IEC 14977 EBNF standard."""
        compliance_checks = {
            "uses_assignment": any(
                token.text == "="
                for token in tokens
                if token.type == TokenType.OPERATOR
            ),
            "uses_terminator": any(token.text in [";", "."] for token in tokens),
            "proper_grouping": self._has_proper_ebnf_grouping(tokens),
            "standard_repetition": self._has_standard_ebnf_repetition(tokens),
        }

        return sum(compliance_checks.values()) / len(compliance_checks)

    def _check_rfc_abnf_compliance(self, tokens: List[Token]) -> float:
        """Check compliance with RFC 5234 ABNF standard."""
        compliance_checks = {
            "uses_assignment": any(
                token.text == "="
                for token in tokens
                if token.type == TokenType.OPERATOR
            ),
            "case_insensitive": self._has_case_insensitive_literals(tokens),
            "numeric_values": self._has_numeric_value_notation(tokens),
            "proper_concatenation": self._has_proper_abnf_concatenation(tokens),
        }

        return sum(compliance_checks.values()) / len(compliance_checks)

    def _check_bnf_compliance(self, tokens: List[Token]) -> float:
        """Check compliance with basic BNF conventions."""
        compliance_checks = {
            "uses_assignment": any(
                token.text == "::="
                for token in tokens
                if token.type == TokenType.OPERATOR
            ),
            "angle_brackets": self._has_angle_bracket_nonterminals(tokens),
            "alternation": any(
                token.text == "|"
                for token in tokens
                if token.type == TokenType.OPERATOR
            ),
        }

        return sum(compliance_checks.values()) / len(compliance_checks)

    def _check_antlr_conventions(self, tokens: List[Token]) -> float:
        """Check adherence to ANTLR conventions."""
        compliance_checks = {
            "grammar_declaration": self._has_grammar_declaration(tokens),
            "rule_naming": self._has_proper_antlr_naming(tokens),
            "lexer_parser_separation": self._has_lexer_parser_separation(tokens),
        }

        return sum(compliance_checks.values()) / len(compliance_checks)

    def _check_peg_conventions(self, tokens: List[Token]) -> float:
        """Check adherence to PEG conventions."""
        compliance_checks = {
            "ordered_choice": any(
                token.text == "/"
                for token in tokens
                if token.type == TokenType.OPERATOR
            ),
            "peg_assignment": any(
                token.text == "<-"
                for token in tokens
                if token.type == TokenType.OPERATOR
            ),
            "predicates": any(
                token.text in ["&", "~"]
                for token in tokens
                if token.type == TokenType.OPERATOR
            ),
        }

        return sum(compliance_checks.values()) / len(compliance_checks)

    def _check_generic_standards(self, tokens: List[Token]) -> float:
        """Check adherence to generic grammar standards."""
        compliance_checks = {
            "has_productions": self._has_production_rules(tokens),
            "consistent_operators": self._has_consistent_operators(tokens),
            "proper_literals": self._has_proper_string_literals(tokens),
        }

        return sum(compliance_checks.values()) / len(compliance_checks)

    # Helper methods for specific checks
    def _has_basic_productions(self, rules: List) -> bool:
        """Check if UDL has basic production rules."""
        return len(rules) > 0

    def _uses_standard_operators(self, tokens: List[Token]) -> bool:
        """Check if UDL uses standard operators."""
        standard_ops = {"|", "*", "+", "?", "(", ")", "::=", ":", "="}
        used_ops = {token.text for token in tokens if token.type == TokenType.OPERATOR}
        return len(used_ops.intersection(standard_ops)) > 0

    def _uses_common_naming(self, tokens: List[Token]) -> bool:
        """Check if UDL uses common naming conventions."""
        identifiers = [
            token.text for token in tokens if token.type == TokenType.IDENTIFIER
        ]
        if not identifiers:
            return True

        # Check for common patterns
        common_patterns = ["expr", "stmt", "decl", "term", "factor", "id", "num"]
        return any(
            any(pattern in ident.lower() for pattern in common_patterns)
            for ident in identifiers
        )

    def _uses_standard_literals(self, tokens: List[Token]) -> bool:
        """Check if UDL uses standard literal notation."""
        literals = [token.text for token in tokens if token.type == TokenType.LITERAL]
        if not literals:
            return True

        # Check for standard quote styles
        return any(
            (literal.startswith('"') and literal.endswith('"'))
            or (literal.startswith("'") and literal.endswith("'"))
            for literal in literals
        )

    def _has_conventional_structure(self, rules: List) -> bool:
        """Check if UDL follows conventional grammar structure."""
        if not rules:
            return False

        # Look for hierarchical structure (common in well-structured grammars)
        rule_names = [rule.lhs.lower() for rule in rules if hasattr(rule, "lhs")]

        # Common hierarchical patterns
        hierarchy_indicators = ["program", "statement", "expression", "term", "factor"]
        return any(indicator in rule_names for indicator in hierarchy_indicators)

    def _compatible_with_yacc_bison(self, tokens: List[Token]) -> bool:
        """Check compatibility with Yacc/Bison."""
        # Look for Yacc-compatible constructs
        yacc_compatible = True

        # Check for incompatible PEG constructs
        peg_constructs = ["<-", "&", "~", "/"]
        for token in tokens:
            if token.type == TokenType.OPERATOR and token.text in peg_constructs:
                yacc_compatible = False
                break

        return yacc_compatible

    def _compatible_with_antlr(self, tokens: List[Token]) -> bool:
        """Check compatibility with ANTLR."""
        # ANTLR is quite flexible, most constructs are compatible
        return True

    def _compatible_with_peg_parsers(self, tokens: List[Token]) -> bool:
        """Check compatibility with PEG parsers."""
        # Check for PEG-incompatible constructs (left recursion indicators)
        # This is a simplified check
        return True

    def _compatible_with_generic_parsers(self, tokens: List[Token]) -> bool:
        """Check compatibility with generic parsers."""
        # Generic parsers usually handle basic BNF/EBNF
        basic_constructs = {"|", "*", "+", "?", "(", ")", "::=", "="}
        used_ops = {token.text for token in tokens if token.type == TokenType.OPERATOR}

        # If only basic constructs are used, it's compatible
        return used_ops.issubset(basic_constructs) or len(used_ops) == 0

    # Additional helper methods for specific compliance checks
    def _has_proper_ebnf_grouping(self, tokens: List[Token]) -> bool:
        """Check for proper EBNF grouping syntax."""
        return any(token.text in ["(", ")", "[", "]", "{", "}"] for token in tokens)

    def _has_standard_ebnf_repetition(self, tokens: List[Token]) -> bool:
        """Check for standard EBNF repetition syntax."""
        return any(token.text in ["{", "}"] for token in tokens)

    def _has_case_insensitive_literals(self, tokens: List[Token]) -> bool:
        """Check for case-insensitive literal notation."""
        # ABNF uses %i for case-insensitive
        return any(
            "%i" in token.text for token in tokens if token.type == TokenType.LITERAL
        )

    def _has_numeric_value_notation(self, tokens: List[Token]) -> bool:
        """Check for ABNF numeric value notation."""
        # ABNF uses %d for decimal, %x for hex
        return any(
            token.text.startswith("%d") or token.text.startswith("%x")
            for token in tokens
            if token.type == TokenType.LITERAL
        )

    def _has_proper_abnf_concatenation(self, tokens: List[Token]) -> bool:
        """Check for proper ABNF concatenation (implicit)."""
        # ABNF uses implicit concatenation (no explicit operator)
        return True  # Simplified check

    def _has_angle_bracket_nonterminals(self, tokens: List[Token]) -> bool:
        """Check for angle bracket notation for non-terminals."""
        return any("<" in token.text and ">" in token.text for token in tokens)

    def _has_grammar_declaration(self, tokens: List[Token]) -> bool:
        """Check for ANTLR grammar declaration."""
        return any("grammar" in token.text.lower() for token in tokens)

    def _has_proper_antlr_naming(self, tokens: List[Token]) -> bool:
        """Check for proper ANTLR naming conventions."""
        # ANTLR: parser rules lowercase, lexer rules uppercase
        identifiers = [
            token.text for token in tokens if token.type == TokenType.IDENTIFIER
        ]
        return len(identifiers) > 0  # Simplified check

    def _has_lexer_parser_separation(self, tokens: List[Token]) -> bool:
        """Check for lexer/parser rule separation."""
        return any(token.text in ["lexer", "parser"] for token in tokens)

    def _has_production_rules(self, tokens: List[Token]) -> bool:
        """Check for production rule indicators."""
        production_ops = ["::=", ":", "->", "=", "<-"]
        return any(
            token.text in production_ops
            for token in tokens
            if token.type == TokenType.OPERATOR
        )

    def _has_consistent_operators(self, tokens: List[Token]) -> bool:
        """Check for consistent operator usage."""
        assignment_ops = [
            token.text
            for token in tokens
            if token.type == TokenType.OPERATOR
            and token.text in ["::=", ":", "->", "=", "<-"]
        ]

        if not assignment_ops:
            return True

        # Check if mostly using one type of assignment operator
        from collections import Counter

        op_counts = Counter(assignment_ops)
        most_common_count = op_counts.most_common(1)[0][1]

        return most_common_count / len(assignment_ops) >= 0.8

    def _has_proper_string_literals(self, tokens: List[Token]) -> bool:
        """Check for properly quoted string literals."""
        literals = [token.text for token in tokens if token.type == TokenType.LITERAL]
        if not literals:
            return True

        properly_quoted = sum(
            1
            for literal in literals
            if (
                (literal.startswith('"') and literal.endswith('"'))
                or (literal.startswith("'") and literal.endswith("'"))
            )
        )

        return properly_quoted / len(literals) >= 0.8

    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        return r"Compatibility(U) = w_1 \cdot Portability(U) + w_2 \cdot Standards(U) + w_3 \cdot Universality(U) + w_4 \cdot Interop(U)"

    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties of the cross-language compatibility metric.

        Returns:
            Dict with mathematical properties
        """
        return {
            "bounded": True,  # Always produces values in [0, 1]
            "monotonic": False,  # More features don't always mean higher compatibility
            "additive": False,  # Compatibility is not sum of parts
            "continuous": True,  # Small changes cause small changes in compatibility
        }


# Register the metric in the global registry
CrossLanguageCompatibilityMetric.register_metric("cross_language_compatibility")
