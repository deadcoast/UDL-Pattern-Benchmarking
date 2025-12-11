"""
Completeness Metric implementation.

Measures coverage of language constructs using set-theoretic formulations.
"""

from typing import Set, Dict, Any, List
from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.representation import UDLRepresentation, Token, TokenType


class Construct:
    """Represents a language construct that can be defined or required."""

    def __init__(self, name: str, construct_type: str, metadata: Dict[str, Any] = None):
        self.name = name
        self.construct_type = (
            construct_type  # e.g., 'production_rule', 'terminal', 'operator'
        )
        self.metadata = metadata or {}

    def __eq__(self, other):
        if not isinstance(other, Construct):
            return False
        return self.name == other.name and self.construct_type == other.construct_type

    def __hash__(self):
        return hash((self.name, self.construct_type))

    def __repr__(self):
        return f"Construct({self.name}, {self.construct_type})"


class CompletenessMetric(QualityMetric):
    """
    Measures construct coverage using set theory.

    Mathematical Definition:
    Completeness(U) = |Defined_Constructs| / |Required_Constructs|

    Where:
    - Defined_Constructs: Set of implemented language features
    - Required_Constructs: Set of expected features for the language class

    Algorithm:
    1. Extract defined constructs from grammar
    2. Determine required constructs based on language type
    3. Compute coverage ratio
    """

    def __init__(self):
        """Initialize completeness metric with language type mappings."""
        # Define required constructs for different language types
        self.language_type_requirements = {
            "basic_grammar": {"production_rules", "terminals", "non_terminals"},
            "expression_language": {
                "production_rules",
                "terminals",
                "non_terminals",
                "operators",
                "precedence",
                "associativity",
            },
            "programming_language": {
                "production_rules",
                "terminals",
                "non_terminals",
                "operators",
                "keywords",
                "identifiers",
                "literals",
                "statements",
                "expressions",
                "control_flow",
            },
            "markup_language": {
                "production_rules",
                "terminals",
                "non_terminals",
                "tags",
                "attributes",
                "content",
                "nesting",
            },
            "configuration_language": {
                "production_rules",
                "terminals",
                "non_terminals",
                "key_value_pairs",
                "sections",
                "comments",
            },
        }

    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute completeness score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Completeness score in [0, 1]
        """
        # Extract defined constructs from the UDL
        defined_constructs = self.extract_defined_constructs(udl)

        # Determine language type and get required constructs
        language_type = self._infer_language_type(udl)
        required_constructs = self.get_required_constructs(language_type)

        # Handle edge case where no constructs are required
        if not required_constructs:
            return 1.0 if not defined_constructs else 0.0

        # Compute coverage ratio
        defined_construct_types = {
            construct.construct_type for construct in defined_constructs
        }
        coverage_count = len(defined_construct_types.intersection(required_constructs))

        # Formula: |Defined_Constructs| / |Required_Constructs|
        completeness_score = coverage_count / len(required_constructs)

        # Ensure result is bounded in [0, 1]
        return max(0.0, min(1.0, completeness_score))

    def extract_defined_constructs(self, udl: UDLRepresentation) -> Set[Construct]:
        """
        Extract D = {c | c is defined in U}.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Set of defined constructs
        """
        constructs = set()

        # Extract constructs from grammar rules
        rules = udl.get_grammar_rules()
        if rules:
            constructs.add(Construct("production_rules", "production_rules"))

            # Extract non-terminals (LHS of rules)
            non_terminals = {rule.lhs for rule in rules}
            if non_terminals:
                constructs.add(Construct("non_terminals", "non_terminals"))

        # Extract constructs from tokens
        tokens = udl.get_tokens()
        token_types_found = set()

        for token in tokens:
            if token.type == TokenType.LITERAL:
                token_types_found.add("terminals")
                token_types_found.add("literals")
            elif token.type == TokenType.OPERATOR:
                token_types_found.add("operators")
            elif token.type == TokenType.KEYWORD:
                token_types_found.add("keywords")
            elif token.type == TokenType.IDENTIFIER:
                token_types_found.add("identifiers")
            elif token.type == TokenType.COMMENT:
                token_types_found.add("comments")

        # Add constructs based on found token types
        for token_type in token_types_found:
            constructs.add(Construct(token_type, token_type))

        # Analyze grammar structure for higher-level constructs
        constructs.update(self._extract_structural_constructs(udl))

        return constructs

    def get_required_constructs(self, language_type: str) -> Set[str]:
        """
        Return R = required constructs for language type.

        Args:
            language_type: Type of language (e.g., 'basic_grammar', 'expression_language')

        Returns:
            Set of required construct types
        """
        return self.language_type_requirements.get(
            language_type, {"production_rules", "terminals"}
        )

    def _infer_language_type(self, udl: UDLRepresentation) -> str:
        """
        Infer the type of language from UDL characteristics.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Inferred language type
        """
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()

        # Count different types of constructs
        has_operators = any(token.type == TokenType.OPERATOR for token in tokens)
        has_keywords = any(token.type == TokenType.KEYWORD for token in tokens)
        has_complex_rules = len(rules) > 5
        has_precedence_patterns = self._has_precedence_patterns(rules)

        # Look for programming language keywords in literals
        programming_keywords = {
            "if",
            "then",
            "else",
            "while",
            "for",
            "function",
            "var",
            "let",
            "const",
        }
        has_prog_keywords = any(
            any(keyword in token.text.lower() for keyword in programming_keywords)
            for token in tokens
            if token.type == TokenType.LITERAL
        )

        # Simple heuristics for language type inference
        if has_prog_keywords or (has_keywords and has_complex_rules):
            return "programming_language"
        elif has_operators and has_precedence_patterns:
            return "expression_language"
        elif any("<" in token.text or ">" in token.text for token in tokens):
            return "markup_language"
        elif any("=" in token.text and ":" not in token.text for token in tokens):
            return "configuration_language"
        else:
            return "basic_grammar"

    def _has_precedence_patterns(self, rules: List) -> bool:
        """
        Check if rules show precedence patterns typical of expression languages.

        Args:
            rules: List of grammar rules

        Returns:
            True if precedence patterns are detected
        """
        # Look for common expression language patterns
        rule_names = [rule.lhs.lower() for rule in rules]

        # Common precedence hierarchy: expression -> term -> factor
        precedence_indicators = ["expr", "term", "factor", "primary"]
        found_indicators = sum(
            1
            for indicator in precedence_indicators
            if any(indicator in name for name in rule_names)
        )

        return found_indicators >= 2

    def _extract_structural_constructs(self, udl: UDLRepresentation) -> Set[Construct]:
        """
        Extract higher-level structural constructs from UDL.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Set of structural constructs
        """
        constructs = set()
        rules = udl.get_grammar_rules()

        # Look for statement patterns
        statement_patterns = ["stmt", "statement", "decl", "declaration"]
        if any(
            any(pattern in rule.lhs.lower() for pattern in statement_patterns)
            for rule in rules
        ):
            constructs.add(Construct("statements", "statements"))

        # Look for expression patterns
        expression_patterns = ["expr", "expression", "term", "factor"]
        if any(
            any(pattern in rule.lhs.lower() for pattern in expression_patterns)
            for rule in rules
        ):
            constructs.add(Construct("expressions", "expressions"))

        # Look for control flow patterns
        control_patterns = ["if", "while", "for", "loop", "branch"]
        tokens = udl.get_tokens()
        if any(
            any(pattern in token.text.lower() for pattern in control_patterns)
            for token in tokens
        ):
            constructs.add(Construct("control_flow", "control_flow"))

        # Look for precedence/associativity (indicated by recursive rules)
        for rule in rules:
            if rule.lhs in rule.rhs:  # Self-recursive rule
                constructs.add(Construct("precedence", "precedence"))
                constructs.add(Construct("associativity", "associativity"))
                break

        # Look for nesting patterns (for markup languages)
        nesting_indicators = ["<", ">", "{", "}", "(", ")"]
        if any(
            any(indicator in token.text for indicator in nesting_indicators)
            for token in tokens
        ):
            constructs.add(Construct("nesting", "nesting"))

        # Look for key-value patterns (for configuration languages)
        if any(
            "=" in token.text and token.type == TokenType.OPERATOR for token in tokens
        ):
            constructs.add(Construct("key_value_pairs", "key_value_pairs"))

        # Look for section patterns
        section_patterns = ["[", "]", "section", "group"]
        if any(
            any(pattern in token.text.lower() for pattern in section_patterns)
            for token in tokens
        ):
            constructs.add(Construct("sections", "sections"))

        # Look for tag patterns (for markup languages)
        if any("<" in token.text and ">" in token.text for token in tokens):
            constructs.add(Construct("tags", "tags"))
            constructs.add(Construct("attributes", "attributes"))
            constructs.add(Construct("content", "content"))

        return constructs

    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        return r"Completeness(U) = \frac{|Defined\_Constructs|}{|Required\_Constructs|}"

    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties of the completeness metric.

        Returns:
            Dict with mathematical properties
        """
        return {
            "bounded": True,  # Always produces values in [0, 1]
            "monotonic": True,  # More defined constructs → higher completeness
            "additive": False,  # Completeness is not sum of parts
            "continuous": False,  # Discrete changes in constructs cause discrete changes
        }


# Register the metric in the global registry
CompletenessMetric.register_metric("completeness")
