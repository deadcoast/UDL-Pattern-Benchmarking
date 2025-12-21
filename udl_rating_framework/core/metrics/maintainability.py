"""
Maintainability Index Metric implementation.

Measures maintainability of UDL using software engineering metrics adapted for grammars.
"""

import math
from typing import Dict, List

from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.representation import Token, TokenType, UDLRepresentation


class MaintainabilityMetric(QualityMetric):
    """
    Measures UDL maintainability using adapted software engineering metrics.

    Mathematical Definition:
    Maintainability(U) = max(0, (171 - 5.2*ln(HV) - 0.23*CC - 16.2*ln(LOC) + 50*sin(√(2.4*CM))) / 171)

    Where:
    - HV: Halstead Volume (complexity measure)
    - CC: Cyclomatic Complexity (control flow complexity)
    - LOC: Lines of Code (size measure)
    - CM: Comment Ratio (documentation measure)

    This is adapted from the Microsoft Maintainability Index for code.

    Algorithm:
    1. Compute Halstead metrics from token analysis
    2. Calculate cyclomatic complexity from grammar structure
    3. Count effective lines of code
    4. Measure comment density
    5. Apply maintainability index formula
    """

    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute maintainability index.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Maintainability score in [0, 1]
        """
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()

        if not tokens and not rules:
            return 0.0

        # Compute component metrics
        halstead_volume = self._compute_halstead_volume(tokens)
        cyclomatic_complexity = self._compute_cyclomatic_complexity(
            tokens, rules)
        lines_of_code = self._compute_lines_of_code(tokens)
        comment_ratio = self._compute_comment_ratio(tokens)

        # Apply maintainability index formula (adapted)
        # Original: 171 - 5.2*ln(HV) - 0.23*CC - 16.2*ln(LOC) + 50*sin(√(2.4*CM))
        # Handle edge cases with safe logarithms
        safe_hv = max(1.0, halstead_volume)
        safe_loc = max(1.0, lines_of_code)
        safe_cm = max(0.01, comment_ratio)

        maintainability_raw = (
            171
            - 5.2 * math.log(safe_hv)
            - 0.23 * cyclomatic_complexity
            - 16.2 * math.log(safe_loc)
            + 50 * math.sin(math.sqrt(2.4 * safe_cm))
        )

        # Normalize to [0, 1] (original scale is roughly 0-171)
        maintainability_normalized = max(0.0, maintainability_raw) / 171.0

        return max(0.0, min(1.0, maintainability_normalized))

    def _compute_halstead_volume(self, tokens: List[Token]) -> float:
        """
        Compute Halstead Volume: V = N * log₂(n)

        Where:
        - N: Total number of operators and operands
        - n: Number of distinct operators and operands

        Args:
            tokens: List of tokens

        Returns:
            Halstead volume
        """
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0

        for token in tokens:
            if token.type == TokenType.OPERATOR:
                operators.add(token.text)
                operator_count += 1
            elif token.type in [
                TokenType.IDENTIFIER,
                TokenType.LITERAL,
                TokenType.KEYWORD,
            ]:
                operands.add(token.text)
                operand_count += 1

        # Halstead metrics
        n1 = len(operators)  # Distinct operators
        n2 = len(operands)  # Distinct operands
        N1 = operator_count  # Total operators
        N2 = operand_count  # Total operands

        n = n1 + n2  # Total distinct tokens
        N = N1 + N2  # Total tokens

        if n <= 0:
            return 1.0

        # Halstead Volume: V = N * log₂(n)
        volume = N * math.log2(n)
        return max(1.0, volume)

    def _compute_cyclomatic_complexity(self, tokens: List[Token], rules: List) -> float:
        """
        Compute cyclomatic complexity adapted for grammars.

        For grammars, complexity comes from:
        - Alternative productions (|)
        - Optional constructs (?, *)
        - Repetition constructs (+, *)
        - Nested structures

        Args:
            tokens: List of tokens
            rules: List of grammar rules

        Returns:
            Cyclomatic complexity
        """
        complexity = 1  # Base complexity

        # Count decision points in tokens
        decision_operators = ["|", "?", "*", "+", "/", "&", "~", "!"]
        for token in tokens:
            if token.type == TokenType.OPERATOR and token.text in decision_operators:
                complexity += 1

        # Count alternative productions in rules
        for rule in rules:
            if hasattr(rule, "rhs"):
                # Count alternatives (assuming | separates alternatives)
                rhs_text = (
                    " ".join(rule.rhs) if isinstance(
                        rule.rhs, list) else str(rule.rhs)
                )
                alternatives = rhs_text.count("|")
                complexity += alternatives

        # Count nesting levels (increases complexity)
        nesting_depth = self._compute_nesting_depth(tokens)
        complexity += nesting_depth

        return float(complexity)

    def _compute_nesting_depth(self, tokens: List[Token]) -> int:
        """
        Compute maximum nesting depth.

        Args:
            tokens: List of tokens

        Returns:
            Maximum nesting depth
        """
        max_depth = 0
        current_depth = 0

        open_brackets = {"(", "[", "{", "<"}
        close_brackets = {")", "]", "}", ">"}

        for token in tokens:
            if token.text in open_brackets:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif token.text in close_brackets:
                current_depth = max(0, current_depth - 1)

        return max_depth

    def _compute_lines_of_code(self, tokens: List[Token]) -> int:
        """
        Compute effective lines of code (non-empty, non-comment lines).

        Args:
            tokens: List of tokens

        Returns:
            Lines of code count
        """
        lines_with_code = set()

        for token in tokens:
            # Skip whitespace, comments, and newlines
            if token.type not in [
                TokenType.WHITESPACE,
                TokenType.COMMENT,
                TokenType.NEWLINE,
                TokenType.EOF,
            ]:
                lines_with_code.add(token.line)

        return len(lines_with_code)

    def _compute_comment_ratio(self, tokens: List[Token]) -> float:
        """
        Compute comment ratio (comment lines / total lines).

        Args:
            tokens: List of tokens

        Returns:
            Comment ratio in [0, 1]
        """
        comment_lines = set()
        total_lines = set()

        for token in tokens:
            if token.type != TokenType.EOF:
                total_lines.add(token.line)
                if token.type == TokenType.COMMENT:
                    comment_lines.add(token.line)

        if not total_lines:
            return 0.0

        return len(comment_lines) / len(total_lines)

    def get_detailed_metrics(self, udl: UDLRepresentation) -> Dict[str, float]:
        """
        Get detailed breakdown of maintainability metrics.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Dict with detailed metrics
        """
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()

        return {
            "halstead_volume": self._compute_halstead_volume(tokens),
            "cyclomatic_complexity": self._compute_cyclomatic_complexity(tokens, rules),
            "lines_of_code": float(self._compute_lines_of_code(tokens)),
            "comment_ratio": self._compute_comment_ratio(tokens),
            "maintainability_index": self.compute(udl),
        }

    def _compute_halstead_metrics(self, tokens: List[Token]) -> Dict[str, float]:
        """
        Compute detailed Halstead metrics.

        Args:
            tokens: List of tokens

        Returns:
            Dict with Halstead metrics
        """
        operators = {}
        operands = {}

        for token in tokens:
            if token.type == TokenType.OPERATOR:
                operators[token.text] = operators.get(token.text, 0) + 1
            elif token.type in [
                TokenType.IDENTIFIER,
                TokenType.LITERAL,
                TokenType.KEYWORD,
            ]:
                operands[token.text] = operands.get(token.text, 0) + 1

        n1 = len(operators)  # Distinct operators
        n2 = len(operands)  # Distinct operands
        N1 = sum(operators.values())  # Total operators
        N2 = sum(operands.values())  # Total operands

        n = n1 + n2  # Vocabulary
        N = N1 + N2  # Length

        if n <= 0:
            return {
                "vocabulary": 0,
                "length": 0,
                "volume": 0,
                "difficulty": 0,
                "effort": 0,
            }

        volume = N * math.log2(n)

        # Difficulty: D = (n1/2) * (N2/n2)
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0

        # Effort: E = D * V
        effort = difficulty * volume

        return {
            "vocabulary": n,
            "length": N,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort,
        }

    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        return r"MI(U) = \max(0, \frac{171 - 5.2\ln(HV) - 0.23CC - 16.2\ln(LOC) + 50\sin(\sqrt{2.4CM})}{171})"

    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties of the maintainability metric.

        Returns:
            Dict with mathematical properties
        """
        return {
            "bounded": True,  # Always produces values in [0, 1]
            "monotonic": False,  # More code doesn't always mean higher maintainability
            "additive": False,  # Maintainability is not sum of parts
            "continuous": True,  # Small changes cause small changes in maintainability
        }


# Register the metric in the global registry
MaintainabilityMetric.register_metric("maintainability")
