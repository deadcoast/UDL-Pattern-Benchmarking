"""
Readability Metric implementation.

Measures readability of UDL syntax using linguistic and structural analysis.
"""

import math
import re
from typing import Any, Dict, List, Set, Tuple

from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.representation import Token, TokenType, UDLRepresentation


class ReadabilityMetric(QualityMetric):
    """
    Measures UDL syntax readability using multiple linguistic metrics.

    Mathematical Definition:
    Readability(U) = w₁·Flesch(U) + w₂·Complexity(U) + w₃·Consistency(U) + w₄·Clarity(U)

    Where:
    - Flesch(U): Adapted Flesch reading ease for code
    - Complexity(U): Syntactic complexity measure
    - Consistency(U): Naming and style consistency
    - Clarity(U): Semantic clarity of constructs
    - wᵢ: Weights summing to 1

    Algorithm:
    1. Analyze token patterns and naming conventions
    2. Compute syntactic complexity metrics
    3. Evaluate consistency in naming and structure
    4. Assess semantic clarity of identifiers
    5. Combine metrics with learned weights
    """

    def __init__(self):
        """Initialize readability metric with default weights."""
        self.weights = {
            "flesch": 0.25,
            "complexity": 0.25,
            "consistency": 0.25,
            "clarity": 0.25,
        }

    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute readability score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Readability score in [0, 1]
        """
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()

        if not tokens and not rules:
            return 0.0

        # Compute individual readability components
        flesch_score = self._compute_flesch_score(tokens)
        complexity_score = self._compute_complexity_score(tokens, rules)
        consistency_score = self._compute_consistency_score(tokens, rules)
        clarity_score = self._compute_clarity_score(tokens)

        # Weighted combination
        readability_score = (
            self.weights["flesch"] * flesch_score
            + self.weights["complexity"] * complexity_score
            + self.weights["consistency"] * consistency_score
            + self.weights["clarity"] * clarity_score
        )

        return max(0.0, min(1.0, readability_score))

    def _compute_flesch_score(self, tokens: List[Token]) -> float:
        """
        Compute adapted Flesch reading ease for UDL syntax.

        Args:
            tokens: List of tokens

        Returns:
            Normalized Flesch score in [0, 1]
        """
        # Extract meaningful tokens (exclude whitespace, delimiters)
        meaningful_tokens = [
            token
            for token in tokens
            if token.type
            in [
                TokenType.IDENTIFIER,
                TokenType.KEYWORD,
                TokenType.LITERAL,
                TokenType.COMMENT,
            ]
        ]

        if not meaningful_tokens:
            return 0.0

        # Count "sentences" (rules or logical units)
        sentences = self._count_logical_units(tokens)
        if sentences == 0:
            return 0.0

        # Count "words" (meaningful tokens)
        words = len(meaningful_tokens)

        # Count "syllables" (character complexity)
        syllables = sum(
            self._count_syllables(token.text) for token in meaningful_tokens
        )

        # Adapted Flesch formula for code
        # Original: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        # Adapted for code readability
        if words == 0:
            return 0.0

        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words

        flesch_raw = (
            206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        )

        # Normalize to [0, 1] (typical Flesch scores range from 0-100)
        flesch_normalized = max(0.0, min(100.0, flesch_raw)) / 100.0

        return flesch_normalized

    def _count_logical_units(self, tokens: List[Token]) -> int:
        """
        Count logical units (sentences equivalent) in UDL.

        Args:
            tokens: List of tokens

        Returns:
            Number of logical units
        """
        # Count production rules, statements, or major constructs
        units = 0

        for token in tokens:
            # Production rule indicators
            if token.type == TokenType.OPERATOR and token.text in [
                "::=",
                ":=",
                "->",
                ":",
                "<-",
            ]:
                units += 1
            # Statement terminators
            elif token.text in [";", ".", "\n\n"]:
                units += 1

        # Minimum of 1 unit
        return max(1, units)

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (adapted for identifiers).

        Args:
            word: Word to analyze

        Returns:
            Estimated syllable count
        """
        # Remove non-alphabetic characters
        word = re.sub(r"[^a-zA-Z]", "", word.lower())

        if not word:
            return 1

        # Simple syllable counting heuristic
        vowels = "aeiouy"
        syllables = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith("e") and syllables > 1:
            syllables -= 1

        return max(1, syllables)

    def _compute_complexity_score(self, tokens: List[Token], rules: List) -> float:
        """
        Compute syntactic complexity score.

        Args:
            tokens: List of tokens
            rules: List of grammar rules

        Returns:
            Complexity score in [0, 1] (higher = less complex = more readable)
        """
        if not tokens:
            return 1.0

        # Measure various complexity factors
        nesting_depth = self._compute_nesting_depth(tokens)
        operator_density = self._compute_operator_density(tokens)
        rule_complexity = self._compute_rule_complexity(rules)
        identifier_complexity = self._compute_identifier_complexity(tokens)

        # Normalize each factor to [0, 1] where 1 = low complexity (high readability)
        max_nesting = 10  # Reasonable maximum
        normalized_nesting = max(0.0, 1.0 - (nesting_depth / max_nesting))

        max_operator_density = 0.5  # 50% operators would be very dense
        normalized_operator_density = max(
            0.0, 1.0 - (operator_density / max_operator_density)
        )

        # Combine complexity factors
        complexity_score = (
            0.3 * normalized_nesting
            + 0.3 * normalized_operator_density
            + 0.2 * rule_complexity
            + 0.2 * identifier_complexity
        )

        return max(0.0, min(1.0, complexity_score))

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

    def _compute_operator_density(self, tokens: List[Token]) -> float:
        """
        Compute operator density (operators / total tokens).

        Args:
            tokens: List of tokens

        Returns:
            Operator density ratio
        """
        if not tokens:
            return 0.0

        operator_count = sum(
            1 for token in tokens if token.type == TokenType.OPERATOR)
        total_meaningful = sum(
            1
            for token in tokens
            if token.type
            not in [TokenType.WHITESPACE, TokenType.NEWLINE, TokenType.EOF]
        )

        if total_meaningful == 0:
            return 0.0

        return operator_count / total_meaningful

    def _compute_rule_complexity(self, rules: List) -> float:
        """
        Compute average rule complexity.

        Args:
            rules: List of grammar rules

        Returns:
            Rule complexity score in [0, 1] (higher = less complex)
        """
        if not rules:
            return 1.0

        total_complexity = 0
        for rule in rules:
            # Rule complexity based on RHS length and structure
            rhs_length = len(rule.rhs) if hasattr(rule, "rhs") else 0
            constraint_count = (
                len(rule.constraints) if hasattr(rule, "constraints") else 0
            )

            # Simple complexity measure
            rule_complexity = rhs_length + constraint_count * 2
            total_complexity += rule_complexity

        avg_complexity = total_complexity / len(rules)

        # Normalize (assume max reasonable complexity is 20)
        max_complexity = 20
        normalized = max(0.0, 1.0 - (avg_complexity / max_complexity))

        return normalized

    def _compute_identifier_complexity(self, tokens: List[Token]) -> float:
        """
        Compute identifier complexity (length, naming patterns).

        Args:
            tokens: List of tokens

        Returns:
            Identifier complexity score in [0, 1] (higher = less complex)
        """
        identifiers = [
            token.text for token in tokens if token.type == TokenType.IDENTIFIER
        ]

        if not identifiers:
            return 1.0

        # Analyze identifier characteristics
        avg_length = sum(len(ident)
                         for ident in identifiers) / len(identifiers)

        # Prefer moderate length identifiers (5-15 characters)
        optimal_length = 10
        length_score = 1.0 - abs(avg_length - optimal_length) / optimal_length
        length_score = max(0.0, min(1.0, length_score))

        return length_score

    def _compute_consistency_score(self, tokens: List[Token], rules: List) -> float:
        """
        Compute naming and style consistency.

        Args:
            tokens: List of tokens
            rules: List of grammar rules

        Returns:
            Consistency score in [0, 1]
        """
        identifiers = [
            token.text for token in tokens if token.type == TokenType.IDENTIFIER
        ]

        if not identifiers:
            return 1.0

        # Analyze naming patterns
        naming_patterns = self._analyze_naming_patterns(identifiers)
        pattern_consistency = self._compute_pattern_consistency(
            naming_patterns)

        # Analyze operator usage consistency
        operators = [
            token.text for token in tokens if token.type == TokenType.OPERATOR]
        operator_consistency = self._compute_operator_consistency(operators)

        # Combine consistency measures
        consistency_score = 0.6 * pattern_consistency + 0.4 * operator_consistency

        return max(0.0, min(1.0, consistency_score))

    def _analyze_naming_patterns(self, identifiers: List[str]) -> Dict[str, int]:
        """
        Analyze naming convention patterns.

        Args:
            identifiers: List of identifier strings

        Returns:
            Dict mapping pattern types to counts
        """
        patterns = {
            "camelCase": 0,
            "snake_case": 0,
            "PascalCase": 0,
            "kebab-case": 0,
            "UPPER_CASE": 0,
            "lowercase": 0,
            "mixed": 0,
        }

        for ident in identifiers:
            if re.match(r"^[a-z][a-zA-Z0-9]*$", ident) and any(
                c.isupper() for c in ident
            ):
                patterns["camelCase"] += 1
            elif re.match(r"^[a-z][a-z0-9_]*$", ident) and "_" in ident:
                patterns["snake_case"] += 1
            elif re.match(r"^[A-Z][a-zA-Z0-9]*$", ident):
                patterns["PascalCase"] += 1
            elif "-" in ident:
                patterns["kebab-case"] += 1
            elif ident.isupper():
                patterns["UPPER_CASE"] += 1
            elif ident.islower():
                patterns["lowercase"] += 1
            else:
                patterns["mixed"] += 1

        return patterns

    def _compute_pattern_consistency(self, patterns: Dict[str, int]) -> float:
        """
        Compute consistency of naming patterns.

        Args:
            patterns: Dict of pattern counts

        Returns:
            Pattern consistency score in [0, 1]
        """
        total_identifiers = sum(patterns.values())
        if total_identifiers == 0:
            return 1.0

        # Find dominant pattern
        max_count = max(patterns.values())
        consistency_ratio = max_count / total_identifiers

        return consistency_ratio

    def _compute_operator_consistency(self, operators: List[str]) -> float:
        """
        Compute consistency of operator usage.

        Args:
            operators: List of operator strings

        Returns:
            Operator consistency score in [0, 1]
        """
        if not operators:
            return 1.0

        # Check for consistent use of assignment operators
        assignment_ops = [op for op in operators if op in [
            "::=", ":=", "->", ":", "="]]

        if not assignment_ops:
            return 1.0

        # Count unique assignment operators
        unique_assignment_ops = set(assignment_ops)

        # Prefer using one consistent assignment operator
        consistency_score = (
            1.0 / len(unique_assignment_ops) if unique_assignment_ops else 1.0
        )

        return min(1.0, consistency_score)

    def _compute_clarity_score(self, tokens: List[Token]) -> float:
        """
        Compute semantic clarity of constructs.

        Args:
            tokens: List of tokens

        Returns:
            Clarity score in [0, 1]
        """
        identifiers = [
            token.text for token in tokens if token.type == TokenType.IDENTIFIER
        ]

        if not identifiers:
            return 1.0

        # Analyze identifier meaningfulness
        meaningful_count = 0
        for ident in identifiers:
            if self._is_meaningful_identifier(ident):
                meaningful_count += 1

        clarity_ratio = meaningful_count / len(identifiers)
        return clarity_ratio

    def _is_meaningful_identifier(self, identifier: str) -> bool:
        """
        Check if an identifier appears meaningful.

        Args:
            identifier: Identifier to check

        Returns:
            True if identifier appears meaningful
        """
        # Heuristics for meaningful identifiers

        # Too short (likely abbreviations)
        if len(identifier) < 3:
            return False

        # All single character or numbers
        if re.match(r"^[a-zA-Z0-9]$", identifier):
            return False

        # Contains vowels (more likely to be words)
        if not re.search(r"[aeiouAEIOU]", identifier):
            return False

        # Not just numbers
        if identifier.isdigit():
            return False

        # Common meaningful patterns
        meaningful_patterns = [
            r".*[Nn]ame.*",
            r".*[Tt]ype.*",
            r".*[Vv]alue.*",
            r".*[Ee]xpr.*",
            r".*[Ss]tmt.*",
            r".*[Dd]ecl.*",
            r".*[Rr]ule.*",
            r".*[Tt]oken.*",
            r".*[Ii]d.*",
        ]

        for pattern in meaningful_patterns:
            if re.match(pattern, identifier):
                return True

        return True  # Default to meaningful if passes basic checks

    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        return r"Readability(U) = w_1 \cdot Flesch(U) + w_2 \cdot Complexity(U) + w_3 \cdot Consistency(U) + w_4 \cdot Clarity(U)"

    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties of the readability metric.

        Returns:
            Dict with mathematical properties
        """
        return {
            "bounded": True,  # Always produces values in [0, 1]
            "monotonic": False,  # More tokens don't always mean higher readability
            "additive": False,  # Readability is not sum of parts
            "continuous": True,  # Small changes cause small changes in readability
        }


# Register the metric in the global registry
ReadabilityMetric.register_metric("readability")
