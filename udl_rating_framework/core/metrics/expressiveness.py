"""
Expressiveness Metric implementation.

Measures language power using formal language theory and Chomsky hierarchy classification.
"""

import zlib
import re
from typing import Dict, List
from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.representation import (
    UDLRepresentation,
    GrammarRule,
)


class Grammar:
    """Represents a formal grammar for Chomsky hierarchy classification."""

    def __init__(self, rules: List[GrammarRule]):
        """Initialize a grammar from a list of production rules.

        Args:
            rules: List of GrammarRule objects defining the grammar.
        """
        self.rules = rules
        self.terminals = set()
        self.non_terminals = set()
        self._analyze_symbols()

    def _analyze_symbols(self):
        """Analyze grammar to identify terminals and non-terminals."""
        # Collect all symbols from rules
        all_symbols = set()
        lhs_symbols = set()

        for rule in self.rules:
            lhs_symbols.add(rule.lhs)
            all_symbols.add(rule.lhs)
            all_symbols.update(rule.rhs)

        # Non-terminals are symbols that appear on LHS
        self.non_terminals = lhs_symbols

        # Terminals are symbols that never appear on LHS
        self.terminals = all_symbols - self.non_terminals

        # Filter out empty strings and operators
        self.terminals = {
            t for t in self.terminals if t and not self._is_grammar_operator(t)
        }

    def _is_grammar_operator(self, symbol: str) -> bool:
        """Check if symbol is a grammar operator rather than a terminal."""
        operators = {"|", "+", "*", "?", "(", ")", "[", "]", "{", "}"}
        return symbol in operators or len(symbol) == 1 and not symbol.isalnum()


class ExpressivenessMetric(QualityMetric):
    """
    Measures language power using Chomsky hierarchy.

    Mathematical Definition:
    Expressiveness(U) = (Chomsky_Level + Complexity_Score) / 2

    Where:
    - Chomsky_Level ∈ {0, 0.33, 0.67, 1.0} for Type-3, Type-2, Type-1, Type-0
    - Complexity_Score: Normalized Kolmogorov complexity approximation

    Algorithm:
    1. Classify grammar into Chomsky hierarchy
    2. Approximate Kolmogorov complexity via compression
    3. Combine scores with equal weighting
    """

    def __init__(self):
        """Initialize expressiveness metric with Chomsky level mappings."""
        # Chomsky hierarchy levels (normalized to [0,1])
        self.chomsky_levels = {
            3: 0.0,  # Type-3: Regular grammars (least expressive)
            2: 0.33,  # Type-2: Context-free grammars
            1: 0.67,  # Type-1: Context-sensitive grammars
            0: 1.0,  # Type-0: Unrestricted grammars (most expressive)
        }

    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute expressiveness score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Expressiveness score in [0, 1]
        """
        # Get grammar rules
        rules = udl.get_grammar_rules()

        # Handle empty grammar case
        if not rules:
            return 0.0  # Empty grammar has no expressiveness

        # Create Grammar object for analysis
        grammar = Grammar(rules)

        # Classify Chomsky hierarchy level
        chomsky_type = self.classify_chomsky_level(grammar)
        chomsky_score = self.chomsky_levels[chomsky_type]

        # Approximate Kolmogorov complexity
        complexity_score = self.approximate_kolmogorov_complexity(udl)

        # Combine scores with equal weighting
        # Formula: (Chomsky_Level + Complexity_Score) / 2
        expressiveness_score = (chomsky_score + complexity_score) / 2.0

        # Ensure result is bounded in [0, 1]
        return max(0.0, min(1.0, expressiveness_score))

    def classify_chomsky_level(self, grammar: Grammar) -> int:
        """
        Classify grammar into Chomsky hierarchy.

        Args:
            grammar: Grammar object to classify

        Returns:
            Chomsky type: 0 (unrestricted), 1 (context-sensitive),
                         2 (context-free), 3 (regular)
        """
        if not grammar.rules:
            return 3  # Empty grammar is regular

        # Check for Type-3 (Regular) grammar
        if self._is_regular_grammar(grammar):
            return 3

        # Check for Type-2 (Context-free) grammar
        if self._is_context_free_grammar(grammar):
            return 2

        # Check for Type-1 (Context-sensitive) grammar
        if self._is_context_sensitive_grammar(grammar):
            return 1

        # Default to Type-0 (Unrestricted)
        return 0

    def _is_regular_grammar(self, grammar: Grammar) -> bool:
        """
        Check if grammar is regular (Type-3).

        Regular grammars have rules of the form:
        - A → a (terminal)
        - A → aB (terminal followed by non-terminal)
        - A → ε (empty string)

        Args:
            grammar: Grammar to check

        Returns:
            True if grammar is regular
        """
        for rule in grammar.rules:
            rhs = rule.rhs

            # Skip empty RHS
            if not rhs:
                continue

            # Rule must have at most 2 symbols on RHS
            if len(rhs) > 2:
                return False

            # If 2 symbols, first must be terminal, second must be non-terminal
            if len(rhs) == 2:
                first, second = rhs[0], rhs[1]
                if (
                    first not in grammar.terminals
                    or second not in grammar.non_terminals
                ):
                    return False

            # If 1 symbol, it must be a terminal
            elif len(rhs) == 1:
                if rhs[0] not in grammar.terminals:
                    # Allow non-terminal if it's a simple substitution
                    if rhs[0] not in grammar.non_terminals:
                        return False

        return True

    def _is_context_free_grammar(self, grammar: Grammar) -> bool:
        """
        Check if grammar is context-free (Type-2).

        Context-free grammars have rules of the form:
        - A → α (single non-terminal on LHS, any string on RHS)

        Args:
            grammar: Grammar to check

        Returns:
            True if grammar is context-free
        """
        for rule in grammar.rules:
            # LHS must be a single non-terminal
            if rule.lhs not in grammar.non_terminals:
                return False

            # RHS can be any combination of terminals and non-terminals
            # This is automatically satisfied if LHS is single non-terminal

        return True

    def _is_context_sensitive_grammar(self, grammar: Grammar) -> bool:
        """
        Check if grammar is context-sensitive (Type-1).

        Context-sensitive grammars have rules of the form:
        - αAβ → αγβ (where |αγβ| ≥ |αAβ|, i.e., non-contracting)

        Args:
            grammar: Grammar to check

        Returns:
            True if grammar is context-sensitive
        """
        has_context_sensitive_rule = False

        for rule in grammar.rules:
            lhs_length = len(rule.lhs)
            rhs_length = len(rule.rhs)

            # Context-sensitive rules must be non-contracting
            # (RHS must be at least as long as LHS)
            if rhs_length < lhs_length:
                return False

            # If LHS has multiple symbols, it suggests context-sensitivity
            if lhs_length > 1:
                has_context_sensitive_rule = True

        # Must have at least one rule with multi-symbol LHS to be context-sensitive
        return has_context_sensitive_rule

    def approximate_kolmogorov_complexity(self, udl: UDLRepresentation) -> float:
        """
        Approximate Kolmogorov complexity using compression ratio.

        The Kolmogorov complexity K(x) of a string x is the length of the
        shortest program that produces x. We approximate this using compression:
        K(x) ≈ |compressed(x)| / |x|

        Args:
            udl: UDLRepresentation instance

        Returns:
            Normalized complexity score in [0, 1]
        """
        # Get the source text
        source_text = udl.source_text

        if not source_text or len(source_text) == 0:
            return 0.0

        # Remove whitespace and comments for more accurate complexity measure
        cleaned_text = self._clean_text_for_complexity(source_text)

        if not cleaned_text:
            return 0.0

        try:
            # Compress using zlib (deflate algorithm)
            compressed = zlib.compress(cleaned_text.encode("utf-8"))

            # Calculate compression ratio
            original_size = len(cleaned_text.encode("utf-8"))
            compressed_size = len(compressed)

            # Compression ratio (lower ratio = higher complexity)
            compression_ratio = compressed_size / original_size

            # Convert to complexity score (higher complexity = higher score)
            # Use 1 - compression_ratio, but normalize to reasonable range
            # Apply sigmoid-like transformation to map to [0,1] more smoothly
            # Most text compresses to 30-70% of original size
            # Map compression ratios: 0.3 → 1.0, 0.7 → 0.3, 1.0 → 0.0
            if compression_ratio <= 0.3:
                normalized_score = 1.0
            elif compression_ratio >= 1.0:
                normalized_score = 0.0
            else:
                # Linear interpolation between key points
                normalized_score = (1.0 - compression_ratio) / 0.7

            return max(0.0, min(1.0, normalized_score))

        except Exception:
            # If compression fails, use simple heuristics
            return self._fallback_complexity_measure(cleaned_text)

    def _clean_text_for_complexity(self, text: str) -> str:
        """
        Clean text for complexity analysis by removing noise.

        Args:
            text: Original source text

        Returns:
            Cleaned text with whitespace and comments removed
        """
        # Remove comments
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove comments (starting with #)
            if "#" in line:
                line = line[: line.index("#")]

            # Remove excessive whitespace
            line = re.sub(r"\s+", " ", line.strip())

            if line:  # Only keep non-empty lines
                cleaned_lines.append(line)

        return " ".join(cleaned_lines)

    def _fallback_complexity_measure(self, text: str) -> float:
        """
        Fallback complexity measure when compression fails.

        Uses simple heuristics based on:
        - Unique character count
        - Repetition patterns
        - Symbol diversity

        Args:
            text: Text to analyze

        Returns:
            Complexity score in [0, 1]
        """
        if not text:
            return 0.0

        # Count unique characters
        unique_chars = len(set(text))
        total_chars = len(text)

        # Character diversity ratio
        char_diversity = unique_chars / total_chars if total_chars > 0 else 0

        # Count unique tokens (split by common delimiters)
        tokens = re.split(r"[\s::=|(){}[\]]+", text)
        tokens = [t for t in tokens if t]  # Remove empty tokens

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)

        # Token diversity ratio
        token_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0

        # Combine measures
        complexity_score = (char_diversity + token_diversity) / 2.0

        return max(0.0, min(1.0, complexity_score))

    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        return r"Expressiveness(U) = \frac{Chomsky\_Level + Complexity\_Score}{2}"

    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties of the expressiveness metric.

        Returns:
            Dict with mathematical properties
        """
        return {
            "bounded": True,  # Always produces values in [0, 1]
            "monotonic": False,  # More rules don't always mean higher expressiveness
            "additive": False,  # Expressiveness is not sum of parts
            "continuous": False,  # Discrete changes in grammar type cause discrete changes
        }


# Register the metric in the global registry
ExpressivenessMetric.register_metric("expressiveness")
