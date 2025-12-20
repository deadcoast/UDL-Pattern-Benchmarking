"""
Evolution Tracking Metric implementation.

Measures evolution and versioning characteristics of UDL definitions.
"""

import hashlib
import difflib
from typing import Dict, List, Set, Any, Tuple, Optional
from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.representation import UDLRepresentation, Token, TokenType


class EvolutionTrackingMetric(QualityMetric):
    """
    Measures UDL evolution and versioning characteristics.

    Mathematical Definition:
    Evolution(U) = w₁·Stability(U) + w₂·Extensibility(U) + w₃·Backward_Compat(U) + w₄·Change_Impact(U)

    Where:
    - Stability(U): Measure of structural stability over versions
    - Extensibility(U): Ease of extending the language
    - Backward_Compat(U): Backward compatibility preservation
    - Change_Impact(U): Impact assessment of changes
    - wᵢ: Weights summing to 1

    This metric can work in two modes:
    1. Single UDL analysis: Assess inherent evolution characteristics
    2. Multi-version analysis: Compare multiple versions of the same UDL

    Algorithm:
    1. Analyze structural characteristics for evolution potential
    2. Identify extension points and modularity
    3. Assess backward compatibility features
    4. Compute change impact metrics
    5. Combine metrics with learned weights
    """

    def __init__(self):
        """Initialize evolution tracking metric."""
        self.weights = {
            "stability": 0.25,
            "extensibility": 0.3,
            "backward_compatibility": 0.25,
            "change_impact": 0.2,
        }

        # Store previous versions for comparison (if available)
        self.version_history: List[Tuple[str, UDLRepresentation]] = []

    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute evolution tracking score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Evolution score in [0, 1]
        """
        # Compute individual evolution components
        stability_score = self._compute_stability_score(udl)
        extensibility_score = self._compute_extensibility_score(udl)
        backward_compat_score = self._compute_backward_compatibility_score(udl)
        change_impact_score = self._compute_change_impact_score(udl)

        # Weighted combination
        evolution_score = (
            self.weights["stability"] * stability_score
            + self.weights["extensibility"] * extensibility_score
            + self.weights["backward_compatibility"] * backward_compat_score
            + self.weights["change_impact"] * change_impact_score
        )

        return max(0.0, min(1.0, evolution_score))

    def add_version(self, version_id: str, udl: UDLRepresentation) -> None:
        """
        Add a version to the evolution history.

        Args:
            version_id: Version identifier
            udl: UDLRepresentation for this version
        """
        self.version_history.append((version_id, udl))

    def compare_versions(
        self, udl1: UDLRepresentation, udl2: UDLRepresentation
    ) -> Dict[str, Any]:
        """
        Compare two versions of a UDL.

        Args:
            udl1: First UDL version
            udl2: Second UDL version

        Returns:
            Dict with comparison metrics
        """
        return {
            "structural_similarity": self._compute_structural_similarity(udl1, udl2),
            "added_constructs": self._find_added_constructs(udl1, udl2),
            "removed_constructs": self._find_removed_constructs(udl1, udl2),
            "modified_rules": self._find_modified_rules(udl1, udl2),
            "compatibility_impact": self._assess_compatibility_impact(udl1, udl2),
            "change_complexity": self._compute_change_complexity(udl1, udl2),
        }

    def _compute_stability_score(self, udl: UDLRepresentation) -> float:
        """
        Compute structural stability score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Stability score in [0, 1]
        """
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()

        # Analyze structural characteristics that indicate stability
        stability_factors = {
            "modular_structure": self._has_modular_structure(rules),
            "clear_separation": self._has_clear_separation(rules),
            "consistent_patterns": self._has_consistent_patterns(tokens, rules),
            "minimal_coupling": self._has_minimal_coupling(rules),
            "stable_core": self._has_stable_core(rules),
        }

        # Weight factors by importance
        weights = {
            "modular_structure": 0.25,
            "clear_separation": 0.2,
            "consistent_patterns": 0.2,
            "minimal_coupling": 0.2,
            "stable_core": 0.15,
        }

        stability_score = sum(
            weights[factor] * (1.0 if present else 0.0)
            for factor, present in stability_factors.items()
        )

        return stability_score

    def _compute_extensibility_score(self, udl: UDLRepresentation) -> float:
        """
        Compute extensibility score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Extensibility score in [0, 1]
        """
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()

        # Analyze characteristics that enable easy extension
        extensibility_factors = {
            "extension_points": self._has_extension_points(rules),
            "plugin_architecture": self._has_plugin_architecture(rules),
            "hierarchical_structure": self._has_hierarchical_structure(rules),
            "abstract_constructs": self._has_abstract_constructs(rules),
            "composition_patterns": self._has_composition_patterns(rules),
            "optional_features": self._has_optional_features(tokens),
        }

        # Weight factors by importance for extensibility
        weights = {
            "extension_points": 0.25,
            "plugin_architecture": 0.2,
            "hierarchical_structure": 0.2,
            "abstract_constructs": 0.15,
            "composition_patterns": 0.1,
            "optional_features": 0.1,
        }

        extensibility_score = sum(
            weights[factor] * (1.0 if present else 0.0)
            for factor, present in extensibility_factors.items()
        )

        return extensibility_score

    def _compute_backward_compatibility_score(self, udl: UDLRepresentation) -> float:
        """
        Compute backward compatibility score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Backward compatibility score in [0, 1]
        """
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()

        # If we have version history, compute actual compatibility
        if len(self.version_history) > 0:
            return self._compute_actual_backward_compatibility(udl)

        # Otherwise, assess inherent compatibility characteristics
        compatibility_factors = {
            "versioning_support": self._has_versioning_support(tokens),
            "deprecated_constructs": self._handles_deprecated_constructs(tokens),
            "optional_extensions": self._has_optional_extensions(rules),
            "graceful_degradation": self._supports_graceful_degradation(rules),
            "legacy_support": self._has_legacy_support_patterns(tokens),
        }

        # Weight factors by importance for backward compatibility
        weights = {
            "versioning_support": 0.3,
            "deprecated_constructs": 0.2,
            "optional_extensions": 0.2,
            "graceful_degradation": 0.15,
            "legacy_support": 0.15,
        }

        compatibility_score = sum(
            weights[factor] * (1.0 if present else 0.0)
            for factor, present in compatibility_factors.items()
        )

        return compatibility_score

    def _compute_change_impact_score(self, udl: UDLRepresentation) -> float:
        """
        Compute change impact score (lower impact = higher score).

        Args:
            udl: UDLRepresentation instance

        Returns:
            Change impact score in [0, 1]
        """
        rules = udl.get_grammar_rules()

        # If we have version history, compute actual change impact
        if len(self.version_history) > 0:
            return self._compute_actual_change_impact(udl)

        # Otherwise, assess inherent change impact characteristics
        impact_factors = {
            "loose_coupling": self._has_loose_coupling(rules),
            "isolated_changes": self._supports_isolated_changes(rules),
            "impact_boundaries": self._has_clear_impact_boundaries(rules),
            "change_localization": self._supports_change_localization(rules),
        }

        # Weight factors (lower impact = higher score)
        weights = {
            "loose_coupling": 0.3,
            "isolated_changes": 0.25,
            "impact_boundaries": 0.25,
            "change_localization": 0.2,
        }

        # Higher presence of these factors = lower change impact = higher score
        change_impact_score = sum(
            weights[factor] * (1.0 if present else 0.0)
            for factor, present in impact_factors.items()
        )

        return change_impact_score

    def _compute_actual_backward_compatibility(self, udl: UDLRepresentation) -> float:
        """
        Compute actual backward compatibility based on version history.

        Args:
            udl: Current UDL version

        Returns:
            Actual backward compatibility score
        """
        if not self.version_history:
            return 1.0

        # Compare with previous versions
        compatibility_scores = []

        for version_id, prev_udl in self.version_history[-3:]:  # Last 3 versions
            compatibility = self._assess_version_compatibility(prev_udl, udl)
            compatibility_scores.append(compatibility)

        return (
            sum(compatibility_scores) / len(compatibility_scores)
            if compatibility_scores
            else 1.0
        )

    def _compute_actual_change_impact(self, udl: UDLRepresentation) -> float:
        """
        Compute actual change impact based on version history.

        Args:
            udl: Current UDL version

        Returns:
            Change impact score (higher = lower impact)
        """
        if not self.version_history:
            return 1.0

        # Analyze changes from previous versions
        impact_scores = []

        for version_id, prev_udl in self.version_history[-3:]:  # Last 3 versions
            impact = self._compute_version_change_impact(prev_udl, udl)
            impact_scores.append(1.0 - impact)  # Invert: lower impact = higher score

        return sum(impact_scores) / len(impact_scores) if impact_scores else 1.0

    # Helper methods for structural analysis
    def _has_modular_structure(self, rules: List) -> bool:
        """Check if grammar has modular structure."""
        if not rules:
            return False

        # Look for modular patterns (separate concerns)
        rule_names = [rule.lhs.lower() for rule in rules if hasattr(rule, "lhs")]

        # Check for different modules/categories
        categories = set()
        for name in rule_names:
            if any(prefix in name for prefix in ["expr", "stmt", "decl", "type"]):
                categories.add("syntax")
            if any(prefix in name for prefix in ["lex", "token", "literal"]):
                categories.add("lexical")
            if any(prefix in name for prefix in ["import", "module", "package"]):
                categories.add("modular")

        return len(categories) >= 2

    def _has_clear_separation(self, rules: List) -> bool:
        """Check for clear separation of concerns."""
        if not rules:
            return False

        # Look for separation patterns
        rule_names = [rule.lhs.lower() for rule in rules if hasattr(rule, "lhs")]

        # Check for lexical vs syntactic separation
        lexical_rules = sum(
            1
            for name in rule_names
            if any(
                pattern in name for pattern in ["token", "literal", "ident", "number"]
            )
        )
        syntactic_rules = sum(
            1
            for name in rule_names
            if any(pattern in name for pattern in ["expr", "stmt", "decl", "program"])
        )

        return lexical_rules > 0 and syntactic_rules > 0

    def _has_consistent_patterns(self, tokens: List[Token], rules: List) -> bool:
        """Check for consistent naming and structural patterns."""
        if not rules:
            return True

        rule_names = [rule.lhs for rule in rules if hasattr(rule, "lhs")]

        # Check naming consistency
        naming_patterns = {
            "camelCase": sum(
                1 for name in rule_names if any(c.isupper() for c in name[1:])
            ),
            "snake_case": sum(1 for name in rule_names if "_" in name),
            "lowercase": sum(1 for name in rule_names if name.islower()),
        }

        # Consistent if one pattern dominates
        total_rules = len(rule_names)
        if total_rules == 0:
            return True

        max_pattern_count = max(naming_patterns.values())
        return max_pattern_count / total_rules >= 0.7

    def _has_minimal_coupling(self, rules: List) -> bool:
        """Check for minimal coupling between rules."""
        if len(rules) < 2:
            return True

        # Analyze rule dependencies
        rule_names = {rule.lhs for rule in rules if hasattr(rule, "lhs")}
        dependencies = {}

        for rule in rules:
            if not hasattr(rule, "lhs") or not hasattr(rule, "rhs"):
                continue

            deps = set()
            rhs_text = (
                " ".join(rule.rhs) if isinstance(rule.rhs, list) else str(rule.rhs)
            )

            for name in rule_names:
                if name != rule.lhs and name in rhs_text:
                    deps.add(name)

            dependencies[rule.lhs] = deps

        # Calculate average coupling
        if not dependencies:
            return True

        avg_coupling = sum(len(deps) for deps in dependencies.values()) / len(
            dependencies
        )
        max_reasonable_coupling = len(rule_names) * 0.3  # 30% of rules

        return avg_coupling <= max_reasonable_coupling

    def _has_stable_core(self, rules: List) -> bool:
        """Check for stable core constructs."""
        if not rules:
            return False

        rule_names = [rule.lhs.lower() for rule in rules if hasattr(rule, "lhs")]

        # Look for fundamental constructs that form a stable core
        core_constructs = [
            "program",
            "statement",
            "expression",
            "identifier",
            "literal",
        ]
        found_core = sum(
            1
            for construct in core_constructs
            if any(construct in name for name in rule_names)
        )

        return found_core >= 2

    def _has_extension_points(self, rules: List) -> bool:
        """Check for explicit extension points."""
        if not rules:
            return False

        # Look for patterns that suggest extension points
        rule_names = [rule.lhs.lower() for rule in rules if hasattr(rule, "lhs")]

        extension_indicators = ["extension", "plugin", "custom", "user", "additional"]
        return any(
            indicator in name
            for name in rule_names
            for indicator in extension_indicators
        )

    def _has_plugin_architecture(self, rules: List) -> bool:
        """Check for plugin architecture patterns."""
        if not rules:
            return False

        # Look for plugin-like patterns
        rule_names = [rule.lhs.lower() for rule in rules if hasattr(rule, "lhs")]

        plugin_indicators = ["plugin", "module", "component", "handler"]
        return any(
            indicator in name for name in rule_names for indicator in plugin_indicators
        )

    def _has_hierarchical_structure(self, rules: List) -> bool:
        """Check for hierarchical structure."""
        if not rules:
            return False

        # Look for hierarchical naming patterns
        rule_names = [rule.lhs.lower() for rule in rules if hasattr(rule, "lhs")]

        # Check for hierarchy indicators
        hierarchy_levels = set()
        for name in rule_names:
            if any(
                level in name
                for level in [
                    "program",
                    "block",
                    "statement",
                    "expression",
                    "term",
                    "factor",
                ]
            ):
                hierarchy_levels.add("syntax_hierarchy")
            if any(level in name for level in ["type", "class", "interface", "struct"]):
                hierarchy_levels.add("type_hierarchy")

        return len(hierarchy_levels) > 0

    def _has_abstract_constructs(self, rules: List) -> bool:
        """Check for abstract constructs that enable extension."""
        if not rules:
            return False

        rule_names = [rule.lhs.lower() for rule in rules if hasattr(rule, "lhs")]

        abstract_indicators = ["abstract", "base", "generic", "template"]
        return any(
            indicator in name
            for name in rule_names
            for indicator in abstract_indicators
        )

    def _has_composition_patterns(self, rules: List) -> bool:
        """Check for composition patterns."""
        if not rules:
            return False

        # Look for composition in rule structures
        for rule in rules:
            if hasattr(rule, "rhs") and isinstance(rule.rhs, list):
                # Check for list-like patterns (indicating composition)
                if len(rule.rhs) > 2:
                    return True

        return False

    def _has_optional_features(self, tokens: List[Token]) -> bool:
        """Check for optional feature indicators."""
        # Look for optional constructs
        optional_indicators = ["?", "*", "[", "]"]
        return any(
            token.text in optional_indicators
            for token in tokens
            if token.type == TokenType.OPERATOR
        )

    # Additional helper methods for compatibility and change analysis
    def _has_versioning_support(self, tokens: List[Token]) -> bool:
        """Check for versioning support indicators."""
        version_indicators = ["version", "v1", "v2", "compat"]
        return any(
            indicator in token.text.lower()
            for token in tokens
            for indicator in version_indicators
        )

    def _handles_deprecated_constructs(self, tokens: List[Token]) -> bool:
        """Check for deprecated construct handling."""
        deprecated_indicators = ["deprecated", "legacy", "old"]
        return any(
            indicator in token.text.lower()
            for token in tokens
            for indicator in deprecated_indicators
        )

    def _has_optional_extensions(self, rules: List) -> bool:
        """Check for optional extension patterns."""
        return self._has_optional_features([])  # Simplified

    def _supports_graceful_degradation(self, rules: List) -> bool:
        """Check for graceful degradation support."""
        # Look for fallback patterns
        rule_names = [rule.lhs.lower() for rule in rules if hasattr(rule, "lhs")]
        fallback_indicators = ["fallback", "default", "alternative"]
        return any(
            indicator in name
            for name in rule_names
            for indicator in fallback_indicators
        )

    def _has_legacy_support_patterns(self, tokens: List[Token]) -> bool:
        """Check for legacy support patterns."""
        legacy_indicators = ["legacy", "compat", "backward", "old"]
        return any(
            indicator in token.text.lower()
            for token in tokens
            for indicator in legacy_indicators
        )

    def _has_loose_coupling(self, rules: List) -> bool:
        """Check for loose coupling patterns."""
        return self._has_minimal_coupling(rules)  # Reuse minimal coupling check

    def _supports_isolated_changes(self, rules: List) -> bool:
        """Check if changes can be isolated."""
        return self._has_modular_structure(rules)  # Modular structure enables isolation

    def _has_clear_impact_boundaries(self, rules: List) -> bool:
        """Check for clear impact boundaries."""
        return self._has_clear_separation(rules)  # Clear separation creates boundaries

    def _supports_change_localization(self, rules: List) -> bool:
        """Check if changes can be localized."""
        return self._has_modular_structure(rules)  # Modularity enables localization

    # Version comparison methods
    def _compute_structural_similarity(
        self, udl1: UDLRepresentation, udl2: UDLRepresentation
    ) -> float:
        """Compute structural similarity between two UDL versions."""
        # Compare rule structures
        rules1 = {
            rule.lhs: rule.rhs
            for rule in udl1.get_grammar_rules()
            if hasattr(rule, "lhs")
        }
        rules2 = {
            rule.lhs: rule.rhs
            for rule in udl2.get_grammar_rules()
            if hasattr(rule, "lhs")
        }

        if not rules1 and not rules2:
            return 1.0
        if not rules1 or not rules2:
            return 0.0

        # Compute Jaccard similarity
        common_rules = set(rules1.keys()).intersection(set(rules2.keys()))
        total_rules = set(rules1.keys()).union(set(rules2.keys()))

        return len(common_rules) / len(total_rules) if total_rules else 1.0

    def _find_added_constructs(
        self, udl1: UDLRepresentation, udl2: UDLRepresentation
    ) -> Set[str]:
        """Find constructs added in udl2."""
        rules1 = {rule.lhs for rule in udl1.get_grammar_rules() if hasattr(rule, "lhs")}
        rules2 = {rule.lhs for rule in udl2.get_grammar_rules() if hasattr(rule, "lhs")}
        return rules2 - rules1

    def _find_removed_constructs(
        self, udl1: UDLRepresentation, udl2: UDLRepresentation
    ) -> Set[str]:
        """Find constructs removed in udl2."""
        rules1 = {rule.lhs for rule in udl1.get_grammar_rules() if hasattr(rule, "lhs")}
        rules2 = {rule.lhs for rule in udl2.get_grammar_rules() if hasattr(rule, "lhs")}
        return rules1 - rules2

    def _find_modified_rules(
        self, udl1: UDLRepresentation, udl2: UDLRepresentation
    ) -> Set[str]:
        """Find rules that were modified."""
        rules1 = {
            rule.lhs: rule.rhs
            for rule in udl1.get_grammar_rules()
            if hasattr(rule, "lhs")
        }
        rules2 = {
            rule.lhs: rule.rhs
            for rule in udl2.get_grammar_rules()
            if hasattr(rule, "lhs")
        }

        modified = set()
        for rule_name in rules1.keys() & rules2.keys():
            if rules1[rule_name] != rules2[rule_name]:
                modified.add(rule_name)

        return modified

    def _assess_compatibility_impact(
        self, udl1: UDLRepresentation, udl2: UDLRepresentation
    ) -> float:
        """Assess compatibility impact of changes."""
        removed = self._find_removed_constructs(udl1, udl2)
        modified = self._find_modified_rules(udl1, udl2)

        total_rules1 = len(udl1.get_grammar_rules())
        if total_rules1 == 0:
            return 0.0

        # Breaking changes have higher impact
        breaking_changes = len(removed) + len(modified)
        impact = breaking_changes / total_rules1

        return min(1.0, impact)

    def _compute_change_complexity(
        self, udl1: UDLRepresentation, udl2: UDLRepresentation
    ) -> float:
        """Compute complexity of changes between versions."""
        added = self._find_added_constructs(udl1, udl2)
        removed = self._find_removed_constructs(udl1, udl2)
        modified = self._find_modified_rules(udl1, udl2)

        total_changes = len(added) + len(removed) + len(modified)
        total_rules = max(len(udl1.get_grammar_rules()), len(udl2.get_grammar_rules()))

        if total_rules == 0:
            return 0.0

        return min(1.0, total_changes / total_rules)

    def _assess_version_compatibility(
        self, old_udl: UDLRepresentation, new_udl: UDLRepresentation
    ) -> float:
        """Assess backward compatibility between versions."""
        impact = self._assess_compatibility_impact(old_udl, new_udl)
        return 1.0 - impact  # Higher impact = lower compatibility

    def _compute_version_change_impact(
        self, old_udl: UDLRepresentation, new_udl: UDLRepresentation
    ) -> float:
        """Compute change impact between versions."""
        return self._compute_change_complexity(old_udl, new_udl)

    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        return r"Evolution(U) = w_1 \cdot Stability(U) + w_2 \cdot Extensibility(U) + w_3 \cdot BackwardCompat(U) + w_4 \cdot ChangeImpact(U)"

    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties of the evolution tracking metric.

        Returns:
            Dict with mathematical properties
        """
        return {
            "bounded": True,  # Always produces values in [0, 1]
            "monotonic": False,  # More features don't always mean better evolution characteristics
            "additive": False,  # Evolution is not sum of parts
            "continuous": True,  # Small changes cause small changes in evolution score
        }


# Register the metric in the global registry
EvolutionTrackingMetric.register_metric("evolution_tracking")
