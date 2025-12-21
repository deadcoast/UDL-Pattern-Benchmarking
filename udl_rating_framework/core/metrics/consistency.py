"""
Consistency Metric implementation.

Measures internal coherence using graph-theoretic analysis of grammar rules.
"""

import networkx as nx
from typing import List, Tuple, Dict
from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.representation import (
    UDLRepresentation,
    GrammarRule,
    Constraint,
)


class ConsistencyMetric(QualityMetric):
    """
    Measures rule coherence using graph analysis.

    Mathematical Definition:
    Consistency(U) = 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)

    Where:
    - Contradictions: Pairs of rules that cannot both be satisfied
    - Cycles: Circular dependencies in grammar graph
    - Rules: Total number of production rules

    Algorithm:
    1. Build dependency graph from grammar rules
    2. Detect cycles using DFS (O(V + E))
    3. Identify contradictions using constraint analysis
    4. Normalize by total rule count
    """

    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute consistency score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Consistency score in [0, 1]
        """
        rules = udl.get_grammar_rules()

        # Handle empty grammar case
        if not rules:
            return 1.0  # Empty grammar is perfectly consistent

        # Get grammar graph for cycle detection
        graph = udl.get_grammar_graph()

        # Detect cycles in the grammar graph
        cycles = self.detect_cycles(graph)
        cycle_count = len(cycles)

        # Find contradictions in rules
        contradictions = self.find_contradictions(rules)
        contradiction_count = len(contradictions)

        # Compute consistency score using the specified formula
        total_issues = cycle_count + contradiction_count
        total_rules = len(rules)

        # Formula: 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)
        # The +1 in denominator prevents division by zero and provides normalization
        consistency_score = 1.0 - (total_issues / (total_rules + 1))

        # Ensure result is bounded in [0, 1]
        return max(0.0, min(1.0, consistency_score))

    def detect_cycles(self, graph: nx.DiGraph) -> List[List[str]]:
        """
        Find all cycles in grammar graph using DFS.

        Args:
            graph: Grammar dependency graph

        Returns:
            List of cycles, where each cycle is a list of nodes
        """
        try:
            # Use NetworkX's simple_cycles which implements Johnson's algorithm
            # This finds all elementary cycles (no repeated nodes except start/end)
            cycles = list(nx.simple_cycles(graph))
            return cycles
        except Exception:
            # Fallback to manual DFS if NetworkX fails
            return self._manual_cycle_detection(graph)

    def _manual_cycle_detection(self, graph: nx.DiGraph) -> List[List[str]]:
        """
        Manual cycle detection using DFS as fallback.

        Args:
            graph: Grammar dependency graph

        Returns:
            List of detected cycles
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> None:
            """DFS helper function for cycle detection."""
            if node in rec_stack:
                # Found a cycle - extract it from the path
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Visit all neighbors
            for neighbor in graph.successors(node):
                dfs(neighbor)

            # Backtrack
            rec_stack.remove(node)
            path.pop()

        # Run DFS from all unvisited nodes
        for node in graph.nodes():
            if node not in visited:
                dfs(node)

        return cycles

    def find_contradictions(
        self, rules: List[GrammarRule]
    ) -> List[Tuple[GrammarRule, GrammarRule]]:
        """
        Identify contradictory rule pairs using constraint analysis.

        Args:
            rules: List of grammar rules to analyze

        Returns:
            List of contradictory rule pairs
        """
        contradictions = []

        # Check for direct contradictions between rules
        for i, rule1 in enumerate(rules):
            for j, rule2 in enumerate(rules[i + 1 :], i + 1):
                if self._are_contradictory(rule1, rule2):
                    contradictions.append((rule1, rule2))

        return contradictions

    def _are_contradictory(self, rule1: GrammarRule, rule2: GrammarRule) -> bool:
        """
        Check if two rules are contradictory.

        Args:
            rule1: First grammar rule
            rule2: Second grammar rule

        Returns:
            True if rules contradict each other
        """
        # Rules with same LHS but conflicting RHS patterns
        if rule1.lhs == rule2.lhs:
            # Check for mutually exclusive patterns
            if self._are_mutually_exclusive(rule1.rhs, rule2.rhs):
                return True

        # Check constraint-based contradictions
        if self._have_conflicting_constraints(rule1, rule2):
            return True

        # Check for semantic contradictions
        if self._have_semantic_conflicts(rule1, rule2):
            return True

        return False

    def _are_mutually_exclusive(self, rhs1: List[str], rhs2: List[str]) -> bool:
        """
        Check if two RHS patterns are mutually exclusive.

        Args:
            rhs1: Right-hand side of first rule
            rhs2: Right-hand side of second rule

        Returns:
            True if patterns cannot both be valid
        """
        # Simple heuristics for mutual exclusion

        # If both are single terminals that are different
        if (
            len(rhs1) == 1
            and len(rhs2) == 1
            and rhs1[0] != rhs2[0]
            and self._is_terminal(rhs1[0])
            and self._is_terminal(rhs2[0])
        ):
            return True

        # If one requires a specific terminal and the other forbids it
        rhs1_terminals = {s for s in rhs1 if self._is_terminal(s)}
        rhs2_terminals = {s for s in rhs2 if self._is_terminal(s)}

        # Check for explicit contradictions (e.g., 'true' vs 'false')
        contradictory_pairs = {
            ("'true'", "'false'"),
            ("'yes'", "'no'"),
            ("'on'", "'off'"),
            ("'1'", "'0'"),
        }

        for term1 in rhs1_terminals:
            for term2 in rhs2_terminals:
                if (term1, term2) in contradictory_pairs or (
                    term2,
                    term1,
                ) in contradictory_pairs:
                    return True

        return False

    def _is_terminal(self, symbol: str) -> bool:
        """
        Check if a symbol is a terminal (quoted string or literal).

        Args:
            symbol: Symbol to check

        Returns:
            True if symbol is a terminal
        """
        return (
            (symbol.startswith("'") and symbol.endswith("'"))
            or (symbol.startswith('"') and symbol.endswith('"'))
            or symbol.isdigit()
        )

    def _have_conflicting_constraints(
        self, rule1: GrammarRule, rule2: GrammarRule
    ) -> bool:
        """
        Check if rules have conflicting constraints.

        Args:
            rule1: First grammar rule
            rule2: Second grammar rule

        Returns:
            True if constraints conflict
        """
        # Check for explicit constraint conflicts
        for constraint1 in rule1.constraints:
            for constraint2 in rule2.constraints:
                if self._constraints_conflict(constraint1, constraint2):
                    return True

        return False

    def _constraints_conflict(
        self, constraint1: Constraint, constraint2: Constraint
    ) -> bool:
        """
        Check if two constraints conflict.

        Args:
            constraint1: First constraint
            constraint2: Second constraint

        Returns:
            True if constraints are contradictory
        """
        # Simple constraint conflict detection
        if constraint1.type == constraint2.type:
            # Same type constraints with different conditions might conflict
            if constraint1.condition != constraint2.condition:
                # Check for obvious conflicts
                if (
                    "not" in constraint1.condition
                    and constraint1.condition.replace("not ", "")
                    == constraint2.condition
                ):
                    return True
                if (
                    "not" in constraint2.condition
                    and constraint2.condition.replace("not ", "")
                    == constraint1.condition
                ):
                    return True

        return False

    def _have_semantic_conflicts(self, rule1: GrammarRule, rule2: GrammarRule) -> bool:
        """
        Check for semantic conflicts between rules.

        Args:
            rule1: First grammar rule
            rule2: Second grammar rule

        Returns:
            True if rules have semantic conflicts
        """
        # Check metadata for semantic conflicts
        meta1 = rule1.metadata or {}
        meta2 = rule2.metadata or {}

        # Check for conflicting operators or semantics
        if "operator" in meta1 and "operator" in meta2:
            op1 = meta1["operator"]
            op2 = meta2["operator"]

            # Different assignment operators for same LHS might indicate conflict
            if (
                rule1.lhs == rule2.lhs
                and op1 != op2
                and op1 in ["::=", ":=", "="]
                and op2 in ["::=", ":=", "="]
            ):
                return True

        return False

    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        return r"Consistency(U) = 1 - \frac{|Contradictions| + |Cycles|}{|Rules| + 1}"

    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties of the consistency metric.

        Returns:
            Dict with mathematical properties
        """
        return {
            "bounded": True,  # Always produces values in [0, 1]
            "monotonic": False,  # Not monotonic (more rules don't always mean higher consistency)
            "additive": False,  # Not additive (consistency is not sum of parts)
            "continuous": False,  # Discrete changes in rules can cause discrete changes in consistency
        }


# Register the metric in the global registry
ConsistencyMetric.register_metric("consistency")
