#!/usr/bin/env python3
"""
Debug script to check broken grammar example.
"""

import os
import sys

from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.representation import UDLRepresentation

sys.path.insert(0, os.path.abspath("."))


def debug_broken_grammar():
    """Debug the broken grammar example."""

    # Load the broken grammar UDL
    broken_udl = """
# Broken Grammar Example
# A grammar with cycles and contradictions

# Grammar Rules with cycles
A ::= B C
B ::= A D  # Creates cycle: A -> B -> A
C ::= 'c'
D ::= 'd'

# Contradictory rules
Value ::= 'true'
Value ::= 'false'  # Same LHS, contradictory RHS

# Incomplete and inconsistent
Statement ::= Expression ';'
Expression ::= Term
# Missing Term definition - incomplete

# More contradictions
Number ::= Digit+
Number ::= Letter+  # Numbers can't be both digits and letters

# Undefined references
Program ::= Statement* EOF  # EOF not defined anywhere
"""

    print("=== Debugging Broken Grammar UDL ===")
    print("UDL Content:")
    print(broken_udl)

    # Create representation
    try:
        udl = UDLRepresentation(broken_udl, "broken.udl")
        print(f"\n✓ UDL representation created successfully")

        # Examine the parsed structure
        rules = udl.get_grammar_rules()
        print(f"✓ Found {len(rules)} grammar rules:")
        for i, rule in enumerate(rules):
            print(f"  {i + 1}. {rule.lhs} ::= {' '.join(rule.rhs)}")

        # Check grammar graph
        graph = udl.get_grammar_graph()
        print(
            f"✓ Grammar graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

        # Calculate consistency manually
        consistency_metric = ConsistencyMetric()

        # Check for cycles
        cycles = consistency_metric.detect_cycles(graph)
        print(f"✓ Cycles detected: {len(cycles)}")
        if cycles:
            for cycle in cycles:
                print(f"  Cycle: {' -> '.join(cycle + [cycle[0]])}")

        # Check for contradictions
        contradictions = consistency_metric.find_contradictions(rules)
        print(f"✓ Contradictions detected: {len(contradictions)}")
        if contradictions:
            for c1, c2 in contradictions:
                print(
                    f"  Contradiction: {c1.lhs} ::= {' '.join(c1.rhs)} vs {c2.lhs} ::= {' '.join(c2.rhs)}"
                )

        # Calculate consistency score
        consistency_score = consistency_metric.compute(udl)
        print(f"✓ Consistency score: {consistency_score:.6f}")

        # Manual calculation
        num_contradictions = len(contradictions)
        num_cycles = len(cycles)
        num_rules = len(rules)

        manual_score = 1.0 - (num_contradictions +
                              num_cycles) / (num_rules + 1)
        print(
            f"✓ Manual calculation: 1 - ({num_contradictions} + {num_cycles}) / ({num_rules} + 1) = {manual_score:.6f}"
        )

        print(f"✓ Difference: {abs(consistency_score - manual_score):.6f}")

        return consistency_score, manual_score

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    debug_broken_grammar()
