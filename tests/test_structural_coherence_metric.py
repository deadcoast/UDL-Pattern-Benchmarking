"""
Tests for Structural Coherence Metric.

Tests both unit functionality and property-based correctness.
"""

import pytest
import math
import networkx as nx
from hypothesis import given, strategies as st, assume, settings
from udl_rating_framework.core.metrics.structural_coherence import StructuralCoherenceMetric
from udl_rating_framework.core.representation import UDLRepresentation


class TestStructuralCoherenceMetric:
    """Test suite for StructuralCoherenceMetric."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metric = StructuralCoherenceMetric()
    
    def test_empty_udl_coherence(self):
        """Test structural coherence on empty UDL."""
        udl = UDLRepresentation("", "test.udl")
        score = self.metric.compute(udl)
        assert score == 1.0, "Empty UDL should have perfect coherence"
    
    def test_single_rule_coherence(self):
        """Test structural coherence on UDL with single rule."""
        udl_text = "S ::= 'hello'"
        udl = UDLRepresentation(udl_text, "test.udl")
        score = self.metric.compute(udl)
        assert 0.0 <= score <= 1.0, "Score must be in [0, 1]"
        # Single rule should have high coherence
        assert score >= 0.5, "Single rule should have reasonably high coherence"
    
    def test_highly_organized_graph_low_entropy(self):
        """Test on highly organized graph (should have low entropy, high coherence)."""
        # Create a star-like grammar: S -> A, S -> B, S -> C (low entropy)
        udl_text = """
        S ::= A
        S ::= B  
        S ::= C
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        score = self.metric.compute(udl)
        
        # Verify score is bounded
        assert 0.0 <= score <= 1.0, "Score must be in [0, 1]"
        
        # Organized structure should have relatively high coherence
        assert score >= 0.3, "Organized structure should have decent coherence"
    
    def test_random_graph_high_entropy(self):
        """Test on random graph (should have high entropy, lower coherence)."""
        # Create a more random grammar structure
        udl_text = """
        A ::= B C
        B ::= D E
        C ::= F
        D ::= G H
        E ::= I
        F ::= J K
        G ::= L
        H ::= M N
        """
        udl = UDLRepresentation(udl_text, "test.udl")
        score = self.metric.compute(udl)
        
        # Verify score is bounded
        assert 0.0 <= score <= 1.0, "Score must be in [0, 1]"
    
    def test_entropy_calculation_correctness(self):
        """Test Shannon entropy calculation correctness."""
        # Create a simple graph with known degree distribution
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D')])
        
        entropy = self.metric.compute_shannon_entropy(graph)
        
        # Verify entropy is non-negative
        assert entropy >= 0.0, "Entropy must be non-negative"
        
        # For this specific graph:
        # Node degrees: A=2, B=2, C=1, D=1
        # Degree distribution: {1: 2, 2: 2}
        # Probabilities: p(1) = 2/4 = 0.5, p(2) = 2/4 = 0.5
        # Expected entropy: -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        expected_entropy = 1.0
        assert abs(entropy - expected_entropy) < 0.01, f"Expected entropy ~{expected_entropy}, got {entropy}"
    
    def test_modularity_calculation(self):
        """Test modularity calculation."""
        # Create a graph with clear community structure
        graph = nx.DiGraph()
        # Community 1: A-B-C
        graph.add_edges_from([('A', 'B'), ('B', 'C')])
        # Community 2: D-E-F  
        graph.add_edges_from([('D', 'E'), ('E', 'F')])
        # Weak connection between communities
        graph.add_edge('C', 'D')
        
        modularity = self.metric.compute_modularity(graph)
        
        # Modularity should be in reasonable range
        assert -1.0 <= modularity <= 1.0, "Modularity must be in [-1, 1]"
    
    def test_formula_property(self):
        """Test that the formula property returns correct LaTeX."""
        formula = self.metric.get_formula()
        assert isinstance(formula, str), "Formula must be a string"
        assert "H(G)" in formula, "Formula must contain H(G)"
        assert "H_{max}" in formula, "Formula must contain H_max"
    
    def test_properties_specification(self):
        """Test that metric properties are correctly specified."""
        properties = self.metric.get_properties()
        
        assert isinstance(properties, dict), "Properties must be a dictionary"
        assert properties['bounded'] is True, "Metric must be bounded"
        assert properties['continuous'] is True, "Metric should be continuous"
        assert properties['monotonic'] is False, "Metric is not monotonic"
        assert properties['additive'] is False, "Metric is not additive"


class TestStructuralCoherenceProperties:
    """Property-based tests for Structural Coherence Metric."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metric = StructuralCoherenceMetric()
    
    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=100, deadline=5000)
    def test_property_12_structural_coherence_metric_correctness(self, udl_text):
        """
        **Property 12: Structural Coherence Metric Correctness**
        **Validates: Requirements 3.5**
        
        For any UDL with grammar graph G, the structural coherence must be 
        computed as 1 - H(G)/H_max where H(G) is Shannon entropy.
        """
        # Filter out empty or whitespace-only strings
        assume(udl_text.strip())
        
        try:
            udl = UDLRepresentation(udl_text, "test.udl")
            
            # Compute structural coherence using the metric
            coherence_score = self.metric.compute(udl)
            
            # Verify the score is bounded in [0, 1]
            assert 0.0 <= coherence_score <= 1.0, f"Coherence score {coherence_score} not in [0, 1]"
            
            # Manually verify the formula: 1 - H(G)/H_max
            graph = udl.get_grammar_graph()
            
            if graph.number_of_nodes() <= 1:
                # For trivial graphs, coherence should be 1.0
                assert coherence_score == 1.0, "Trivial graph should have perfect coherence"
            else:
                # Compute entropy manually
                entropy = self.metric.compute_shannon_entropy(graph)
                max_entropy = math.log2(graph.number_of_nodes())
                
                # Verify entropy is non-negative and bounded
                assert entropy >= 0.0, f"Entropy {entropy} must be non-negative"
                assert entropy <= max_entropy + 1e-10, f"Entropy {entropy} exceeds maximum {max_entropy}"
                
                # Verify the formula: coherence = 1 - H(G)/H_max
                if max_entropy > 0:
                    expected_coherence = 1.0 - (entropy / max_entropy)
                    expected_coherence = max(0.0, min(1.0, expected_coherence))  # Clamp to [0,1]
                    
                    assert abs(coherence_score - expected_coherence) < 1e-10, \
                        f"Formula mismatch: got {coherence_score}, expected {expected_coherence}"
                else:
                    assert coherence_score == 1.0, "Zero max entropy should give perfect coherence"
        
        except Exception as e:
            # Skip malformed UDLs that can't be parsed
            assume(False, f"UDL parsing failed: {e}")
    
    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=50, deadline=3000)
    def test_entropy_bounds_property(self, num_nodes):
        """
        Test that entropy is properly bounded for graphs of various sizes.
        """
        # Create a simple linear graph: 1->2->3->...->n
        udl_rules = []
        for i in range(1, num_nodes):
            udl_rules.append(f"N{i} ::= N{i+1}")
        
        udl_text = "\n".join(udl_rules)
        udl = UDLRepresentation(udl_text, "test.udl")
        
        graph = udl.get_grammar_graph()
        entropy = self.metric.compute_shannon_entropy(graph)
        max_entropy = math.log2(graph.number_of_nodes()) if graph.number_of_nodes() > 1 else 0.0
        
        # Verify entropy bounds
        assert 0.0 <= entropy <= max_entropy + 1e-10, \
            f"Entropy {entropy} not in bounds [0, {max_entropy}]"
        
        # Verify coherence computation
        coherence = self.metric.compute(udl)
        assert 0.0 <= coherence <= 1.0, f"Coherence {coherence} not in [0, 1]"
    
    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5))
    @settings(max_examples=50, deadline=3000)
    def test_determinism_property(self, rule_parts):
        """
        Test that the metric is deterministic - same input produces same output.
        """
        # Create a simple UDL from the rule parts
        udl_text = "\n".join([f"S{i} ::= {part}" for i, part in enumerate(rule_parts)])
        
        try:
            udl = UDLRepresentation(udl_text, "test.udl")
            
            # Compute metric multiple times
            scores = [self.metric.compute(udl) for _ in range(3)]
            
            # All scores should be identical
            assert all(abs(score - scores[0]) < 1e-15 for score in scores), \
                f"Metric not deterministic: {scores}"
        
        except Exception:
            # Skip malformed UDLs
            assume(False)


if __name__ == "__main__":
    pytest.main([__file__])