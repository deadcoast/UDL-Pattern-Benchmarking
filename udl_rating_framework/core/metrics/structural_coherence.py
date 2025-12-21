"""
Structural Coherence Metric implementation.

Measures organizational quality using information theory and graph analysis.
"""

import math
from collections import Counter
from typing import Any, Dict, List

import networkx as nx
import numpy as np

from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.representation import UDLRepresentation


class StructuralCoherenceMetric(QualityMetric):
    """
    Measures organization using Shannon entropy and graph modularity.

    Mathematical Definition:
    Structural_Coherence(U) = 1 - H(G) / H_max

    Where:
    - H(G): Shannon entropy of grammar graph structure
    - H_max: Maximum possible entropy (log₂|V|)
    - Lower entropy indicates better organization

    Algorithm:
    1. Compute degree distribution of grammar graph
    2. Calculate Shannon entropy H = -Σ p(d) log₂ p(d)
    3. Normalize by maximum entropy
    4. Optionally incorporate modularity for additional structural insight
    """

    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute structural coherence score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Structural coherence score in [0, 1]
        """
        graph = udl.get_grammar_graph()

        # Handle empty or trivial graphs
        if graph.number_of_nodes() <= 1:
            return 1.0  # Single node or empty graph is perfectly coherent

        # Compute Shannon entropy of the graph structure
        entropy = self.compute_shannon_entropy(graph)

        # Compute maximum possible entropy
        num_nodes = graph.number_of_nodes()
        max_entropy = math.log2(num_nodes) if num_nodes > 1 else 0.0

        # Compute structural coherence: 1 - H(G)/H_max
        if max_entropy == 0.0:
            coherence = 1.0
        else:
            coherence = 1.0 - (entropy / max_entropy)

        # Ensure result is bounded in [0, 1]
        return max(0.0, min(1.0, coherence))

    def compute_shannon_entropy(self, graph: nx.DiGraph) -> float:
        """
        Calculate Shannon entropy H(G) = -Σ p_i log₂(p_i) of graph structure.

        The entropy is computed based on the degree distribution of the graph,
        which captures the structural organization. Lower entropy indicates
        more organized, hierarchical structure.

        Args:
            graph: Grammar dependency graph

        Returns:
            Shannon entropy of the graph structure
        """
        if graph.number_of_nodes() <= 1:
            return 0.0

        # Compute degree distribution (considering both in-degree and out-degree)
        degrees = []
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            # Use total degree as the structural measure
            total_degree = in_degree + out_degree
            degrees.append(total_degree)

        # Count frequency of each degree value
        degree_counts = Counter(degrees)
        total_nodes = len(degrees)

        # Compute probability distribution
        probabilities = []
        for count in degree_counts.values():
            prob = count / total_nodes
            probabilities.append(prob)

        # Compute Shannon entropy: H = -Σ p_i log₂(p_i)
        entropy = 0.0
        for prob in probabilities:
            if prob > 0:  # Avoid log(0)
                entropy -= prob * math.log2(prob)

        return entropy

    def compute_modularity(self, graph: nx.DiGraph) -> float:
        """
        Calculate Newman modularity Q of the graph.

        Modularity measures the strength of division of a network into modules
        (communities). Higher modularity indicates better structural organization
        with clear separation between different parts of the grammar.

        Args:
            graph: Grammar dependency graph

        Returns:
            Modularity score (typically in [-1, 1])
        """
        if graph.number_of_nodes() <= 1:
            return 0.0

        try:
            # Convert to undirected graph for community detection
            undirected_graph = graph.to_undirected()

            # Use NetworkX's modularity calculation with greedy community detection
            communities = nx.community.greedy_modularity_communities(
                undirected_graph)
            modularity = nx.community.modularity(undirected_graph, communities)

            return modularity
        except Exception:
            # Fallback to manual modularity calculation
            return self._manual_modularity_calculation(graph)

    def _manual_modularity_calculation(self, graph: nx.DiGraph) -> float:
        """
        Manual modularity calculation as fallback.

        Args:
            graph: Grammar dependency graph

        Returns:
            Modularity score
        """
        # Simple modularity approximation based on clustering coefficient
        try:
            # Convert to undirected for clustering coefficient
            undirected_graph = graph.to_undirected()

            # Use average clustering coefficient as a proxy for modularity
            clustering = nx.average_clustering(undirected_graph)

            # Normalize to approximate modularity range
            return clustering * 2.0 - 1.0  # Map [0,1] to [-1,1]
        except Exception:
            return 0.0

    def compute_alternative_entropy_measures(
        self, graph: nx.DiGraph
    ) -> Dict[str, float]:
        """
        Compute alternative entropy measures for comprehensive analysis.

        Args:
            graph: Grammar dependency graph

        Returns:
            Dictionary of various entropy measures
        """
        measures = {}

        if graph.number_of_nodes() <= 1:
            return {
                "degree_entropy": 0.0,
                "betweenness_entropy": 0.0,
                "clustering_entropy": 0.0,
            }

        # Degree-based entropy (already computed in main method)
        measures["degree_entropy"] = self.compute_shannon_entropy(graph)

        # Betweenness centrality entropy
        try:
            betweenness = nx.betweenness_centrality(graph)
            betweenness_values = list(betweenness.values())
            measures["betweenness_entropy"] = self._compute_entropy_from_values(
                betweenness_values
            )
        except Exception:
            measures["betweenness_entropy"] = 0.0

        # Clustering coefficient entropy
        try:
            undirected = graph.to_undirected()
            clustering = nx.clustering(undirected)
            clustering_values = list(clustering.values())
            measures["clustering_entropy"] = self._compute_entropy_from_values(
                clustering_values
            )
        except Exception:
            measures["clustering_entropy"] = 0.0

        return measures

    def _compute_entropy_from_values(self, values: List[float]) -> float:
        """
        Compute entropy from a list of continuous values by discretizing them.

        Args:
            values: List of numerical values

        Returns:
            Shannon entropy of the discretized distribution
        """
        if not values or len(values) <= 1:
            return 0.0

        # Discretize continuous values into bins
        num_bins = min(10, len(values))  # Use at most 10 bins

        try:
            # Create histogram
            counts, _ = np.histogram(values, bins=num_bins)

            # Remove zero counts
            counts = counts[counts > 0]

            if len(counts) <= 1:
                return 0.0

            # Compute probabilities
            total = sum(counts)
            probabilities = counts / total

            # Compute entropy
            entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
            return entropy
        except Exception:
            return 0.0

    def analyze_graph_structure(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Comprehensive analysis of graph structural properties.

        Args:
            graph: Grammar dependency graph

        Returns:
            Dictionary with various structural measures
        """
        analysis = {}

        # Basic graph properties
        analysis["num_nodes"] = graph.number_of_nodes()
        analysis["num_edges"] = graph.number_of_edges()
        analysis["density"] = nx.density(graph)

        if graph.number_of_nodes() <= 1:
            return analysis

        # Connectivity measures
        try:
            analysis["is_strongly_connected"] = nx.is_strongly_connected(graph)
            analysis["num_strongly_connected_components"] = (
                nx.number_strongly_connected_components(graph)
            )
            analysis["is_weakly_connected"] = nx.is_weakly_connected(graph)
        except Exception:
            analysis["is_strongly_connected"] = False
            analysis["num_strongly_connected_components"] = graph.number_of_nodes()
            analysis["is_weakly_connected"] = False

        # Centrality measures
        try:
            analysis["average_degree"] = (
                sum(dict(graph.degree()).values()) / graph.number_of_nodes()
            )
        except Exception:
            analysis["average_degree"] = 0.0

        # Structural measures
        analysis["shannon_entropy"] = self.compute_shannon_entropy(graph)
        analysis["modularity"] = self.compute_modularity(graph)

        # Alternative entropy measures
        analysis.update(self.compute_alternative_entropy_measures(graph))

        return analysis

    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        return r"Structural\_Coherence(U) = 1 - \frac{H(G)}{H_{max}} \text{ where } H(G) = -\sum_{i} p_i \log_2(p_i)"

    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties of the structural coherence metric.

        Returns:
            Dict with mathematical properties
        """
        return {
            "bounded": True,  # Always produces values in [0, 1]
            # Not monotonic (more structure doesn't always mean higher coherence)
            "monotonic": False,
            "additive": False,  # Not additive (coherence is not sum of parts)
            "continuous": True,  # Small changes in structure lead to small changes in coherence
        }


# Register the metric in the global registry
StructuralCoherenceMetric.register_metric("structural_coherence")
