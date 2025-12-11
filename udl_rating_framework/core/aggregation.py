"""
Metric aggregation module.

Combines individual metrics into overall quality score.
"""

from typing import Dict


class MetricAggregator:
    """
    Combines metrics using weighted sum.

    Mathematical Definition:
    Q(U) = Σᵢ wᵢ · mᵢ(U)

    Where:
    - wᵢ: Weight for metric i (Σwᵢ = 1, wᵢ ≥ 0)
    - mᵢ: Individual metric function
    - Q: Overall quality score ∈ [0, 1]

    Properties:
    1. Boundedness: If all mᵢ ∈ [0,1] and Σwᵢ=1, then Q ∈ [0,1]
    2. Monotonicity: If mᵢ increases, Q increases (for wᵢ > 0)
    3. Linearity: Q is linear in each metric
    """

    def __init__(self, weights: Dict[str, float]):
        """
        Initialize with metric weights.

        Args:
            weights: Dict mapping metric names to weights

        Requires:
            sum(weights.values()) == 1.0
            all(w >= 0 for w in weights.values())
        """
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        if any(w < 0 for w in weights.values()):
            raise ValueError("All weights must be non-negative")

        self.weights = weights

    def aggregate(self, metric_values: Dict[str, float]) -> float:
        """
        Compute Q = Σ wᵢ · mᵢ.

        Args:
            metric_values: Dict mapping metric names to computed values

        Returns:
            Overall quality score in [0, 1]
        """
        return sum(
            self.weights[name] * value
            for name, value in metric_values.items()
            if name in self.weights
        )
