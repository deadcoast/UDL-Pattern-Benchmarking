"""
Base class for quality metrics.

Defines the abstract interface that all metrics must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict


class QualityMetric(ABC):
    """
    Abstract base class for quality metrics.
    
    Mathematical Contract:
    Each metric must define a function f: UDL_Space → [0,1]
    with the following properties:
    1. Boundedness: ∀u ∈ UDL_Space, 0 ≤ f(u) ≤ 1
    2. Determinism: f(u₁) = f(u₂) if u₁ = u₂
    3. Computability: f must terminate in polynomial time
    """
    
    @abstractmethod
    def compute(self, udl) -> float:
        """
        Compute metric value.
        
        Args:
            udl: UDLRepresentation instance
        
        Returns:
            float in [0, 1]
        """
        pass
    
    @abstractmethod
    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties.
        
        Returns:
            Dict with keys: 'bounded', 'monotonic', 'additive', 'continuous'
        """
        pass
    
    def verify_boundedness(self, udl) -> bool:
        """Verify 0 ≤ compute(udl) ≤ 1."""
        value = self.compute(udl)
        return 0.0 <= value <= 1.0
    
    def verify_determinism(self, udl, trials: int = 10) -> bool:
        """Verify same input produces same output."""
        values = [self.compute(udl) for _ in range(trials)]
        return all(v == values[0] for v in values)
