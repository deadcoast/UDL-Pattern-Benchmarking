"""
Base class for quality metrics.

Defines the abstract interface that all metrics must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

logger = logging.getLogger(__name__)


class MetricRegistry:
    """
    Registry for quality metrics implementing plugin architecture.

    Allows dynamic registration and discovery of metrics with validation
    of mathematical properties.
    """

    _instance: Optional["MetricRegistry"] = None
    _metrics: Dict[str, Type["QualityMetric"]] = {}

    def __new__(cls) -> "MetricRegistry":
        """Singleton pattern for global metric registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, metric_class: Type["QualityMetric"]) -> None:
        """
        Register a metric class.

        Args:
            name: Unique name for the metric
            metric_class: Class implementing QualityMetric interface

        Raises:
            ValueError: If metric doesn't satisfy required properties
            TypeError: If metric_class is not a QualityMetric subclass
        """
        if not issubclass(metric_class, QualityMetric):
            raise TypeError(
                f"Metric class {metric_class} must inherit from QualityMetric"
            )

        if name in cls._metrics:
            logger.warning(f"Overriding existing metric registration: {name}")

        # Validate metric properties (will be checked during instantiation)
        cls._metrics[name] = metric_class
        logger.info(f"Registered metric: {name}")

    @classmethod
    def get_metric(cls, name: str) -> Type["QualityMetric"]:
        """
        Get a registered metric class by name.

        Args:
            name: Name of the metric

        Returns:
            Metric class

        Raises:
            KeyError: If metric is not registered
        """
        if name not in cls._metrics:
            raise KeyError(
                f"Metric '{name}' not found in registry. "
                f"Available metrics: {list(cls._metrics.keys())}"
            )
        return cls._metrics[name]

    @classmethod
    def list_metrics(cls) -> Dict[str, Type["QualityMetric"]]:
        """Return all registered metrics."""
        return cls._metrics.copy()

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a metric.

        Args:
            name: Name of the metric to remove
        """
        if name in cls._metrics:
            del cls._metrics[name]
            logger.info(f"Unregistered metric: {name}")

    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics (mainly for testing)."""
        cls._metrics.clear()
        logger.info("Cleared all metric registrations")


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

    def validate_properties(self, udl) -> Dict[str, bool]:
        """
        Validate all mathematical properties of the metric.

        Args:
            udl: UDLRepresentation instance for testing

        Returns:
            Dict mapping property names to validation results
        """
        results = {}

        try:
            results["bounded"] = self.verify_boundedness(udl)
        except Exception as e:
            logger.error(f"Boundedness validation failed: {e}")
            results["bounded"] = False

        try:
            results["deterministic"] = self.verify_determinism(udl)
        except Exception as e:
            logger.error(f"Determinism validation failed: {e}")
            results["deterministic"] = False

        # Check if metric claims to have properties it actually satisfies
        claimed_properties = self.get_properties()
        for prop, claimed in claimed_properties.items():
            if prop in results and claimed and not results[prop]:
                logger.warning(
                    f"Metric claims to be {prop} but validation failed")

        return results

    @classmethod
    def register_metric(cls, name: str) -> None:
        """
        Register this metric class in the global registry.

        Args:
            name: Unique name for the metric
        """
        MetricRegistry.register(name, cls)
