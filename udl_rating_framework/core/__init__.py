"""
Core components of the UDL Rating Framework.

This module contains the fundamental building blocks:
- UDL representation and parsing
- Quality metrics
- Aggregation and confidence calculation
"""

__all__ = [
    "UDLRepresentation",
    "MetricAggregator",
    "ConfidenceCalculator",
]


def __getattr__(name):
    """Lazy import of core classes."""
    if name == "UDLRepresentation":
        from udl_rating_framework.core.representation import UDLRepresentation
        return UDLRepresentation
    elif name == "MetricAggregator":
        from udl_rating_framework.core.aggregation import MetricAggregator
        return MetricAggregator
    elif name == "ConfidenceCalculator":
        from udl_rating_framework.core.confidence import ConfidenceCalculator
        return ConfidenceCalculator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
