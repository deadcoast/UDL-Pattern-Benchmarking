"""
UDL Rating Framework

A mathematically-grounded system for evaluating the quality of User Defined Languages (UDLs).
Built on the Continuous Thought Machine (CTM) architecture.
"""

__version__ = "0.1.0"
__author__ = "UDL Rating Framework Team"

# Lazy imports to avoid dependency issues during setup
__all__ = [
    "UDLRepresentation",
    "MetricAggregator",
    "ConfidenceCalculator",
]


def __getattr__(name):
    """Lazy import of main classes."""
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
