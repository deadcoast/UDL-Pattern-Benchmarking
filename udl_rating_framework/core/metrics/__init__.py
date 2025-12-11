"""
Quality metrics for UDL evaluation.

Each metric is a mathematical function f: UDL_Space → [0,1]
with proven properties (boundedness, determinism, computability).
"""

from udl_rating_framework.core.metrics.base import QualityMetric

__all__ = [
    "QualityMetric",
]
