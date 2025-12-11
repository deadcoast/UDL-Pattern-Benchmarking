"""
Quality metrics for UDL evaluation.

Each metric is a mathematical function f: UDL_Space → [0,1]
with proven properties (boundedness, determinism, computability).
"""

from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.metrics.completeness import CompletenessMetric
from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
from udl_rating_framework.core.metrics.structural_coherence import StructuralCoherenceMetric

__all__ = [
    "QualityMetric",
    "ConsistencyMetric",
    "CompletenessMetric",
    "ExpressivenessMetric",
    "StructuralCoherenceMetric",
]
