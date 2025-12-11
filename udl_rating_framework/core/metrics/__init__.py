"""
Quality metrics for UDL evaluation.

Each metric is a mathematical function f: UDL_Space → [0,1]
with proven properties (boundedness, determinism, computability).
"""

from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.metrics.completeness import CompletenessMetric
from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
from udl_rating_framework.core.metrics.structural_coherence import (
    StructuralCoherenceMetric,
)
# Advanced metrics (Task 29)
from udl_rating_framework.core.metrics.semantic_similarity import SemanticSimilarityMetric
from udl_rating_framework.core.metrics.readability import ReadabilityMetric
from udl_rating_framework.core.metrics.maintainability import MaintainabilityMetric
from udl_rating_framework.core.metrics.cross_language_compatibility import CrossLanguageCompatibilityMetric
from udl_rating_framework.core.metrics.evolution_tracking import EvolutionTrackingMetric

__all__ = [
    "QualityMetric",
    "ConsistencyMetric",
    "CompletenessMetric",
    "ExpressivenessMetric",
    "StructuralCoherenceMetric",
    # Advanced metrics
    "SemanticSimilarityMetric",
    "ReadabilityMetric",
    "MaintainabilityMetric",
    "CrossLanguageCompatibilityMetric",
    "EvolutionTrackingMetric",
]
