"""
Quality metrics for UDL evaluation.

Each metric is a mathematical function f: UDL_Space → [0,1]
with proven properties (boundedness, determinism, computability).
"""

from udl_rating_framework.core.metrics.base import QualityMetric, MetricRegistry
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.metrics.completeness import CompletenessMetric
from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
from udl_rating_framework.core.metrics.structural_coherence import (
    StructuralCoherenceMetric,
)

# Advanced metrics (Task 29)
from udl_rating_framework.core.metrics.semantic_similarity import (
    SemanticSimilarityMetric,
)
from udl_rating_framework.core.metrics.readability import ReadabilityMetric
from udl_rating_framework.core.metrics.maintainability import MaintainabilityMetric
from udl_rating_framework.core.metrics.cross_language_compatibility import (
    CrossLanguageCompatibilityMetric,
)
from udl_rating_framework.core.metrics.evolution_tracking import EvolutionTrackingMetric


def _register_default_metrics():
    """Register all default metrics in the MetricRegistry."""
    # Core metrics
    MetricRegistry.register("consistency", ConsistencyMetric)
    MetricRegistry.register("completeness", CompletenessMetric)
    MetricRegistry.register("expressiveness", ExpressivenessMetric)
    MetricRegistry.register("structural_coherence", StructuralCoherenceMetric)

    # Advanced metrics
    MetricRegistry.register("semantic_similarity", SemanticSimilarityMetric)
    MetricRegistry.register("readability", ReadabilityMetric)
    MetricRegistry.register("maintainability", MaintainabilityMetric)
    MetricRegistry.register(
        "cross_language_compatibility", CrossLanguageCompatibilityMetric
    )
    MetricRegistry.register("evolution_tracking", EvolutionTrackingMetric)


# Register metrics on module import
_register_default_metrics()


__all__ = [
    "QualityMetric",
    "MetricRegistry",
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
    # Registration function
    "_register_default_metrics",
]
