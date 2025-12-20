"""
Evaluation and statistical analysis components.

This module provides:
- Performance metrics computation
- Statistical significance testing
- Result visualization
- UDL comparison and ranking
- Comprehensive evaluation suite with cross-validation and statistical analysis
"""

from .comparison import (
    ComparisonEngine,
    ComparisonResult,
    RankingResult,
    ComparisonSummary,
)
from .evaluation_suite import EvaluationSuite, EvaluationResult

__all__ = [
    "ComparisonEngine",
    "ComparisonResult",
    "RankingResult",
    "ComparisonSummary",
    "EvaluationSuite",
    "EvaluationResult",
]
