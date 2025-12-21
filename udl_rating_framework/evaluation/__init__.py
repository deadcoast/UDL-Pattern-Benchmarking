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
    ComparisonSummary,
    RankingResult,
)
from .evaluation_suite import EvaluationResult, EvaluationSuite

__all__ = [
    "ComparisonEngine",
    "ComparisonResult",
    "RankingResult",
    "ComparisonSummary",
    "EvaluationSuite",
    "EvaluationResult",
]
