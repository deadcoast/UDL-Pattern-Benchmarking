"""
Evaluation and statistical analysis components.

This module provides:
- Performance metrics computation
- Statistical significance testing
- Result visualization
- UDL comparison and ranking
"""

from .comparison import ComparisonEngine, ComparisonResult, RankingResult, ComparisonSummary

__all__ = [
    "ComparisonEngine",
    "ComparisonResult", 
    "RankingResult",
    "ComparisonSummary",
]
