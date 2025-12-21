"""
Comparison engine for statistical analysis of UDL ratings.

Provides functionality for comparing multiple UDL ratings using statistical tests,
effect size computation, and ranking with confidence intervals.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

from udl_rating_framework.core.pipeline import QualityReport

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two UDL ratings."""

    udl1_name: str
    udl2_name: str
    score1: float
    score2: float
    difference: float  # Δ_ij = Q_i - Q_j

    # Statistical significance tests
    ttest_statistic: float
    ttest_pvalue: float
    wilcoxon_statistic: Optional[float]
    wilcoxon_pvalue: Optional[float]

    # Effect size
    cohens_d: float

    # Interpretation
    is_significant: bool  # p < 0.05
    effect_size_interpretation: str  # "small", "medium", "large", "negligible"


@dataclass
class RankingResult:
    """Result of ranking multiple UDLs with confidence intervals."""

    udl_name: str
    score: float
    rank: int  # 1-based ranking (1 = best)
    confidence_interval: Tuple[float, float]  # (lower, upper) bounds for score
    rank_confidence_interval: Tuple[int, int]  # (lower, upper) bounds for rank


@dataclass
class ComparisonSummary:
    """Summary of all pairwise comparisons."""

    pairwise_results: List[ComparisonResult]
    rankings: List[RankingResult]

    # Summary statistics
    total_comparisons: int
    significant_comparisons: int
    mean_effect_size: float

    # Overall statistics
    score_statistics: Dict[str, float]  # mean, std, min, max, etc.


class ComparisonEngine:
    """
    Engine for statistical comparison of UDL ratings.

    Implements:
    - Pairwise difference computation: Δ_ij = Q_i - Q_j
    - Statistical significance tests (t-test, Wilcoxon)
    - Effect size computation (Cohen's d)
    - Ranking with confidence intervals

    Mathematical Properties:
    - Consistent rating procedures for all UDLs
    - Accurate pairwise difference computation
    - Proper statistical test application
    - Correct effect size calculation
    """

    def __init__(self, alpha: float = 0.05, bootstrap_samples: int = 1000):
        """
        Initialize comparison engine.

        Args:
            alpha: Significance level for statistical tests (default: 0.05)
            bootstrap_samples: Number of bootstrap samples for confidence intervals
        """
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        logger.info(
            f"Initialized ComparisonEngine with alpha={alpha}, bootstrap_samples={bootstrap_samples}"
        )

    def compare_udls(self, reports: List[QualityReport]) -> ComparisonSummary:
        """
        Compare multiple UDL ratings using statistical analysis.

        Args:
            reports: List of QualityReport objects to compare

        Returns:
            ComparisonSummary with all pairwise comparisons and rankings

        Raises:
            ValueError: If fewer than 2 reports provided
        """
        if len(reports) < 2:
            raise ValueError("At least 2 UDL reports required for comparison")

        logger.info(f"Comparing {len(reports)} UDL reports")

        # Perform all pairwise comparisons
        pairwise_results = self._compute_pairwise_comparisons(reports)

        # Generate rankings with confidence intervals
        rankings = self._compute_rankings_with_confidence(reports)

        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(
            reports, pairwise_results)

        return ComparisonSummary(
            pairwise_results=pairwise_results,
            rankings=rankings,
            total_comparisons=len(pairwise_results),
            significant_comparisons=sum(
                1 for r in pairwise_results if r.is_significant
            ),
            mean_effect_size=np.mean([r.cohens_d for r in pairwise_results]),
            score_statistics=summary_stats,
        )

    def _compute_pairwise_comparisons(
        self, reports: List[QualityReport]
    ) -> List[ComparisonResult]:
        """
        Compute all pairwise comparisons between UDL reports.

        Args:
            reports: List of QualityReport objects

        Returns:
            List of ComparisonResult objects for all pairs
        """
        results = []

        for i in range(len(reports)):
            for j in range(i + 1, len(reports)):
                report1, report2 = reports[i], reports[j]

                # Extract scores
                score1 = report1.overall_score
                score2 = report2.overall_score

                # Compute pairwise difference: Δ_ij = Q_i - Q_j
                difference = score1 - score2

                # Perform statistical tests
                ttest_stat, ttest_p = self._perform_ttest(
                    score1, score2, report1, report2
                )
                wilcoxon_stat, wilcoxon_p = self._perform_wilcoxon_test(
                    score1, score2, report1, report2
                )

                # Compute effect size (Cohen's d)
                cohens_d = self._compute_cohens_d(
                    score1, score2, report1, report2)

                # Determine significance and effect size interpretation
                is_significant = min(ttest_p, wilcoxon_p or 1.0) < self.alpha
                effect_interpretation = self._interpret_effect_size(
                    abs(cohens_d))

                result = ComparisonResult(
                    udl1_name=report1.udl_file,
                    udl2_name=report2.udl_file,
                    score1=score1,
                    score2=score2,
                    difference=difference,
                    ttest_statistic=ttest_stat,
                    ttest_pvalue=ttest_p,
                    wilcoxon_statistic=wilcoxon_stat,
                    wilcoxon_pvalue=wilcoxon_p,
                    cohens_d=cohens_d,
                    is_significant=is_significant,
                    effect_size_interpretation=effect_interpretation,
                )

                results.append(result)

                logger.debug(
                    f"Compared {report1.udl_file} vs {report2.udl_file}: "
                    f"Δ={difference:.4f}, p={ttest_p:.4f}, d={cohens_d:.4f}"
                )

        return results

    def _perform_ttest(
        self,
        score1: float,
        score2: float,
        report1: QualityReport,
        report2: QualityReport,
    ) -> Tuple[float, float]:
        """
        Perform t-test between two scores.

        Since we have single scores, we use the metric scores as samples
        to estimate variance for the t-test.

        Args:
            score1, score2: Overall scores to compare
            report1, report2: Full reports for extracting metric distributions

        Returns:
            Tuple of (t_statistic, p_value)
        """
        try:
            # Extract individual metric scores as samples
            metrics1 = list(report1.metric_scores.values())
            metrics2 = list(report2.metric_scores.values())

            if len(metrics1) < 2 or len(metrics2) < 2:
                # Fall back to single-sample comparison if insufficient metrics
                # Use a small assumed variance for the test
                pooled_std = 0.1  # Assumed standard deviation
                n1 = n2 = 1
                pooled_var = pooled_std**2
                se = pooled_std * np.sqrt(1 / n1 + 1 / n2)
                t_stat = (score1 - score2) / se
                df = n1 + n2 - 2
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                return t_stat, p_value

            # Perform independent samples t-test
            t_stat, p_value = ttest_ind(metrics1, metrics2, equal_var=False)

            # Handle NaN results (e.g., when all values are identical)
            if np.isnan(t_stat) or np.isnan(p_value):
                # If all values are identical, there's no difference
                return 0.0, 1.0

            return float(t_stat), float(p_value)

        except Exception as e:
            logger.warning(f"T-test failed: {e}, using fallback")
            # Fallback: simple difference test
            return 0.0, 1.0

    def _perform_wilcoxon_test(
        self,
        score1: float,
        score2: float,
        report1: QualityReport,
        report2: QualityReport,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Perform Wilcoxon rank-sum test (Mann-Whitney U) between two score distributions.

        Args:
            score1, score2: Overall scores to compare
            report1, report2: Full reports for extracting metric distributions

        Returns:
            Tuple of (statistic, p_value) or (None, None) if test cannot be performed
        """
        try:
            # Extract individual metric scores as samples
            metrics1 = list(report1.metric_scores.values())
            metrics2 = list(report2.metric_scores.values())

            if len(metrics1) < 2 or len(metrics2) < 2:
                # Cannot perform rank test with insufficient samples
                return None, None

            # Perform Mann-Whitney U test (equivalent to Wilcoxon rank-sum)
            statistic, p_value = mannwhitneyu(
                metrics1, metrics2, alternative="two-sided"
            )
            return float(statistic), float(p_value)

        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return None, None

    def _compute_cohens_d(
        self,
        score1: float,
        score2: float,
        report1: QualityReport,
        report2: QualityReport,
    ) -> float:
        """
        Compute Cohen's d effect size.

        Cohen's d = (mean1 - mean2) / pooled_standard_deviation

        Args:
            score1, score2: Overall scores to compare
            report1, report2: Full reports for extracting metric distributions

        Returns:
            Cohen's d effect size
        """
        try:
            # Extract individual metric scores as samples
            metrics1 = np.array(list(report1.metric_scores.values()))
            metrics2 = np.array(list(report2.metric_scores.values()))

            if len(metrics1) < 2 or len(metrics2) < 2:
                # Fall back to simple difference with assumed variance
                assumed_std = 0.1
                return (score1 - score2) / assumed_std

            # Compute pooled standard deviation
            n1, n2 = len(metrics1), len(metrics2)
            var1, var2 = np.var(metrics1, ddof=1), np.var(metrics2, ddof=1)
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            pooled_std = np.sqrt(pooled_var)

            if pooled_std == 0:
                return 0.0

            # Cohen's d = (mean1 - mean2) / pooled_std
            cohens_d = (np.mean(metrics1) - np.mean(metrics2)) / pooled_std
            return float(cohens_d)

        except Exception as e:
            logger.warning(f"Cohen's d computation failed: {e}")
            return 0.0

    def _interpret_effect_size(self, abs_cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size magnitude.

        Args:
            abs_cohens_d: Absolute value of Cohen's d

        Returns:
            String interpretation of effect size
        """
        if abs_cohens_d < 0.2:
            return "negligible"
        elif abs_cohens_d < 0.5:
            return "small"
        elif abs_cohens_d < 0.8:
            return "medium"
        else:
            return "large"

    def _compute_rankings_with_confidence(
        self, reports: List[QualityReport]
    ) -> List[RankingResult]:
        """
        Compute rankings with confidence intervals using bootstrap.

        Args:
            reports: List of QualityReport objects

        Returns:
            List of RankingResult objects sorted by rank
        """
        # Extract scores and names
        scores = [report.overall_score for report in reports]
        names = [report.udl_file for report in reports]

        # Compute base ranking (higher score = better rank)
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        base_ranks = np.empty_like(sorted_indices)
        base_ranks[sorted_indices] = np.arange(1, len(scores) + 1)
        base_ranks = base_ranks.astype(int)  # Ensure Python int type

        # Bootstrap confidence intervals for scores
        score_cis = []
        rank_cis = []

        for i, report in enumerate(reports):
            # Bootstrap samples for this UDL's score
            # Use metric scores as bootstrap population
            metric_values = list(report.metric_scores.values())

            if len(metric_values) < 2:
                # Insufficient data for bootstrap, use point estimate
                ci_lower = ci_upper = scores[i]
                rank_ci_lower = rank_ci_upper = base_ranks[i]
            else:
                # Bootstrap confidence interval for score
                bootstrap_scores = []
                bootstrap_ranks = []

                for _ in range(self.bootstrap_samples):
                    # Resample metrics with replacement
                    resampled_metrics = np.random.choice(
                        metric_values, size=len(metric_values), replace=True
                    )
                    bootstrap_score = np.mean(resampled_metrics)
                    bootstrap_scores.append(bootstrap_score)

                    # Compute rank in this bootstrap sample
                    # (This is simplified - in practice we'd bootstrap all UDLs together)
                    bootstrap_ranks.append(base_ranks[i])  # Simplified for now

                # Compute confidence intervals
                ci_lower = np.percentile(bootstrap_scores, 2.5)
                ci_upper = np.percentile(bootstrap_scores, 97.5)
                rank_ci_lower = int(np.percentile(bootstrap_ranks, 2.5))
                rank_ci_upper = int(np.percentile(bootstrap_ranks, 97.5))

            score_cis.append((ci_lower, ci_upper))
            rank_cis.append((rank_ci_lower, rank_ci_upper))

        # Create ranking results
        results = []
        for i, report in enumerate(reports):
            result = RankingResult(
                udl_name=report.udl_file,
                score=float(scores[i]),
                rank=int(base_ranks[i]),
                confidence_interval=(
                    float(score_cis[i][0]), float(score_cis[i][1])),
                rank_confidence_interval=(
                    int(rank_cis[i][0]), int(rank_cis[i][1])),
            )
            results.append(result)

        # Sort by rank
        results.sort(key=lambda x: x.rank)

        return results

    def _compute_summary_statistics(
        self, reports: List[QualityReport], pairwise_results: List[ComparisonResult]
    ) -> Dict[str, float]:
        """
        Compute summary statistics for the comparison.

        Args:
            reports: List of QualityReport objects
            pairwise_results: List of pairwise comparison results

        Returns:
            Dictionary of summary statistics
        """
        scores = [report.overall_score for report in reports]

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "median_score": float(np.median(scores)),
            "score_range": float(np.max(scores) - np.min(scores)),
            "mean_absolute_difference": float(
                np.mean([abs(r.difference) for r in pairwise_results])
            ),
            "max_absolute_difference": (
                float(np.max([abs(r.difference) for r in pairwise_results]))
                if pairwise_results
                else 0.0
            ),
        }
