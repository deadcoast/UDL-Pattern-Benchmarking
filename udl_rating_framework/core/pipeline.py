"""
Rating computation pipeline.

Orchestrates metric computation, aggregation, and report generation.
"""

import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.core.metrics.base import QualityMetric, MetricRegistry
from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.core.confidence import ConfidenceCalculator
from udl_rating_framework.core.caching import get_udl_cache, get_metric_cache

logger = logging.getLogger(__name__)


@dataclass
class ComputationStep:
    """Single step in computation trace."""

    step_number: int
    operation: str  # Description of operation
    formula: str  # LaTeX formula
    inputs: Dict[str, Any]
    output: Any
    intermediate_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Complete quality assessment report."""

    overall_score: float  # Q ∈ [0, 1]
    confidence: float  # C ∈ [0, 1]
    metric_scores: Dict[str, float]  # Individual metric values
    metric_formulas: Dict[str, str]  # LaTeX formulas
    computation_trace: List[ComputationStep]  # Step-by-step trace
    error_bounds: Dict[str, Tuple[float, float]]  # (lower, upper) bounds
    timestamp: datetime
    udl_file: str
    errors: List[str] = field(default_factory=list)  # Any errors encountered
    warnings: List[str] = field(default_factory=list)  # Any warnings


class RatingPipeline:
    """
    Orchestrates metric computation and rating generation.

    The pipeline:
    1. Computes each metric independently with error handling
    2. Aggregates results using weighted combination
    3. Computes confidence based on prediction entropy
    4. Generates comprehensive quality report with computation trace
    """

    def __init__(
        self,
        metric_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        enable_tracing: bool = True,
        enable_caching: bool = True,
    ):
        """
        Initialize rating pipeline.

        Args:
            metric_names: List of metric names to compute
            weights: Optional weights for aggregation (defaults to equal weights)
            enable_tracing: Whether to generate computation traces
            enable_caching: Whether to enable caching for performance
        """
        self.metric_names = metric_names
        self.enable_tracing = enable_tracing
        self.enable_caching = enable_caching

        # Initialize metrics from registry
        self.metrics: Dict[str, QualityMetric] = {}
        for name in metric_names:
            try:
                metric_class = MetricRegistry.get_metric(name)
                self.metrics[name] = metric_class()
                logger.info(f"Initialized metric: {name}")
            except KeyError as e:
                logger.error(f"Failed to initialize metric {name}: {e}")
                raise ValueError(f"Unknown metric: {name}")

        # Set up weights (default to equal weights)
        if weights is None:
            weight_value = 1.0 / len(metric_names)
            weights = {name: weight_value for name in metric_names}

        # Initialize aggregator
        self.aggregator = MetricAggregator(weights)

        # Initialize confidence calculator
        self.confidence_calculator = ConfidenceCalculator()

        # Initialize caches if enabled
        if self.enable_caching:
            self.udl_cache = get_udl_cache()
            self.metric_cache = get_metric_cache()
        else:
            self.udl_cache = None
            self.metric_cache = None

        logger.info(
            f"Initialized pipeline with metrics: {metric_names}, caching: {enable_caching}"
        )

    def compute_rating(self, udl: UDLRepresentation) -> QualityReport:
        """
        Compute complete quality rating for a UDL.

        Args:
            udl: UDLRepresentation to analyze

        Returns:
            QualityReport with all computed metrics and metadata
        """
        logger.info(f"Computing rating for UDL: {udl.file_path}")

        # Initialize report
        report = QualityReport(
            overall_score=0.0,
            confidence=0.0,
            metric_scores={},
            metric_formulas={},
            computation_trace=[],
            error_bounds={},
            timestamp=datetime.now(),
            udl_file=udl.file_path,
        )

        step_counter = 1

        try:
            # Step 1: Compute individual metrics independently
            if self.enable_tracing:
                step = ComputationStep(
                    step_number=step_counter,
                    operation="Initialize metric computation",
                    formula="",
                    inputs={
                        "udl_file": udl.file_path,
                        "metrics": list(self.metric_names),
                    },
                    output="Starting independent metric computation",
                )
                report.computation_trace.append(step)
                step_counter += 1

            metric_values = {}
            metric_errors = {}

            # Compute UDL hash for caching if enabled
            udl_hash = None
            if self.enable_caching and self.metric_cache:
                udl_hash = self._compute_udl_hash(udl)

            for name, metric in self.metrics.items():
                try:
                    logger.debug(f"Computing metric: {name}")

                    # Check cache first if enabled
                    value = None
                    if self.enable_caching and self.metric_cache and udl_hash:
                        value = self.metric_cache.get_metric(udl_hash, name)
                        if value is not None:
                            logger.debug(f"Cache hit for metric {name}")

                    # Compute metric value if not cached
                    if value is None:
                        value = metric.compute(udl)

                        # Cache the result if enabled
                        if self.enable_caching and self.metric_cache and udl_hash:
                            self.metric_cache.put_metric(udl_hash, name, value)

                    # Validate boundedness
                    if not (0.0 <= value <= 1.0):
                        warning = f"Metric {name} produced out-of-bounds value: {value}"
                        logger.warning(warning)
                        report.warnings.append(warning)
                        value = max(0.0, min(1.0, value))  # Clamp to [0,1]

                    metric_values[name] = value
                    report.metric_scores[name] = value
                    report.metric_formulas[name] = metric.get_formula()

                    # Add computation step
                    if self.enable_tracing:
                        step = ComputationStep(
                            step_number=step_counter,
                            operation=f"Compute {name} metric",
                            formula=metric.get_formula(),
                            inputs={"udl": str(udl.file_path)},
                            output=value,
                            intermediate_values={name: value},
                        )
                        report.computation_trace.append(step)
                        step_counter += 1

                    logger.debug(f"Metric {name}: {value}")

                except Exception as e:
                    error_msg = f"Failed to compute metric {name}: {str(e)}"
                    logger.error(error_msg)
                    report.errors.append(error_msg)
                    metric_errors[name] = str(e)

                    # Use default value for failed metrics
                    metric_values[name] = 0.0
                    report.metric_scores[name] = 0.0

                    if self.enable_tracing:
                        step = ComputationStep(
                            step_number=step_counter,
                            operation=f"Error computing {name} metric",
                            formula="N/A",
                            inputs={"udl": str(udl.file_path)},
                            output=f"ERROR: {str(e)}",
                            intermediate_values={name: 0.0},
                        )
                        report.computation_trace.append(step)
                        step_counter += 1

            # Step 2: Aggregate metrics
            if metric_values:
                try:
                    overall_score = self.aggregator.aggregate(metric_values)
                    report.overall_score = overall_score

                    if self.enable_tracing:
                        step = ComputationStep(
                            step_number=step_counter,
                            operation="Aggregate metrics",
                            formula=r"Q = \sum_{i} w_i \cdot m_i",
                            inputs={
                                "weights": dict(self.aggregator.weights),
                                "metric_values": metric_values,
                            },
                            output=overall_score,
                            intermediate_values={"overall_score": overall_score},
                        )
                        report.computation_trace.append(step)
                        step_counter += 1

                    logger.debug(f"Overall score: {overall_score}")

                except Exception as e:
                    error_msg = f"Failed to aggregate metrics: {str(e)}"
                    logger.error(error_msg)
                    report.errors.append(error_msg)
                    report.overall_score = 0.0

            # Step 3: Compute confidence
            try:
                # Create prediction distribution from metric values
                # Use metric values as a probability distribution
                if metric_values:
                    values = list(metric_values.values())
                    # Normalize to create probability distribution
                    total = sum(values) if sum(values) > 0 else 1.0
                    prediction_probs = [v / total for v in values]

                    confidence = self.confidence_calculator.compute_confidence(
                        prediction_probs
                    )
                    report.confidence = confidence

                    if self.enable_tracing:
                        step = ComputationStep(
                            step_number=step_counter,
                            operation="Compute confidence",
                            formula=r"C = 1 - \frac{H(p)}{H_{max}}",
                            inputs={"prediction_probs": prediction_probs},
                            output=confidence,
                            intermediate_values={"confidence": confidence},
                        )
                        report.computation_trace.append(step)
                        step_counter += 1

                    logger.debug(f"Confidence: {confidence}")
                else:
                    report.confidence = 0.0

            except Exception as e:
                error_msg = f"Failed to compute confidence: {str(e)}"
                logger.error(error_msg)
                report.errors.append(error_msg)
                report.confidence = 0.0

            # Step 4: Compute error bounds (simple approach)
            try:
                for name, value in metric_values.items():
                    # Simple error bounds based on metric properties
                    # In a real implementation, this could be more sophisticated
                    error_margin = 0.05  # 5% error margin
                    lower_bound = max(0.0, value - error_margin)
                    upper_bound = min(1.0, value + error_margin)
                    report.error_bounds[name] = (lower_bound, upper_bound)

                # Overall score error bounds
                if report.overall_score > 0:
                    error_margin = 0.05
                    lower_bound = max(0.0, report.overall_score - error_margin)
                    upper_bound = min(1.0, report.overall_score + error_margin)
                    report.error_bounds["overall"] = (lower_bound, upper_bound)

            except Exception as e:
                error_msg = f"Failed to compute error bounds: {str(e)}"
                logger.warning(error_msg)
                report.warnings.append(error_msg)

        except Exception as e:
            error_msg = f"Critical error in rating pipeline: {str(e)}"
            logger.error(error_msg)
            report.errors.append(error_msg)
            # Ensure report has valid default values
            report.overall_score = 0.0
            report.confidence = 0.0

        logger.info(f"Rating computation completed for {udl.file_path}")
        logger.info(
            f"Overall score: {report.overall_score:.3f}, Confidence: {report.confidence:.3f}"
        )

        return report

    def compute_batch_ratings(
        self, udls: List[UDLRepresentation]
    ) -> List[QualityReport]:
        """
        Compute ratings for multiple UDLs.

        Args:
            udls: List of UDLRepresentation objects to analyze

        Returns:
            List of QualityReport objects
        """
        logger.info(f"Computing batch ratings for {len(udls)} UDLs")

        reports = []
        for i, udl in enumerate(udls):
            try:
                logger.debug(f"Processing UDL {i + 1}/{len(udls)}: {udl.file_path}")
                report = self.compute_rating(udl)
                reports.append(report)
            except Exception as e:
                error_msg = f"Failed to process UDL {udl.file_path}: {str(e)}"
                logger.error(error_msg)

                # Create error report
                error_report = QualityReport(
                    overall_score=0.0,
                    confidence=0.0,
                    metric_scores={},
                    metric_formulas={},
                    computation_trace=[],
                    error_bounds={},
                    timestamp=datetime.now(),
                    udl_file=udl.file_path,
                    errors=[error_msg],
                )
                reports.append(error_report)

        logger.info(f"Batch processing completed. {len(reports)} reports generated.")
        return reports

    def get_available_metrics(self) -> Dict[str, str]:
        """
        Get information about available metrics.

        Returns:
            Dict mapping metric names to their formulas
        """
        available = {}
        for name in MetricRegistry.list_metrics():
            try:
                metric_class = MetricRegistry.get_metric(name)
                metric = metric_class()
                available[name] = metric.get_formula()
            except Exception as e:
                logger.warning(f"Could not get formula for metric {name}: {e}")
                available[name] = "Formula unavailable"

        return available

    def validate_pipeline(self, test_udl: UDLRepresentation) -> Dict[str, bool]:
        """
        Validate pipeline functionality with a test UDL.

        Args:
            test_udl: UDL to use for validation

        Returns:
            Dict mapping validation checks to results
        """
        results = {}

        try:
            # Test metric computation
            for name, metric in self.metrics.items():
                try:
                    value = metric.compute(test_udl)
                    results[f"{name}_computable"] = True
                    results[f"{name}_bounded"] = bool(0.0 <= value <= 1.0)
                except Exception:
                    results[f"{name}_computable"] = False
                    results[f"{name}_bounded"] = False

            # Test aggregation
            try:
                test_values = {name: 0.5 for name in self.metric_names}
                aggregated = self.aggregator.aggregate(test_values)
                results["aggregation_works"] = True
                results["aggregation_bounded"] = bool(0.0 <= aggregated <= 1.0)
            except Exception:
                results["aggregation_works"] = False
                results["aggregation_bounded"] = False

            # Test confidence computation
            try:
                test_probs = [0.5, 0.3, 0.2]
                confidence = self.confidence_calculator.compute_confidence(test_probs)
                results["confidence_works"] = True
                results["confidence_bounded"] = bool(0.0 <= confidence <= 1.0)
            except Exception:
                results["confidence_works"] = False
                results["confidence_bounded"] = False

            # Test full pipeline
            try:
                report = self.compute_rating(test_udl)
                results["pipeline_works"] = True
                results["report_complete"] = (
                    hasattr(report, "overall_score")
                    and hasattr(report, "confidence")
                    and hasattr(report, "metric_scores")
                )
            except Exception:
                results["pipeline_works"] = False
                results["report_complete"] = False

        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            results["validation_error"] = str(e)

        return results

    def _compute_udl_hash(self, udl: UDLRepresentation) -> str:
        """
        Compute hash of UDL for caching.

        Args:
            udl: UDL representation

        Returns:
            SHA-256 hash of UDL content
        """
        content = udl.source_text + str(udl.file_path)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def clear_caches(self) -> None:
        """Clear all caches used by this pipeline."""
        if self.enable_caching:
            if self.udl_cache:
                self.udl_cache.clear()
            if self.metric_cache:
                self.metric_cache.clear()
            logger.info("Cleared pipeline caches")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enable_caching:
            return {"caching_enabled": False}

        stats = {"caching_enabled": True}

        if self.udl_cache:
            stats["udl_cache"] = self.udl_cache.get_stats()

        if self.metric_cache:
            stats["metric_cache"] = self.metric_cache.get_stats()

        return stats
