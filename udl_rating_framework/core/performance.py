"""
Comprehensive performance module for UDL Rating Framework.

Integrates all performance and scalability improvements including distributed
computing, GPU acceleration, streaming processing, memory mapping, and
incremental computation.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.core.pipeline import QualityReport

# Import metrics to ensure they are registered in the MetricRegistry
import udl_rating_framework.core.metrics  # noqa: F401

# Import performance modules
try:
    from udl_rating_framework.core.distributed import DistributedBatchProcessor

    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

try:
    from udl_rating_framework.core.gpu_acceleration import GPUAcceleratedProcessor

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from udl_rating_framework.core.streaming import StreamingProcessor, process_large_file
from udl_rating_framework.core.memory_mapping import (
    MemoryMappedUDLProcessor,
)
from udl_rating_framework.core.incremental import (
    IncrementalProcessor,
)
from udl_rating_framework.core.multiprocessing import (
    ParallelProcessor,
    process_udl_files_parallel,
)

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Processing strategy options."""

    AUTO = "auto"  # Automatic strategy selection
    SEQUENTIAL = "sequential"  # Sequential processing
    PARALLEL = "parallel"  # Multiprocessing
    DISTRIBUTED = "distributed"  # Distributed computing
    GPU = "gpu"  # GPU acceleration
    STREAMING = "streaming"  # Streaming for large files
    MEMORY_MAPPED = "memory_mapped"  # Memory-mapped files
    INCREMENTAL = "incremental"  # Incremental computation


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    # Strategy selection
    strategy: ProcessingStrategy = ProcessingStrategy.AUTO
    fallback_strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL

    # Resource limits
    max_memory_mb: int = 1024
    max_workers: Optional[int] = None
    max_file_size_mb: int = 100

    # Distributed computing
    distributed_backend: str = "auto"  # 'ray', 'dask', 'auto'
    cluster_address: Optional[str] = None

    # GPU acceleration
    enable_gpu: bool = True
    gpu_batch_size: int = 32
    gpu_max_sequence_length: int = 512

    # Streaming processing
    streaming_chunk_size: int = 1024 * 1024  # 1MB
    streaming_overlap_size: int = 1024  # 1KB

    # Memory mapping
    memory_map_threshold_mb: int = 50
    memory_map_page_size: int = 4096

    # Incremental computation
    enable_incremental: bool = True
    incremental_cache_dir: Optional[Path] = None

    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24

    # Progress tracking
    enable_progress: bool = True
    progress_callback: Optional[Callable[[int, int], None]] = None


@dataclass
class PerformanceResult:
    """Result of performance-optimized processing."""

    strategy_used: ProcessingStrategy
    total_files: int
    successful_files: int
    failed_files: int
    processing_time: float
    throughput: float  # files per second
    memory_peak_mb: float
    cache_hit_ratio: float
    reports: List[QualityReport]
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceOptimizer:
    """
    Performance optimizer that automatically selects the best processing
    strategy based on workload characteristics and available resources.
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        """
        Initialize performance optimizer.

        Args:
            config: Performance configuration
        """
        self.config = config or PerformanceConfig()

        # Initialize processors
        self.processors = {}
        self._initialize_processors()

        # Performance tracking
        self.performance_history = []
        self.strategy_performance = {}

    def _initialize_processors(self) -> None:
        """Initialize available processors."""
        # Always available processors
        self.processors[ProcessingStrategy.PARALLEL] = ParallelProcessor(
            max_workers=self.config.max_workers
        )

        self.processors[ProcessingStrategy.STREAMING] = StreamingProcessor()

        self.processors[ProcessingStrategy.MEMORY_MAPPED] = MemoryMappedUDLProcessor()

        self.processors[ProcessingStrategy.INCREMENTAL] = IncrementalProcessor(
            cache_dir=self.config.incremental_cache_dir
        )

        # Optional processors
        if DISTRIBUTED_AVAILABLE:
            try:
                self.processors[ProcessingStrategy.DISTRIBUTED] = (
                    DistributedBatchProcessor(
                        backend=self.config.distributed_backend,
                        cluster_address=self.config.cluster_address,
                        num_workers=self.config.max_workers,
                    )
                )
                logger.info("Distributed processing available")
            except Exception as e:
                logger.warning(f"Distributed processing not available: {e}")

        if GPU_AVAILABLE and self.config.enable_gpu:
            try:
                self.processors[ProcessingStrategy.GPU] = GPUAcceleratedProcessor(
                    batch_size=self.config.gpu_batch_size,
                    max_sequence_length=self.config.gpu_max_sequence_length,
                )
                logger.info("GPU acceleration available")
            except Exception as e:
                logger.warning(f"GPU acceleration not available: {e}")

    def process_files(
        self,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> PerformanceResult:
        """
        Process files using optimal performance strategy.

        Args:
            file_paths: List of file paths to process
            metric_names: List of metrics to compute
            weights: Optional metric weights

        Returns:
            PerformanceResult with processing information
        """
        start_time = time.time()

        logger.info(f"Processing {len(file_paths)} files with performance optimization")

        # Analyze workload characteristics
        workload_info = self._analyze_workload(file_paths)

        # Select optimal strategy
        strategy = self._select_strategy(workload_info)

        logger.info(f"Selected processing strategy: {strategy.value}")

        # Process files with selected strategy
        try:
            result = self._process_with_strategy(
                strategy, file_paths, metric_names, weights, workload_info
            )

            # Record performance
            self._record_performance(strategy, result)

            return result

        except Exception as e:
            logger.error(f"Processing failed with strategy {strategy.value}: {e}")

            # Try fallback strategy
            if strategy != self.config.fallback_strategy:
                logger.info(
                    f"Trying fallback strategy: {self.config.fallback_strategy.value}"
                )

                try:
                    result = self._process_with_strategy(
                        self.config.fallback_strategy,
                        file_paths,
                        metric_names,
                        weights,
                        workload_info,
                    )
                    result.strategy_used = self.config.fallback_strategy
                    return result

                except Exception as fallback_error:
                    logger.error(f"Fallback strategy also failed: {fallback_error}")

            # Create error result
            processing_time = time.time() - start_time
            return PerformanceResult(
                strategy_used=strategy,
                total_files=len(file_paths),
                successful_files=0,
                failed_files=len(file_paths),
                processing_time=processing_time,
                throughput=0.0,
                memory_peak_mb=0.0,
                cache_hit_ratio=0.0,
                reports=[],
                errors=[str(e)],
            )

    def process_directory(
        self,
        directory: Path,
        metric_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        file_patterns: List[str] = None,
    ) -> PerformanceResult:
        """
        Process directory using optimal performance strategy.

        Args:
            directory: Directory containing UDL files
            metric_names: List of metrics to compute
            weights: Optional metric weights
            file_patterns: File patterns to match

        Returns:
            PerformanceResult with processing information
        """
        from udl_rating_framework.io.file_discovery import FileDiscovery

        # Discover files
        if file_patterns is None:
            file_patterns = ["*.udl", "*.dsl", "*.grammar", "*.ebnf"]

        discovery = FileDiscovery()
        extensions = [p.replace("*", "") for p in file_patterns]
        files = discovery.discover_files(directory, extensions)

        return self.process_files(files, metric_names, weights)

    def _analyze_workload(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Analyze workload characteristics to inform strategy selection."""
        total_size = 0
        max_file_size = 0
        file_count = len(file_paths)

        # Sample files for analysis (don't read all files)
        sample_size = min(10, len(file_paths))
        sample_files = file_paths[:sample_size]

        for file_path in sample_files:
            try:
                size = file_path.stat().st_size
                total_size += size
                max_file_size = max(max_file_size, size)
            except OSError:
                pass

        avg_file_size = total_size / sample_size if sample_size > 0 else 0

        # Estimate total workload size
        estimated_total_size = avg_file_size * file_count

        return {
            "file_count": file_count,
            "avg_file_size_mb": avg_file_size / (1024 * 1024),
            "max_file_size_mb": max_file_size / (1024 * 1024),
            "estimated_total_size_mb": estimated_total_size / (1024 * 1024),
            "has_large_files": max_file_size
            > self.config.max_file_size_mb * 1024 * 1024,
            "is_large_workload": file_count > 100
            or estimated_total_size > self.config.max_memory_mb * 1024 * 1024,
        }

    def _select_strategy(self, workload_info: Dict[str, Any]) -> ProcessingStrategy:
        """Select optimal processing strategy based on workload characteristics."""
        if self.config.strategy != ProcessingStrategy.AUTO:
            return self.config.strategy

        file_count = workload_info["file_count"]
        avg_file_size_mb = workload_info["avg_file_size_mb"]
        max_file_size_mb = workload_info["max_file_size_mb"]
        has_large_files = workload_info["has_large_files"]
        is_large_workload = workload_info["is_large_workload"]

        # Strategy selection logic

        # Single large file - use streaming or memory mapping
        if file_count == 1 and max_file_size_mb > self.config.memory_map_threshold_mb:
            if max_file_size_mb > 500:  # Very large file
                return ProcessingStrategy.STREAMING
            else:
                return ProcessingStrategy.MEMORY_MAPPED

        # Large workload with many files - use distributed if available
        if is_large_workload and ProcessingStrategy.DISTRIBUTED in self.processors:
            return ProcessingStrategy.DISTRIBUTED

        # GPU acceleration for medium workloads
        if (
            ProcessingStrategy.GPU in self.processors
            and 10 <= file_count <= 1000
            and avg_file_size_mb < 10
        ):
            return ProcessingStrategy.GPU

        # Incremental processing for repeated workloads
        if self.config.enable_incremental and file_count > 5 and not has_large_files:
            return ProcessingStrategy.INCREMENTAL

        # Memory mapping for large files
        if has_large_files and max_file_size_mb > self.config.memory_map_threshold_mb:
            return ProcessingStrategy.MEMORY_MAPPED

        # Streaming for very large files
        if max_file_size_mb > 200:
            return ProcessingStrategy.STREAMING

        # Default to parallel processing
        return ProcessingStrategy.PARALLEL

    def _process_with_strategy(
        self,
        strategy: ProcessingStrategy,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]],
        workload_info: Dict[str, Any],
    ) -> PerformanceResult:
        """Process files with specific strategy."""
        start_time = time.time()

        if strategy == ProcessingStrategy.SEQUENTIAL:
            result = self._process_sequential(file_paths, metric_names, weights)

        elif strategy == ProcessingStrategy.PARALLEL:
            result = self._process_parallel(file_paths, metric_names, weights)

        elif strategy == ProcessingStrategy.DISTRIBUTED:
            result = self._process_distributed(file_paths, metric_names, weights)

        elif strategy == ProcessingStrategy.GPU:
            result = self._process_gpu(file_paths, metric_names, weights)

        elif strategy == ProcessingStrategy.STREAMING:
            result = self._process_streaming(file_paths, metric_names, weights)

        elif strategy == ProcessingStrategy.MEMORY_MAPPED:
            result = self._process_memory_mapped(file_paths, metric_names, weights)

        elif strategy == ProcessingStrategy.INCREMENTAL:
            result = self._process_incremental(file_paths, metric_names, weights)

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        result.strategy_used = strategy
        result.processing_time = time.time() - start_time
        result.throughput = (
            result.successful_files / result.processing_time
            if result.processing_time > 0
            else 0.0
        )

        return result

    def _process_sequential(
        self,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]],
    ) -> PerformanceResult:
        """Process files sequentially."""
        from udl_rating_framework.core.pipeline import RatingPipeline

        pipeline = RatingPipeline(metric_names=metric_names, weights=weights)
        reports = []
        errors = []

        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding="utf-8")
                udl = UDLRepresentation(content, str(file_path))
                report = pipeline.compute_rating(udl)
                reports.append(report)
            except Exception as e:
                errors.append(f"Error processing {file_path}: {e}")

        return PerformanceResult(
            strategy_used=ProcessingStrategy.SEQUENTIAL,
            total_files=len(file_paths),
            successful_files=len(reports),
            failed_files=len(errors),
            processing_time=0.0,  # Will be set by caller
            throughput=0.0,  # Will be set by caller
            memory_peak_mb=0.0,  # TODO: Implement memory tracking
            cache_hit_ratio=0.0,
            reports=reports,
            errors=errors,
        )

    def _process_parallel(
        self,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]],
    ) -> PerformanceResult:
        """Process files with multiprocessing."""
        reports, stats = process_udl_files_parallel(
            file_paths=file_paths,
            metric_names=metric_names,
            weights=weights,
            max_workers=self.config.max_workers,
            show_progress=self.config.enable_progress,
        )

        return PerformanceResult(
            strategy_used=ProcessingStrategy.PARALLEL,
            total_files=stats.total_files,
            successful_files=stats.successful,
            failed_files=stats.failed,
            processing_time=stats.total_time,
            throughput=(
                stats.successful / stats.total_time if stats.total_time > 0 else 0.0
            ),
            memory_peak_mb=0.0,  # TODO: Get from stats
            cache_hit_ratio=0.0,
            reports=reports,
            metadata={"stats": stats},
        )

    def _process_distributed(
        self,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]],
    ) -> PerformanceResult:
        """Process files with distributed computing."""
        if ProcessingStrategy.DISTRIBUTED not in self.processors:
            raise RuntimeError("Distributed processing not available")

        processor = self.processors[ProcessingStrategy.DISTRIBUTED]
        reports, stats = processor.process_files(
            file_paths=file_paths, metric_names=metric_names, weights=weights
        )

        return PerformanceResult(
            strategy_used=ProcessingStrategy.DISTRIBUTED,
            total_files=len(file_paths),
            successful_files=len([r for r in reports if not r.errors]),
            failed_files=len([r for r in reports if r.errors]),
            processing_time=0.0,  # Will be set by caller
            throughput=0.0,  # Will be set by caller
            memory_peak_mb=0.0,
            cache_hit_ratio=0.0,
            reports=reports,
        )

    def _process_gpu(
        self,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]],
    ) -> PerformanceResult:
        """Process files with GPU acceleration."""
        if ProcessingStrategy.GPU not in self.processors:
            raise RuntimeError("GPU processing not available")

        # Load UDL representations
        udl_representations = []
        errors = []

        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding="utf-8")
                udl = UDLRepresentation(content, str(file_path))
                udl_representations.append(udl)
            except Exception as e:
                errors.append(f"Error loading {file_path}: {e}")

        # Process with GPU
        processor = self.processors[ProcessingStrategy.GPU]
        predictions = processor.process_udls(udl_representations)

        # Convert predictions to reports (simplified)
        reports = []
        for pred in predictions:
            report = QualityReport(
                overall_score=pred["overall_score"],
                confidence=pred["confidence"],
                metric_scores={},  # TODO: Include individual metrics
                metric_formulas={},
                computation_trace=[],
                error_bounds={},
                timestamp=time.time(),
                udl_file=pred["file_path"],
            )
            reports.append(report)

        return PerformanceResult(
            strategy_used=ProcessingStrategy.GPU,
            total_files=len(file_paths),
            successful_files=len(reports),
            failed_files=len(errors),
            processing_time=0.0,  # Will be set by caller
            throughput=0.0,  # Will be set by caller
            memory_peak_mb=0.0,
            cache_hit_ratio=0.0,
            reports=reports,
            errors=errors,
        )

    def _process_streaming(
        self,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]],
    ) -> PerformanceResult:
        """Process files with streaming."""
        reports = []
        errors = []

        for file_path in file_paths:
            try:
                # Process file with streaming
                streaming_results = list(
                    process_large_file(
                        file_path=file_path,
                        metric_names=metric_names,
                        chunk_size=self.config.streaming_chunk_size,
                        overlap_size=self.config.streaming_overlap_size,
                    )
                )

                # Aggregate streaming results
                if streaming_results:
                    # Use the last result as the final report (simplified)
                    final_result = streaming_results[-1]
                    if final_result.partial_results:
                        reports.append(final_result.partial_results[0])

            except Exception as e:
                errors.append(f"Error streaming {file_path}: {e}")

        return PerformanceResult(
            strategy_used=ProcessingStrategy.STREAMING,
            total_files=len(file_paths),
            successful_files=len(reports),
            failed_files=len(errors),
            processing_time=0.0,  # Will be set by caller
            throughput=0.0,  # Will be set by caller
            memory_peak_mb=0.0,
            cache_hit_ratio=0.0,
            reports=reports,
            errors=errors,
        )

    def _process_memory_mapped(
        self,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]],
    ) -> PerformanceResult:
        """Process files with memory mapping."""
        processor = self.processors[ProcessingStrategy.MEMORY_MAPPED]
        reports = processor.process_multiple_files(file_paths)

        successful_reports = [r for r in reports if not r.errors]
        failed_reports = [r for r in reports if r.errors]

        return PerformanceResult(
            strategy_used=ProcessingStrategy.MEMORY_MAPPED,
            total_files=len(file_paths),
            successful_files=len(successful_reports),
            failed_files=len(failed_reports),
            processing_time=0.0,  # Will be set by caller
            throughput=0.0,  # Will be set by caller
            memory_peak_mb=0.0,
            cache_hit_ratio=0.0,
            reports=reports,
        )

    def _process_incremental(
        self,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]],
    ) -> PerformanceResult:
        """Process files with incremental computation."""
        processor = self.processors[ProcessingStrategy.INCREMENTAL]
        incremental_results = processor.process_multiple_files(file_paths)

        reports = [r.report for r in incremental_results]
        cache_hits = sum(1 for r in incremental_results if r.was_cached)
        cache_hit_ratio = (
            cache_hits / len(incremental_results) if incremental_results else 0.0
        )

        successful_results = [r for r in incremental_results if not r.report.errors]
        failed_results = [r for r in incremental_results if r.report.errors]

        return PerformanceResult(
            strategy_used=ProcessingStrategy.INCREMENTAL,
            total_files=len(file_paths),
            successful_files=len(successful_results),
            failed_files=len(failed_results),
            processing_time=0.0,  # Will be set by caller
            throughput=0.0,  # Will be set by caller
            memory_peak_mb=0.0,
            cache_hit_ratio=cache_hit_ratio,
            reports=reports,
            metadata={"incremental_results": incremental_results},
        )

    def _record_performance(
        self, strategy: ProcessingStrategy, result: PerformanceResult
    ) -> None:
        """Record performance metrics for strategy optimization."""
        performance_record = {
            "strategy": strategy,
            "timestamp": time.time(),
            "file_count": result.total_files,
            "processing_time": result.processing_time,
            "throughput": result.throughput,
            "success_rate": (
                result.successful_files / result.total_files
                if result.total_files > 0
                else 0.0
            ),
            "cache_hit_ratio": result.cache_hit_ratio,
        }

        self.performance_history.append(performance_record)

        # Update strategy performance tracking
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []

        self.strategy_performance[strategy].append(performance_record)

        # Keep limited history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics and recommendations."""
        if not self.performance_history:
            return {"message": "No performance data available"}

        # Overall statistics
        total_files = sum(r["file_count"] for r in self.performance_history)
        total_time = sum(r["processing_time"] for r in self.performance_history)
        avg_throughput = sum(r["throughput"] for r in self.performance_history) / len(
            self.performance_history
        )

        # Strategy performance
        strategy_stats = {}
        for strategy, records in self.strategy_performance.items():
            if records:
                strategy_stats[strategy.value] = {
                    "usage_count": len(records),
                    "avg_throughput": sum(r["throughput"] for r in records)
                    / len(records),
                    "avg_success_rate": sum(r["success_rate"] for r in records)
                    / len(records),
                    "avg_cache_hit_ratio": sum(r["cache_hit_ratio"] for r in records)
                    / len(records),
                }

        # Find best performing strategy
        best_strategy = None
        best_throughput = 0.0

        for strategy, stats in strategy_stats.items():
            if stats["avg_throughput"] > best_throughput:
                best_throughput = stats["avg_throughput"]
                best_strategy = strategy

        return {
            "total_files_processed": total_files,
            "total_processing_time": total_time,
            "average_throughput": avg_throughput,
            "strategy_statistics": strategy_stats,
            "best_performing_strategy": best_strategy,
            "recommendations": self._generate_recommendations(strategy_stats),
        }

    def _generate_recommendations(
        self, strategy_stats: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        if not strategy_stats:
            return ["Insufficient performance data for recommendations"]

        # Find strategies with high cache hit ratios
        high_cache_strategies = [
            strategy
            for strategy, stats in strategy_stats.items()
            if stats.get("avg_cache_hit_ratio", 0) > 0.5
        ]

        if high_cache_strategies:
            recommendations.append(
                f"Consider using incremental processing more often. "
                f"Strategies with high cache hit ratios: {', '.join(high_cache_strategies)}"
            )

        # Find strategies with low success rates
        low_success_strategies = [
            strategy
            for strategy, stats in strategy_stats.items()
            if stats.get("avg_success_rate", 1.0) < 0.9
        ]

        if low_success_strategies:
            recommendations.append(
                f"Investigate reliability issues with: {', '.join(low_success_strategies)}"
            )

        # GPU utilization recommendation
        if "gpu" in strategy_stats:
            gpu_stats = strategy_stats["gpu"]
            if gpu_stats["usage_count"] < 5:
                recommendations.append(
                    "Consider using GPU acceleration more frequently for medium-sized workloads"
                )

        # Distributed computing recommendation
        if "distributed" not in strategy_stats and DISTRIBUTED_AVAILABLE:
            recommendations.append(
                "Consider setting up distributed computing for large workloads"
            )

        return recommendations


# Convenience functions
def process_files_optimized(
    file_paths: List[Path],
    metric_names: List[str],
    weights: Optional[Dict[str, float]] = None,
    config: Optional[PerformanceConfig] = None,
) -> PerformanceResult:
    """
    Convenience function to process files with automatic performance optimization.

    Args:
        file_paths: List of file paths to process
        metric_names: List of metrics to compute
        weights: Optional metric weights
        config: Performance configuration

    Returns:
        PerformanceResult with processing information
    """
    optimizer = PerformanceOptimizer(config)
    return optimizer.process_files(file_paths, metric_names, weights)


def process_directory_optimized(
    directory: Path,
    metric_names: List[str],
    weights: Optional[Dict[str, float]] = None,
    file_patterns: List[str] = None,
    config: Optional[PerformanceConfig] = None,
) -> PerformanceResult:
    """
    Convenience function to process directory with automatic performance optimization.

    Args:
        directory: Directory containing UDL files
        metric_names: List of metrics to compute
        weights: Optional metric weights
        file_patterns: File patterns to match
        config: Performance configuration

    Returns:
        PerformanceResult with processing information
    """
    optimizer = PerformanceOptimizer(config)
    return optimizer.process_directory(directory, metric_names, weights, file_patterns)


def benchmark_all_strategies(
    test_files: List[Path], metric_names: List[str], iterations: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark all available processing strategies.

    Args:
        test_files: List of test files
        metric_names: List of metrics to compute
        iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results for each strategy
    """
    results = {}

    # Test each strategy
    strategies_to_test = [
        ProcessingStrategy.SEQUENTIAL,
        ProcessingStrategy.PARALLEL,
        ProcessingStrategy.INCREMENTAL,
        ProcessingStrategy.MEMORY_MAPPED,
        ProcessingStrategy.STREAMING,
    ]

    # Add optional strategies if available
    if DISTRIBUTED_AVAILABLE:
        strategies_to_test.append(ProcessingStrategy.DISTRIBUTED)

    if GPU_AVAILABLE:
        strategies_to_test.append(ProcessingStrategy.GPU)

    for strategy in strategies_to_test:
        strategy_results = []

        for i in range(iterations):
            try:
                config = PerformanceConfig(strategy=strategy, enable_progress=False)
                optimizer = PerformanceOptimizer(config)

                result = optimizer.process_files(test_files, metric_names)

                strategy_results.append(
                    {
                        "processing_time": result.processing_time,
                        "throughput": result.throughput,
                        "success_rate": (
                            result.successful_files / result.total_files
                            if result.total_files > 0
                            else 0.0
                        ),
                        "cache_hit_ratio": result.cache_hit_ratio,
                    }
                )

            except Exception as e:
                logger.error(f"Benchmark failed for {strategy.value}: {e}")
                strategy_results.append(
                    {
                        "processing_time": float("inf"),
                        "throughput": 0.0,
                        "success_rate": 0.0,
                        "cache_hit_ratio": 0.0,
                        "error": str(e),
                    }
                )

        # Calculate averages
        if strategy_results:
            valid_results = [r for r in strategy_results if "error" not in r]

            if valid_results:
                results[strategy.value] = {
                    "avg_processing_time": sum(
                        r["processing_time"] for r in valid_results
                    )
                    / len(valid_results),
                    "avg_throughput": sum(r["throughput"] for r in valid_results)
                    / len(valid_results),
                    "avg_success_rate": sum(r["success_rate"] for r in valid_results)
                    / len(valid_results),
                    "avg_cache_hit_ratio": sum(
                        r["cache_hit_ratio"] for r in valid_results
                    )
                    / len(valid_results),
                    "successful_runs": len(valid_results),
                    "total_runs": len(strategy_results),
                }
            else:
                results[strategy.value] = {
                    "error": "All benchmark runs failed",
                    "successful_runs": 0,
                    "total_runs": len(strategy_results),
                }

    return results
