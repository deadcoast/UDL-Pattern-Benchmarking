"""
Performance benchmarks for UDL Rating Framework.

Provides comprehensive benchmarking of metric computation, CTM inference,
and batch processing performance.
"""

import gc
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

try:
    from memory_profiler import profile as memory_profile
except ImportError:
    # Memory profiler is optional
    def memory_profile(func):
        return func


from udl_rating_framework.core.metrics.base import MetricRegistry
from udl_rating_framework.core.multiprocessing import ParallelProcessor
from udl_rating_framework.core.pipeline import RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.models.ctm_adapter import UDLRatingCTM, UDLTokenVocabulary

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    input_size: int
    throughput: float  # items per second
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    total_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the suite."""
        self.results.append(result)
        self.total_time += result.execution_time

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the benchmark suite."""
        if not self.results:
            return {}

        execution_times = [r.execution_time for r in self.results]
        memory_usages = [r.memory_usage_mb for r in self.results]
        throughputs = [r.throughput for r in self.results]

        return {
            "total_benchmarks": len(self.results),
            "total_time": self.total_time,
            "avg_execution_time": statistics.mean(execution_times),
            "median_execution_time": statistics.median(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "avg_memory_usage_mb": statistics.mean(memory_usages),
            "max_memory_usage_mb": max(memory_usages),
            "avg_throughput": statistics.mean(throughputs),
            "max_throughput": max(throughputs),
        }


class PerformanceBenchmark:
    """
    Performance benchmarking engine for UDL Rating Framework.

    Provides comprehensive benchmarking of all major components with
    detailed performance metrics and complexity analysis.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize performance benchmark.

        Args:
            output_dir: Directory to save benchmark results and plots
        """
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize system monitoring
        self.process = psutil.Process()

    def benchmark_metric_computation(
        self,
        udl_sizes: List[int] = None,
        metric_names: List[str] = None,
        iterations: int = 5,
    ) -> BenchmarkSuite:
        """
        Benchmark metric computation time vs UDL size.

        Args:
            udl_sizes: List of UDL sizes to test (in number of rules)
            metric_names: List of metrics to benchmark
            iterations: Number of iterations per test

        Returns:
            BenchmarkSuite with results
        """
        if udl_sizes is None:
            udl_sizes = [10, 50, 100, 200, 500, 1000]

        if metric_names is None:
            # Get all available metrics
            metric_names = list(MetricRegistry.list_metrics().keys())

        suite = BenchmarkSuite("Metric Computation Performance")

        logger.info(
            f"Benchmarking metric computation for {len(metric_names)} metrics across {len(udl_sizes)} sizes"
        )

        for metric_name in metric_names:
            for udl_size in udl_sizes:
                # Generate UDL of specified size
                udl_content = self._generate_udl_content(udl_size)
                udl = UDLRepresentation(udl_content, f"test_{udl_size}.udl")

                # Get metric instance
                try:
                    metric_class = MetricRegistry.get_metric(metric_name)
                    metric = metric_class()
                except KeyError:
                    logger.warning(f"Metric {metric_name} not found, skipping")
                    continue

                # Benchmark metric computation
                times = []
                memory_usages = []

                for _ in range(iterations):
                    # Force garbage collection
                    gc.collect()

                    # Measure memory before
                    memory_before = self.process.memory_info().rss / 1024 / 1024

                    # Measure computation time
                    start_time = time.perf_counter()
                    try:
                        value = metric.compute(udl)
                        end_time = time.perf_counter()

                        execution_time = end_time - start_time
                        times.append(execution_time)

                        # Measure memory after
                        memory_after = self.process.memory_info().rss / 1024 / 1024
                        memory_usages.append(memory_after - memory_before)

                    except Exception as e:
                        logger.error(
                            f"Error computing {metric_name} for size {udl_size}: {e}"
                        )
                        continue

                if times:
                    # Calculate statistics
                    avg_time = statistics.mean(times)
                    avg_memory = statistics.mean(memory_usages)
                    throughput = 1.0 / avg_time if avg_time > 0 else 0.0

                    result = BenchmarkResult(
                        name=f"{metric_name}_size_{udl_size}",
                        execution_time=avg_time,
                        memory_usage_mb=avg_memory,
                        cpu_usage_percent=0.0,  # Not measured for individual metrics
                        input_size=udl_size,
                        throughput=throughput,
                        metadata={
                            "metric_name": metric_name,
                            "udl_size": udl_size,
                            "iterations": iterations,
                            "std_time": (
                                statistics.stdev(times) if len(
                                    times) > 1 else 0.0
                            ),
                            "min_time": min(times),
                            "max_time": max(times),
                        },
                    )

                    suite.add_result(result)

                    logger.debug(
                        f"{metric_name} (size {udl_size}): {avg_time:.4f}s avg, {throughput:.2f} ops/s"
                    )

        # Save results
        self._save_benchmark_results(suite, "metric_computation")

        return suite

    def benchmark_ctm_inference(
        self,
        sequence_lengths: List[int] = None,
        vocab_sizes: List[int] = None,
        batch_sizes: List[int] = None,
        iterations: int = 3,
    ) -> BenchmarkSuite:
        """
        Benchmark CTM inference time vs sequence length.

        Args:
            sequence_lengths: List of sequence lengths to test
            vocab_sizes: List of vocabulary sizes to test
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations per test

        Returns:
            BenchmarkSuite with results
        """
        if sequence_lengths is None:
            sequence_lengths = [10, 25, 50, 100, 200, 500]

        if vocab_sizes is None:
            vocab_sizes = [100, 500, 1000]

        if batch_sizes is None:
            batch_sizes = [1, 4, 8]

        suite = BenchmarkSuite("CTM Inference Performance")

        logger.info(
            f"Benchmarking CTM inference across {len(sequence_lengths)} seq lengths, "
            f"{len(vocab_sizes)} vocab sizes, {len(batch_sizes)} batch sizes"
        )

        for vocab_size in vocab_sizes:
            for batch_size in batch_sizes:
                for seq_len in sequence_lengths:
                    # Create CTM model
                    model = UDLRatingCTM(
                        vocab_size=vocab_size,
                        d_model=64,  # Smaller for benchmarking
                        iterations=10,
                        n_synch_out=16,
                    )
                    model.eval()

                    # Generate random input
                    token_ids = torch.randint(
                        0, vocab_size, (batch_size, seq_len))

                    # Warm up
                    with torch.no_grad():
                        _ = model(token_ids)

                    # Benchmark inference
                    times = []
                    memory_usages = []

                    for _ in range(iterations):
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                        memory_before = self.process.memory_info().rss / 1024 / 1024

                        start_time = time.perf_counter()
                        with torch.no_grad():
                            ratings, certainties, _, _ = model(token_ids)
                        end_time = time.perf_counter()

                        execution_time = end_time - start_time
                        times.append(execution_time)

                        memory_after = self.process.memory_info().rss / 1024 / 1024
                        memory_usages.append(memory_after - memory_before)

                    # Calculate statistics
                    avg_time = statistics.mean(times)
                    avg_memory = statistics.mean(memory_usages)
                    throughput = (
                        (batch_size * seq_len) /
                        avg_time if avg_time > 0 else 0.0
                    )

                    result = BenchmarkResult(
                        name=f"ctm_vocab_{vocab_size}_batch_{batch_size}_seq_{seq_len}",
                        execution_time=avg_time,
                        memory_usage_mb=avg_memory,
                        cpu_usage_percent=0.0,
                        input_size=seq_len,
                        throughput=throughput,
                        metadata={
                            "vocab_size": vocab_size,
                            "batch_size": batch_size,
                            "sequence_length": seq_len,
                            "iterations": iterations,
                            "tokens_per_second": throughput,
                            "std_time": (
                                statistics.stdev(times) if len(
                                    times) > 1 else 0.0
                            ),
                        },
                    )

                    suite.add_result(result)

                    logger.debug(
                        f"CTM (vocab={vocab_size}, batch={batch_size}, seq={seq_len}): "
                        f"{avg_time:.4f}s, {throughput:.0f} tokens/s"
                    )

        # Save results
        self._save_benchmark_results(suite, "ctm_inference")

        return suite

    def benchmark_batch_processing(
        self,
        file_counts: List[int] = None,
        worker_counts: List[int] = None,
        iterations: int = 3,
    ) -> BenchmarkSuite:
        """
        Benchmark batch processing throughput.

        Args:
            file_counts: List of file counts to test
            worker_counts: List of worker counts to test
            iterations: Number of iterations per test

        Returns:
            BenchmarkSuite with results
        """
        if file_counts is None:
            file_counts = [10, 50, 100, 200]

        if worker_counts is None:
            worker_counts = [1, 2, 4, 8]

        suite = BenchmarkSuite("Batch Processing Performance")

        logger.info(
            f"Benchmarking batch processing across {len(file_counts)} file counts, "
            f"{len(worker_counts)} worker counts"
        )

        # Prepare test data
        test_udls = []
        for i in range(max(file_counts)):
            content = self._generate_udl_content(50 + i % 100)  # Varying sizes
            test_udls.append((f"test_{i}.udl", content))

        for file_count in file_counts:
            for worker_count in worker_counts:
                # Use subset of test UDLs
                udl_subset = test_udls[:file_count]

                times = []
                memory_usages = []

                for _ in range(iterations):
                    gc.collect()

                    memory_before = self.process.memory_info().rss / 1024 / 1024

                    # Create processor
                    processor = ParallelProcessor(max_workers=worker_count)

                    start_time = time.perf_counter()
                    try:
                        reports, stats = processor.process_files_parallel(
                            file_contents=udl_subset,
                            metric_names=["consistency", "completeness"],
                            enable_tracing=False,
                        )
                        end_time = time.perf_counter()

                        execution_time = end_time - start_time
                        times.append(execution_time)

                        memory_after = self.process.memory_info().rss / 1024 / 1024
                        memory_usages.append(memory_after - memory_before)

                    except Exception as e:
                        logger.error(
                            f"Error in batch processing (files={file_count}, workers={worker_count}): {e}"
                        )
                        continue

                if times:
                    # Calculate statistics
                    avg_time = statistics.mean(times)
                    avg_memory = statistics.mean(memory_usages)
                    throughput = file_count / avg_time if avg_time > 0 else 0.0

                    result = BenchmarkResult(
                        name=f"batch_files_{file_count}_workers_{worker_count}",
                        execution_time=avg_time,
                        memory_usage_mb=avg_memory,
                        cpu_usage_percent=0.0,
                        input_size=file_count,
                        throughput=throughput,
                        metadata={
                            "file_count": file_count,
                            "worker_count": worker_count,
                            "iterations": iterations,
                            "files_per_second": throughput,
                            "std_time": (
                                statistics.stdev(times) if len(
                                    times) > 1 else 0.0
                            ),
                            "speedup": (
                                throughput / (file_count / max(times))
                                if worker_count == 1
                                else None
                            ),
                        },
                    )

                    suite.add_result(result)

                    logger.debug(
                        f"Batch processing (files={file_count}, workers={worker_count}): "
                        f"{avg_time:.4f}s, {throughput:.2f} files/s"
                    )

        # Save results
        self._save_benchmark_results(suite, "batch_processing")

        return suite

    def verify_complexity_bounds(
        self,
        max_size: int = 1000,
        step_size: int = 100,
        target_complexity: str = "O(n)",
    ) -> Dict[str, bool]:
        """
        Verify that algorithms meet their complexity bounds.

        Args:
            max_size: Maximum input size to test
            step_size: Step size for input sizes
            target_complexity: Expected complexity ("O(n)", "O(n log n)", "O(n^2)")

        Returns:
            Dictionary mapping component names to whether they meet complexity bounds
        """
        logger.info(f"Verifying complexity bounds up to size {max_size}")

        sizes = list(range(step_size, max_size + 1, step_size))
        results = {}

        # Test metric computation complexity
        metric_names = list(MetricRegistry.list_metrics().keys())[
            :2
        ]  # Test first 2 metrics

        for metric_name in metric_names:
            try:
                metric_class = MetricRegistry.get_metric(metric_name)
                metric = metric_class()

                times = []
                for size in sizes:
                    udl_content = self._generate_udl_content(size)
                    udl = UDLRepresentation(udl_content, f"test_{size}.udl")

                    start_time = time.perf_counter()
                    _ = metric.compute(udl)
                    end_time = time.perf_counter()

                    times.append(end_time - start_time)

                # Check if growth matches expected complexity
                complexity_met = self._check_complexity_growth(
                    sizes, times, target_complexity
                )
                results[f"metric_{metric_name}"] = complexity_met

                logger.debug(
                    f"Metric {metric_name} complexity check: {'PASS' if complexity_met else 'FAIL'}"
                )

            except Exception as e:
                logger.error(
                    f"Error testing complexity for {metric_name}: {e}")
                results[f"metric_{metric_name}"] = False

        return results

    def _generate_udl_content(self, num_rules: int) -> str:
        """
        Generate UDL content with specified number of rules.

        Args:
            num_rules: Number of grammar rules to generate

        Returns:
            Generated UDL content
        """
        lines = [f"# Generated UDL with {num_rules} rules"]

        for i in range(num_rules):
            if i == 0:
                # Start rule
                lines.append(f"start ::= rule_{i}")
            else:
                # Regular rules with some variety
                if i % 5 == 0:
                    # Alternation rule
                    lines.append(f"rule_{i} ::= 'token_{i}' | rule_{i - 1}")
                elif i % 3 == 0:
                    # Sequence rule
                    lines.append(
                        f"rule_{i} ::= 'prefix_{i}' rule_{max(0, i - 2)} 'suffix_{i}'"
                    )
                else:
                    # Simple rule
                    lines.append(f"rule_{i} ::= 'literal_{i}'")

        return "\n".join(lines)

    def _check_complexity_growth(
        self, sizes: List[int], times: List[float], expected_complexity: str
    ) -> bool:
        """
        Check if timing data matches expected complexity growth.

        Args:
            sizes: Input sizes
            times: Execution times
            expected_complexity: Expected complexity string

        Returns:
            True if complexity bounds are met
        """
        if len(sizes) < 3 or len(times) < 3:
            return False

        # Calculate growth ratios
        ratios = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i -
                                          1] if times[i - 1] > 0 else float("inf")
            ratios.append(time_ratio / size_ratio)

        # Check if ratios are reasonable for expected complexity
        avg_ratio = statistics.mean(ratios)

        if expected_complexity == "O(n)":
            # For O(n), time ratio should be close to size ratio (ratio ~1)
            return 0.5 <= avg_ratio <= 2.0
        elif expected_complexity == "O(n log n)":
            # For O(n log n), ratio should grow slowly
            return avg_ratio <= 3.0
        elif expected_complexity == "O(n^2)":
            # For O(n^2), ratio should grow with size
            return avg_ratio >= 1.0
        else:
            # Unknown complexity, assume it passes
            return True

    def _save_benchmark_results(self, suite: BenchmarkSuite, filename_prefix: str):
        """
        Save benchmark results to files.

        Args:
            suite: Benchmark suite to save
            filename_prefix: Prefix for output files
        """
        # Save raw results as JSON
        import json

        results_data = {
            "suite_name": suite.suite_name,
            "timestamp": suite.timestamp,
            "total_time": suite.total_time,
            "summary": suite.get_summary(),
            "results": [
                {
                    "name": r.name,
                    "execution_time": r.execution_time,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "input_size": r.input_size,
                    "throughput": r.throughput,
                    "metadata": r.metadata,
                }
                for r in suite.results
            ],
        }

        json_file = self.output_dir / f"{filename_prefix}_results.json"
        with open(json_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved benchmark results to {json_file}")

        # Generate plots
        self._generate_performance_plots(suite, filename_prefix)

    def _generate_performance_plots(self, suite: BenchmarkSuite, filename_prefix: str):
        """
        Generate performance plots from benchmark results.

        Args:
            suite: Benchmark suite
            filename_prefix: Prefix for plot files
        """
        if not suite.results:
            return

        # Group results by type for plotting
        grouped_results = {}
        for result in suite.results:
            key = (
                result.metadata.get("metric_name")
                or result.metadata.get("vocab_size")
                or "default"
            )
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)

        # Create performance plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"{suite.suite_name} - Performance Analysis")

        # Plot 1: Execution time vs input size
        ax1 = axes[0, 0]
        for key, results in grouped_results.items():
            sizes = [r.input_size for r in results]
            times = [r.execution_time for r in results]
            ax1.plot(sizes, times, marker="o", label=str(key))
        ax1.set_xlabel("Input Size")
        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title("Execution Time vs Input Size")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Throughput vs input size
        ax2 = axes[0, 1]
        for key, results in grouped_results.items():
            sizes = [r.input_size for r in results]
            throughputs = [r.throughput for r in results]
            ax2.plot(sizes, throughputs, marker="s", label=str(key))
        ax2.set_xlabel("Input Size")
        ax2.set_ylabel("Throughput (ops/s)")
        ax2.set_title("Throughput vs Input Size")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Memory usage vs input size
        ax3 = axes[1, 0]
        for key, results in grouped_results.items():
            sizes = [r.input_size for r in results]
            memory = [r.memory_usage_mb for r in results]
            ax3.plot(sizes, memory, marker="^", label=str(key))
        ax3.set_xlabel("Input Size")
        ax3.set_ylabel("Memory Usage (MB)")
        ax3.set_title("Memory Usage vs Input Size")
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Performance distribution
        ax4 = axes[1, 1]
        all_times = [r.execution_time for r in suite.results]
        ax4.hist(all_times, bins=20, alpha=0.7, edgecolor="black")
        ax4.set_xlabel("Execution Time (s)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Execution Time Distribution")
        ax4.grid(True)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / f"{filename_prefix}_performance.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved performance plots to {plot_file}")


def run_comprehensive_benchmarks(
    output_dir: Optional[Path] = None,
) -> Dict[str, BenchmarkSuite]:
    """
    Run comprehensive performance benchmarks.

    Args:
        output_dir: Directory to save results

    Returns:
        Dictionary mapping benchmark names to results
    """
    benchmark = PerformanceBenchmark(output_dir)
    results = {}

    logger.info("Starting comprehensive performance benchmarks")

    # Benchmark metric computation
    logger.info("Benchmarking metric computation...")
    results["metric_computation"] = benchmark.benchmark_metric_computation(
        udl_sizes=[10, 50, 100, 200, 500], iterations=3
    )

    # Benchmark CTM inference
    logger.info("Benchmarking CTM inference...")
    results["ctm_inference"] = benchmark.benchmark_ctm_inference(
        sequence_lengths=[10, 25, 50, 100, 200],
        vocab_sizes=[100, 500],
        batch_sizes=[1, 4],
        iterations=3,
    )

    # Benchmark batch processing
    logger.info("Benchmarking batch processing...")
    results["batch_processing"] = benchmark.benchmark_batch_processing(
        file_counts=[10, 50, 100], worker_counts=[1, 2, 4], iterations=2
    )

    # Verify complexity bounds
    logger.info("Verifying complexity bounds...")
    complexity_results = benchmark.verify_complexity_bounds(
        max_size=500, step_size=50)

    # Log summary
    logger.info("Benchmark Summary:")
    for name, suite in results.items():
        summary = suite.get_summary()
        logger.info(
            f"  {name}: {summary.get('total_benchmarks', 0)} tests, "
            f"avg time: {summary.get('avg_execution_time', 0):.4f}s, "
            f"avg throughput: {summary.get('avg_throughput', 0):.2f} ops/s"
        )

    logger.info(
        f"Complexity verification: {sum(complexity_results.values())}/{len(complexity_results)} passed"
    )

    return results
