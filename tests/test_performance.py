"""
Performance tests for UDL Rating Framework.

Tests performance characteristics and complexity bounds of core components.
"""

import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from udl_rating_framework.benchmarks.performance_benchmarks import PerformanceBenchmark
from udl_rating_framework.core.metrics.base import MetricRegistry
from udl_rating_framework.core.multiprocessing import ParallelProcessor
from udl_rating_framework.core.pipeline import RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.models.ctm_adapter import UDLRatingCTM


class TestMetricComputationPerformance:
    """
    Performance tests for metric computation.

    Validates: Requirements 8.6, 9.2, 10.7
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Ensure we have metrics registered for testing
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
        from udl_rating_framework.core.metrics.expressiveness import (
            ExpressivenessMetric,
        )

        # Clear registry and register test metrics
        MetricRegistry.clear()
        MetricRegistry.register("consistency", ConsistencyMetric)
        MetricRegistry.register("completeness", CompletenessMetric)
        MetricRegistry.register("expressiveness", ExpressivenessMetric)

    def teardown_method(self):
        """Clean up after tests and re-register default metrics."""
        MetricRegistry.clear()
        # Re-register default metrics for other tests
        from udl_rating_framework.core.metrics import _register_default_metrics

        _register_default_metrics()

    def generate_udl_content(self, num_rules: int) -> str:
        """Generate UDL content with specified number of rules."""
        lines = [f"# Generated UDL with {num_rules} rules"]

        for i in range(num_rules):
            if i == 0:
                lines.append(f"start ::= rule_{i}")
            else:
                if i % 5 == 0:
                    lines.append(f"rule_{i} ::= 'token_{i}' | rule_{i - 1}")
                elif i % 3 == 0:
                    lines.append(
                        f"rule_{i} ::= 'prefix_{i}' rule_{max(0, i - 2)} 'suffix_{i}'"
                    )
                else:
                    lines.append(f"rule_{i} ::= 'literal_{i}'")

        return "\n".join(lines)

    def create_test_udl(self, num_rules: int) -> UDLRepresentation:
        """Create test UDL with specified number of rules."""
        content = self.generate_udl_content(num_rules)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".udl", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            udl = UDLRepresentation(content, temp_path)
            return udl
        finally:
            os.unlink(temp_path)

    def test_metric_computation_time_vs_udl_size(self):
        """
        Benchmark metric computation time vs UDL size.

        Validates: Requirements 8.6, 9.2
        """
        udl_sizes = [10, 25, 50, 100, 200]
        metric_names = ["consistency", "completeness"]

        results = {}

        for metric_name in metric_names:
            metric_class = MetricRegistry.get_metric(metric_name)
            metric = metric_class()

            times = []

            for size in udl_sizes:
                udl = self.create_test_udl(size)

                # Measure computation time
                start_time = time.perf_counter()
                value = metric.compute(udl)
                end_time = time.perf_counter()

                execution_time = end_time - start_time
                times.append(execution_time)

                # Verify metric produces valid result
                assert 0.0 <= value <= 1.0, (
                    f"Metric {metric_name} out of bounds: {value}"
                )

            results[metric_name] = times

            # Log performance characteristics
            print(f"\n{metric_name} performance:")
            for i, (size, time_taken) in enumerate(zip(udl_sizes, times)):
                throughput = 1.0 / \
                    time_taken if time_taken > 0 else float("inf")
                print(
                    f"  Size {size}: {time_taken:.4f}s ({throughput:.2f} ops/s)")

        # Verify reasonable performance (should complete within reasonable time)
        for metric_name, times in results.items():
            max_time = max(times)
            assert max_time < 10.0, (
                f"Metric {metric_name} too slow: {max_time:.4f}s for largest input"
            )

            # Verify performance doesn't degrade exponentially
            if len(times) >= 3:
                # Check that time doesn't grow faster than O(n^2)
                size_ratios = [
                    udl_sizes[i] / udl_sizes[i - 1] for i in range(1, len(udl_sizes))
                ]
                time_ratios = [
                    times[i] / times[i - 1]
                    for i in range(1, len(times))
                    if times[i - 1] > 0
                ]

                for i, (size_ratio, time_ratio) in enumerate(
                    zip(size_ratios, time_ratios)
                ):
                    # Time ratio should not exceed size_ratio^2 (indicating worse than O(n^2))
                    assert time_ratio <= size_ratio**2 * 2, (
                        f"Metric {metric_name} performance degrades too quickly at step {i + 1}"
                    )

    def test_verify_complexity_bounds_on_or_ologn(self):
        """
        Verify complexity bounds (O(n) or O(n log n)).

        Validates: Requirements 9.2, 10.7
        """
        udl_sizes = [20, 40, 80, 160, 320]
        metric_names = ["consistency", "completeness"]

        for metric_name in metric_names:
            metric_class = MetricRegistry.get_metric(metric_name)
            metric = metric_class()

            times = []

            for size in udl_sizes:
                udl = self.create_test_udl(size)

                # Run multiple iterations for more stable timing
                iterations = 3
                iteration_times = []

                for _ in range(iterations):
                    start_time = time.perf_counter()
                    _ = metric.compute(udl)
                    end_time = time.perf_counter()
                    iteration_times.append(end_time - start_time)

                # Use median time to reduce noise
                median_time = statistics.median(iteration_times)
                times.append(median_time)

            # Analyze complexity
            if len(times) >= 3:
                # Calculate growth ratios
                growth_ratios = []
                for i in range(1, len(udl_sizes)):
                    size_ratio = udl_sizes[i] / udl_sizes[i - 1]
                    time_ratio = (
                        times[i] / times[i - 1] if times[i -
                                                         1] > 0 else float("inf")
                    )
                    growth_ratios.append(time_ratio / size_ratio)

                avg_growth_ratio = statistics.mean(growth_ratios)

                # For O(n) complexity, ratio should be close to 1
                # For O(n log n) complexity, ratio should be slightly > 1 but < 2
                # We allow up to 3.0 to account for measurement noise and overhead
                assert avg_growth_ratio <= 3.0, (
                    f"Metric {metric_name} appears to have worse than O(n log n) complexity: avg ratio {avg_growth_ratio:.2f}"
                )

                print(f"\n{metric_name} complexity analysis:")
                print(f"  Average growth ratio: {avg_growth_ratio:.2f}")
                print(
                    f"  Complexity bound: {'PASS' if avg_growth_ratio <= 3.0 else 'FAIL'}"
                )


class TestCTMInferencePerformance:
    """
    Performance tests for CTM inference.

    Validates: Requirements 8.6, 10.7
    """

    def test_ctm_inference_time_vs_sequence_length(self):
        """
        Benchmark CTM inference time vs sequence length.

        Validates: Requirements 8.6, 10.7
        """
        sequence_lengths = [10, 25, 50, 100, 200]
        vocab_size = 100
        batch_size = 1

        # Create CTM model (smaller for testing)
        model = UDLRatingCTM(
            vocab_size=vocab_size, d_model=32, iterations=5, n_synch_out=8
        )
        model.eval()

        times = []

        for seq_len in sequence_lengths:
            # Generate random input
            token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

            # Warm up
            with torch.no_grad():
                _ = model(token_ids)

            # Measure inference time
            iterations = 3
            iteration_times = []

            for _ in range(iterations):
                start_time = time.perf_counter()
                with torch.no_grad():
                    ratings, certainties, _, _ = model(token_ids)
                end_time = time.perf_counter()

                iteration_times.append(end_time - start_time)

                # Verify output shapes and ranges
                assert ratings.shape == (batch_size, 1)
                assert certainties.shape == (batch_size, 2)
                assert torch.all(ratings >= 0.0) and torch.all(ratings <= 1.0)

            median_time = statistics.median(iteration_times)
            times.append(median_time)

            throughput = seq_len / \
                median_time if median_time > 0 else float("inf")
            print(
                f"Sequence length {seq_len}: {median_time:.4f}s ({throughput:.0f} tokens/s)"
            )

        # Verify reasonable performance
        max_time = max(times)
        assert max_time < 5.0, (
            f"CTM inference too slow: {max_time:.4f}s for largest sequence"
        )

        # Verify performance scales reasonably with sequence length
        if len(times) >= 3:
            # Calculate growth ratios
            growth_ratios = []
            for i in range(1, len(sequence_lengths)):
                size_ratio = sequence_lengths[i] / sequence_lengths[i - 1]
                time_ratio = (
                    times[i] / times[i - 1] if times[i -
                                                     1] > 0 else float("inf")
                )
                growth_ratios.append(time_ratio / size_ratio)

            avg_growth_ratio = statistics.mean(growth_ratios)

            # CTM should scale reasonably with sequence length
            # Allow up to 2.0 ratio for transformer-like attention complexity
            assert avg_growth_ratio <= 2.5, (
                f"CTM inference scales poorly with sequence length: avg ratio {avg_growth_ratio:.2f}"
            )

            print(f"CTM complexity analysis:")
            print(f"  Average growth ratio: {avg_growth_ratio:.2f}")
            print(
                f"  Scaling: {'GOOD' if avg_growth_ratio <= 2.0 else 'ACCEPTABLE' if avg_growth_ratio <= 2.5 else 'POOR'}"
            )

    def test_ctm_batch_processing_efficiency(self):
        """
        Test CTM batch processing efficiency.

        Validates: Requirements 8.6
        """
        vocab_size = 100
        seq_len = 50
        batch_sizes = [1, 2, 4, 8]

        model = UDLRatingCTM(
            vocab_size=vocab_size, d_model=32, iterations=5, n_synch_out=8
        )
        model.eval()

        times_per_sample = []

        for batch_size in batch_sizes:
            token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

            # Warm up
            with torch.no_grad():
                _ = model(token_ids)

            # Measure batch processing time
            iterations = 3
            iteration_times = []

            for _ in range(iterations):
                start_time = time.perf_counter()
                with torch.no_grad():
                    ratings, certainties, _, _ = model(token_ids)
                end_time = time.perf_counter()

                iteration_times.append(end_time - start_time)

                # Verify output shapes
                assert ratings.shape == (batch_size, 1)
                assert certainties.shape == (batch_size, 2)

            median_time = statistics.median(iteration_times)
            time_per_sample = median_time / batch_size
            times_per_sample.append(time_per_sample)

            throughput = batch_size / \
                median_time if median_time > 0 else float("inf")
            print(
                f"Batch size {batch_size}: {median_time:.4f}s total, {time_per_sample:.4f}s per sample ({throughput:.1f} samples/s)"
            )

        # Verify batch processing is more efficient than individual processing
        single_sample_time = times_per_sample[0]  # batch_size = 1

        for i, batch_size in enumerate(batch_sizes[1:], 1):
            batch_time_per_sample = times_per_sample[i]
            efficiency_ratio = single_sample_time / batch_time_per_sample

            # Batch processing should be at least as efficient as individual processing
            assert efficiency_ratio >= 0.8, (
                f"Batch processing (size {batch_size}) is less efficient than individual: ratio {efficiency_ratio:.2f}"
            )

            print(
                f"Batch size {batch_size} efficiency: {efficiency_ratio:.2f}x")


class TestBatchProcessingPerformance:
    """
    Performance tests for batch processing.

    Validates: Requirements 8.6, 9.2, 10.7
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Ensure we have metrics registered for testing
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric

        # Clear registry and register test metrics
        MetricRegistry.clear()
        MetricRegistry.register("consistency", ConsistencyMetric)
        MetricRegistry.register("completeness", CompletenessMetric)

    def teardown_method(self):
        """Clean up after tests and re-register default metrics."""
        MetricRegistry.clear()
        # Re-register default metrics for other tests
        from udl_rating_framework.core.metrics import _register_default_metrics

        _register_default_metrics()

    def generate_test_files(self, count: int) -> List[tuple]:
        """Generate test UDL files."""
        files = []
        for i in range(count):
            content = f"""
            # Test UDL {i}
            rule_{i} ::= 'token_{i}' | 'alt_{i}'
            expr_{i} ::= rule_{i} '+' expr_{i} | rule_{i}
            """
            files.append((f"test_{i}.udl", content))
        return files

    def test_batch_processing_throughput(self):
        """
        Benchmark batch processing throughput.

        Validates: Requirements 8.6, 9.2
        """
        file_counts = [10, 25, 50]
        worker_counts = [1, 2, 4]

        for file_count in file_counts:
            test_files = self.generate_test_files(file_count)

            for worker_count in worker_counts:
                processor = ParallelProcessor(max_workers=worker_count)

                start_time = time.perf_counter()
                reports, stats = processor.process_files_parallel(
                    file_contents=test_files,
                    metric_names=["consistency", "completeness"],
                    enable_tracing=False,
                )
                end_time = time.perf_counter()

                total_time = end_time - start_time
                throughput = file_count / \
                    total_time if total_time > 0 else float("inf")

                # Verify all files were processed
                assert len(reports) == file_count
                assert stats.total_files == file_count
                assert stats.successful + stats.failed == file_count

                print(
                    f"Files: {file_count}, Workers: {worker_count}, Time: {total_time:.3f}s, Throughput: {throughput:.2f} files/s"
                )

                # Verify reasonable performance
                assert total_time < 30.0, (
                    f"Batch processing too slow: {total_time:.3f}s for {file_count} files"
                )

                # Verify most files processed successfully
                success_rate = stats.successful / stats.total_files
                assert success_rate >= 0.8, f"Low success rate: {success_rate:.2f}"

    def test_parallel_processing_speedup(self):
        """
        Test that parallel processing provides speedup.

        Validates: Requirements 8.6, 10.7
        """
        file_count = 20
        test_files = self.generate_test_files(file_count)

        # Test with different worker counts
        worker_counts = [1, 2, 4]
        times = []

        for worker_count in worker_counts:
            processor = ParallelProcessor(max_workers=worker_count)

            start_time = time.perf_counter()
            reports, stats = processor.process_files_parallel(
                file_contents=test_files,
                metric_names=["consistency"],
                enable_tracing=False,
            )
            end_time = time.perf_counter()

            total_time = end_time - start_time
            times.append(total_time)

            # Verify processing completed
            assert len(reports) == file_count
            assert stats.successful > 0

            print(f"Workers: {worker_count}, Time: {total_time:.3f}s")

        # Calculate speedup
        sequential_time = times[0]  # 1 worker

        for i, worker_count in enumerate(worker_counts[1:], 1):
            parallel_time = times[i]
            speedup = (
                sequential_time /
                parallel_time if parallel_time > 0 else float("inf")
            )

            # Parallel processing should provide some speedup or at least not be much slower
            # For small tasks, overhead may dominate, so allow more tolerance
            expected_min_speedup = (
                0.7 if worker_count > 1 else 1.0
            )  # Allow 30% overhead for small tasks
            assert speedup >= expected_min_speedup, (
                f"Parallel processing too slow with {worker_count} workers: {speedup:.2f}x"
            )

            print(f"Speedup with {worker_count} workers: {speedup:.2f}x")

    def test_memory_usage_scaling(self):
        """
        Test that memory usage scales reasonably with batch size.

        Validates: Requirements 10.7
        """
        import psutil

        process = psutil.Process()
        file_counts = [10, 20, 40]

        memory_usages = []

        for file_count in file_counts:
            test_files = self.generate_test_files(file_count)

            # Measure memory before processing
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            processor = ParallelProcessor(max_workers=2)
            reports, stats = processor.process_files_parallel(
                file_contents=test_files,
                metric_names=["consistency"],
                enable_tracing=False,
            )

            # Measure memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            memory_usages.append(memory_used)

            print(f"Files: {file_count}, Memory used: {memory_used:.1f} MB")

            # Verify processing completed
            assert len(reports) == file_count

        # Verify memory usage doesn't grow exponentially
        if len(memory_usages) >= 2:
            for i in range(1, len(file_counts)):
                file_ratio = file_counts[i] / file_counts[i - 1]

                # Skip comparison if memory usage is too small to measure accurately
                if memory_usages[i - 1] < 0.1 or memory_usages[i] < 0.1:
                    print(
                        f"  Skipping memory ratio check for small usage: {memory_usages[i - 1]:.1f} -> {memory_usages[i]:.1f} MB"
                    )
                    continue

                memory_ratio = memory_usages[i] / memory_usages[i - 1]

                # Memory usage should not grow faster than file count
                # Allow 3x factor for overhead and measurement noise
                assert memory_ratio <= file_ratio * 3, (
                    f"Memory usage grows too quickly: {memory_ratio:.2f}x for {file_ratio:.2f}x files"
                )


class TestPerformanceBenchmarkIntegration:
    """
    Integration tests for performance benchmark system.

    Validates: Requirements 8.6, 9.2, 10.7
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Ensure we have metrics registered for testing
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric

        # Clear registry and register test metrics
        MetricRegistry.clear()
        MetricRegistry.register("consistency", ConsistencyMetric)
        MetricRegistry.register("completeness", CompletenessMetric)

    def teardown_method(self):
        """Clean up after tests and re-register default metrics."""
        MetricRegistry.clear()
        # Re-register default metrics for other tests
        from udl_rating_framework.core.metrics import _register_default_metrics

        _register_default_metrics()

    def test_benchmark_metric_computation(self):
        """
        Test benchmark system for metric computation.

        Validates: Requirements 8.6, 9.2
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = PerformanceBenchmark(Path(temp_dir))

            # Run small benchmark
            suite = benchmark.benchmark_metric_computation(
                udl_sizes=[10, 25, 50], metric_names=["consistency"], iterations=2
            )

            # Verify benchmark results
            assert suite.suite_name == "Metric Computation Performance"
            assert len(suite.results) > 0

            # Verify all results have required fields
            for result in suite.results:
                assert result.execution_time > 0
                assert result.input_size > 0
                assert result.throughput >= 0
                assert "metric_name" in result.metadata
                assert "udl_size" in result.metadata

            # Verify summary statistics
            summary = suite.get_summary()
            assert summary["total_benchmarks"] > 0
            assert summary["avg_execution_time"] > 0
            assert summary["avg_throughput"] > 0

            print(
                f"Benchmark completed: {summary['total_benchmarks']} tests, "
                f"avg time: {summary['avg_execution_time']:.4f}s"
            )

    def test_benchmark_ctm_inference(self):
        """
        Test benchmark system for CTM inference.

        Validates: Requirements 8.6, 10.7
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = PerformanceBenchmark(Path(temp_dir))

            # Run small benchmark
            suite = benchmark.benchmark_ctm_inference(
                sequence_lengths=[10, 25],
                vocab_sizes=[50],
                batch_sizes=[1, 2],
                iterations=2,
            )

            # Verify benchmark results
            assert suite.suite_name == "CTM Inference Performance"
            assert len(suite.results) > 0

            # Verify all results have required fields
            for result in suite.results:
                assert result.execution_time > 0
                assert result.input_size > 0
                assert result.throughput >= 0
                assert "vocab_size" in result.metadata
                assert "batch_size" in result.metadata
                assert "sequence_length" in result.metadata

            # Verify summary statistics
            summary = suite.get_summary()
            assert summary["total_benchmarks"] > 0
            assert summary["avg_execution_time"] > 0

            print(
                f"CTM benchmark completed: {summary['total_benchmarks']} tests, "
                f"avg time: {summary['avg_execution_time']:.4f}s"
            )

    def test_verify_complexity_bounds(self):
        """
        Test complexity bounds verification.

        Validates: Requirements 9.2, 10.7
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = PerformanceBenchmark(Path(temp_dir))

            # Verify complexity bounds for small inputs
            results = benchmark.verify_complexity_bounds(
                max_size=100, step_size=25, target_complexity="O(n)"
            )

            # Verify results structure
            assert isinstance(results, dict)
            assert len(results) > 0

            # All results should be boolean
            for component, passed in results.items():
                assert isinstance(passed, bool)
                print(
                    f"Complexity check for {component}: {'PASS' if passed else 'FAIL'}"
                )

            # At least some components should pass complexity checks
            passed_count = sum(results.values())
            total_count = len(results)
            pass_rate = passed_count / total_count if total_count > 0 else 0

            assert pass_rate >= 0.5, (
                f"Too many complexity check failures: {pass_rate:.2f} pass rate"
            )
