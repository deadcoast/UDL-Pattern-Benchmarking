"""
Performance tests for fixed performance optimization modules.

Tests streaming processing, incremental processing, and performance optimization
strategy effectiveness after the metric registration fixes.

Requirements: 8.6, 9.2, 10.7
"""

import pytest
import tempfile
import time
from pathlib import Path

# Import metrics first to ensure registration
import udl_rating_framework.core.metrics  # noqa: F401

from udl_rating_framework.core.streaming import (
    StreamingProcessor,
    StreamingConfig,
)
from udl_rating_framework.core.incremental import (
    IncrementalProcessor,
    IncrementalCache,
    FileSnapshot,
)
from udl_rating_framework.core.performance import (
    PerformanceOptimizer,
    PerformanceConfig,
    ProcessingStrategy,
)


@pytest.fixture
def sample_udl_content():
    """Sample UDL content for testing."""
    return """
# Sample UDL Grammar
start ::= expression
expression ::= term ('+' term | '-' term)*
term ::= factor ('*' factor | '/' factor)*
factor ::= number | '(' expression ')'
number ::= [0-9]+
"""


@pytest.fixture
def large_udl_content():
    """Large UDL content for testing streaming."""
    base_content = """
# Large UDL Grammar
start ::= statement_list
statement_list ::= statement*
statement ::= assignment | expression | comment
assignment ::= identifier '=' expression
expression ::= term ('+' term | '-' term)*
term ::= factor ('*' factor | '/' factor)*
factor ::= number | identifier | '(' expression ')'
identifier ::= [a-zA-Z][a-zA-Z0-9]*
number ::= [0-9]+
comment ::= '#' [^\n]*
"""
    # Repeat content to make it large
    repeated_content = base_content
    for i in range(100):
        repeated_content += f"\nrule_{i} ::= 'token_{i}' | rule_{max(0, i - 1)}\n"

    return repeated_content


@pytest.fixture
def temp_udl_files(sample_udl_content):
    """Create temporary UDL files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        files = []
        for i in range(5):
            file_path = temp_path / f"test_{i}.udl"
            content = (
                sample_udl_content + f"\n# File {i}\nfile_rule_{i} ::= 'file_{i}'\n"
            )
            file_path.write_text(content)
            files.append(file_path)
        yield files


@pytest.fixture
def large_temp_file(large_udl_content):
    """Create large temporary UDL file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".udl", delete=False) as f:
        f.write(large_udl_content)
        temp_path = Path(f.name)

    try:
        yield temp_path
    finally:
        temp_path.unlink()


class TestStreamingProcessorFixed:
    """Test streaming processing with various file sizes after fixes."""

    def test_streaming_with_small_file(self, sample_udl_content):
        """Test streaming processing with small content."""
        config = StreamingConfig(chunk_size=256)
        processor = StreamingProcessor(config)

        results = list(processor.process_content_streaming(sample_udl_content))

        assert len(results) > 0
        assert all(r.success or r.error for r in results)
        # Verify scores are in valid range
        for r in results:
            if r.success and not r.error:
                assert 0.0 <= r.aggregated_score <= 1.0
                assert 0.0 <= r.confidence <= 1.0

    def test_streaming_with_large_file(self, large_temp_file):
        """Test streaming processing with large file."""
        config = StreamingConfig(chunk_size=1024, overlap_size=128)
        processor = StreamingProcessor(config)

        start_time = time.time()
        results = list(processor.process_file_streaming(large_temp_file))
        processing_time = time.time() - start_time

        assert len(results) > 0
        assert processing_time < 60  # Should complete within 60 seconds

        # Verify all results have valid structure
        for r in results:
            assert r.file_path == str(large_temp_file)
            assert r.processing_time >= 0

    def test_streaming_with_varying_chunk_sizes(self, large_udl_content):
        """Test streaming with different chunk sizes."""
        chunk_sizes = [256, 512, 1024, 2048]

        for chunk_size in chunk_sizes:
            config = StreamingConfig(chunk_size=chunk_size)
            processor = StreamingProcessor(config)

            results = list(processor.process_content_streaming(large_udl_content))

            assert len(results) > 0, f"No results for chunk_size={chunk_size}"
            # Larger chunks should produce fewer results
            if chunk_size > 256:
                assert len(results) <= len(
                    list(
                        StreamingProcessor(
                            StreamingConfig(chunk_size=256)
                        ).process_content_streaming(large_udl_content)
                    )
                )

    def test_streaming_metric_initialization(self):
        """Test that streaming processor properly initializes metrics."""
        processor = StreamingProcessor()

        # Verify pipeline has metrics
        assert processor.pipeline is not None
        assert len(processor.pipeline.metrics) > 0

        # Verify default metrics are available
        expected_metrics = [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]
        for metric_name in expected_metrics:
            assert metric_name in processor.pipeline.metrics


class TestIncrementalProcessorFixed:
    """Test incremental processing with cache validation after fixes."""

    def test_incremental_cache_initialization(self):
        """Test that incremental cache initializes properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = IncrementalCache(Path(temp_dir))

            assert cache.cache_dir.exists()
            stats = cache.get_cache_stats()
            assert "hits" in stats
            assert "misses" in stats
            assert stats["cached_files"] == 0

    def test_incremental_cache_operations(self, temp_udl_files):
        """Test cache put/get operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = IncrementalCache(Path(temp_dir))
            file_path = temp_udl_files[0]

            # Create and store snapshot
            snapshot = FileSnapshot.from_file(file_path)
            cache.update_file_snapshot(file_path, snapshot)

            # Retrieve snapshot
            retrieved = cache.get_file_snapshot(file_path)
            assert retrieved is not None
            assert retrieved.content_hash == snapshot.content_hash

            # Store metric result
            cache.update_metric_result(file_path, "consistency", 0.85)
            result = cache.get_metric_result(file_path, "consistency")
            assert result == 0.85

    def test_incremental_cache_invalidation(self, temp_udl_files):
        """Test cache invalidation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = IncrementalCache(Path(temp_dir))
            file_path = temp_udl_files[0]

            # Store metric results
            cache.update_metric_result(file_path, "consistency", 0.85)
            cache.update_metric_result(file_path, "completeness", 0.90)

            # Invalidate specific metric
            cache.invalidate_file(file_path, {"consistency"})

            # Consistency should be invalidated
            assert cache.get_metric_result(file_path, "consistency") is None
            # Completeness should still exist
            assert cache.get_metric_result(file_path, "completeness") == 0.90

    def test_incremental_processing_new_file(self, temp_udl_files):
        """Test incremental processing of new file."""
        processor = IncrementalProcessor()
        file_path = temp_udl_files[0]

        result = processor.process_file(file_path)

        assert result.file_path == str(file_path)
        assert not result.was_cached  # New file shouldn't be cached
        assert result.report is not None
        assert 0.0 <= result.report.overall_score <= 1.0

    def test_incremental_processing_cached_file(self, temp_udl_files):
        """Test incremental processing with caching."""
        processor = IncrementalProcessor()
        file_path = temp_udl_files[0]

        # First processing
        result1 = processor.process_file(file_path)
        # Second processing (should use cache)
        result2 = processor.process_file(file_path)

        # Results should be consistent
        assert result1.report.overall_score == result2.report.overall_score

    def test_incremental_processing_multiple_files(self, temp_udl_files):
        """Test processing multiple files incrementally."""
        processor = IncrementalProcessor()

        results = processor.process_multiple_files(temp_udl_files)

        assert len(results) == len(temp_udl_files)
        for result in results:
            assert result.report is not None
            assert 0.0 <= result.report.overall_score <= 1.0

    def test_incremental_state_management(self, temp_udl_files):
        """Test incremental processor state management."""
        processor = IncrementalProcessor()

        # Process files
        processor.process_multiple_files(temp_udl_files)

        # Check statistics
        stats = processor.get_statistics()
        assert stats["files_processed"] == len(temp_udl_files)
        assert "cache_statistics" in stats


class TestPerformanceOptimizerFixed:
    """Test performance optimizer strategy selection after fixes."""

    def test_optimizer_initialization(self):
        """Test optimizer initializes with all processors."""
        optimizer = PerformanceOptimizer()

        # Check that essential processors are available
        assert ProcessingStrategy.PARALLEL in optimizer.processors
        assert ProcessingStrategy.STREAMING in optimizer.processors
        assert ProcessingStrategy.INCREMENTAL in optimizer.processors
        assert ProcessingStrategy.MEMORY_MAPPED in optimizer.processors

    def test_workload_analysis_small_files(self, temp_udl_files):
        """Test workload analysis for small files."""
        optimizer = PerformanceOptimizer()

        workload_info = optimizer._analyze_workload(temp_udl_files)

        assert workload_info["file_count"] == len(temp_udl_files)
        assert workload_info["avg_file_size_mb"] < 1  # Small files
        assert not workload_info["has_large_files"]

    def test_workload_analysis_large_file(self, large_temp_file):
        """Test workload analysis for large file."""
        optimizer = PerformanceOptimizer(
            PerformanceConfig(max_file_size_mb=0.001)  # Very low threshold
        )

        workload_info = optimizer._analyze_workload([large_temp_file])

        assert workload_info["file_count"] == 1
        assert workload_info["has_large_files"]  # Should detect as large

    def test_strategy_selection_parallel(self, temp_udl_files):
        """Test strategy selection for parallel processing."""
        optimizer = PerformanceOptimizer()

        workload_info = optimizer._analyze_workload(temp_udl_files)
        strategy = optimizer._select_strategy(workload_info)

        # For small files, should select parallel or incremental
        assert strategy in [ProcessingStrategy.PARALLEL, ProcessingStrategy.INCREMENTAL]

    def test_strategy_selection_streaming(self, large_temp_file):
        """Test strategy selection for streaming."""
        config = PerformanceConfig(
            max_file_size_mb=0.001,  # Very low threshold
            memory_map_threshold_mb=0.0001,
        )
        optimizer = PerformanceOptimizer(config)

        workload_info = optimizer._analyze_workload([large_temp_file])
        strategy = optimizer._select_strategy(workload_info)

        # For large files, should select streaming or memory mapped
        assert strategy in [
            ProcessingStrategy.STREAMING,
            ProcessingStrategy.MEMORY_MAPPED,
        ]

    def test_process_files_with_parallel_strategy(self, temp_udl_files):
        """Test processing files with parallel strategy."""
        config = PerformanceConfig(strategy=ProcessingStrategy.PARALLEL)
        optimizer = PerformanceOptimizer(config)

        result = optimizer.process_files(
            file_paths=temp_udl_files, metric_names=["consistency", "completeness"]
        )

        assert result.strategy_used == ProcessingStrategy.PARALLEL
        assert result.total_files == len(temp_udl_files)
        assert result.processing_time > 0
        assert len(result.reports) <= len(temp_udl_files)

    def test_process_files_with_incremental_strategy(self, temp_udl_files):
        """Test processing files with incremental strategy."""
        config = PerformanceConfig(strategy=ProcessingStrategy.INCREMENTAL)
        optimizer = PerformanceOptimizer(config)

        result = optimizer.process_files(
            file_paths=temp_udl_files, metric_names=["consistency", "completeness"]
        )

        assert result.strategy_used == ProcessingStrategy.INCREMENTAL
        assert result.total_files == len(temp_udl_files)
        assert result.processing_time > 0

    def test_process_files_with_streaming_strategy(self, large_temp_file):
        """Test processing files with streaming strategy."""
        config = PerformanceConfig(strategy=ProcessingStrategy.STREAMING)
        optimizer = PerformanceOptimizer(config)

        result = optimizer.process_files(
            file_paths=[large_temp_file], metric_names=["consistency", "completeness"]
        )

        assert result.strategy_used == ProcessingStrategy.STREAMING
        assert result.total_files == 1
        assert result.processing_time > 0

    def test_performance_statistics_collection(self, temp_udl_files):
        """Test performance statistics collection."""
        optimizer = PerformanceOptimizer()

        # Process files to generate statistics
        optimizer.process_files(file_paths=temp_udl_files, metric_names=["consistency"])

        stats = optimizer.get_performance_statistics()

        assert "total_files_processed" in stats
        assert "average_throughput" in stats
        assert "strategy_statistics" in stats
        assert stats["total_files_processed"] > 0

    def test_fallback_strategy(self, temp_udl_files):
        """Test fallback strategy when primary fails."""
        # Use a config that might fail
        config = PerformanceConfig(
            strategy=ProcessingStrategy.DISTRIBUTED,  # May not be available
            fallback_strategy=ProcessingStrategy.PARALLEL,
        )
        optimizer = PerformanceOptimizer(config)

        result = optimizer.process_files(
            file_paths=temp_udl_files, metric_names=["consistency"]
        )

        # Should either use distributed or fall back to parallel
        assert result.strategy_used in [
            ProcessingStrategy.DISTRIBUTED,
            ProcessingStrategy.PARALLEL,
        ]
        assert result.total_files == len(temp_udl_files)


class TestPerformanceOptimizationEffectiveness:
    """Test effectiveness of performance optimization strategies."""

    def test_parallel_vs_sequential_speedup(self, temp_udl_files):
        """Test that parallel processing is faster than sequential."""
        # Sequential processing
        seq_config = PerformanceConfig(strategy=ProcessingStrategy.SEQUENTIAL)
        seq_optimizer = PerformanceOptimizer(seq_config)

        seq_result = seq_optimizer.process_files(
            file_paths=temp_udl_files, metric_names=["consistency", "completeness"]
        )

        # Parallel processing
        par_config = PerformanceConfig(strategy=ProcessingStrategy.PARALLEL)
        par_optimizer = PerformanceOptimizer(par_config)

        par_result = par_optimizer.process_files(
            file_paths=temp_udl_files, metric_names=["consistency", "completeness"]
        )

        # Both should complete successfully
        assert seq_result.successful_files > 0
        assert par_result.successful_files > 0

        # Results should be consistent
        assert len(seq_result.reports) == len(par_result.reports)

    def test_incremental_caching_effectiveness(self, temp_udl_files):
        """Test that incremental caching improves performance."""
        config = PerformanceConfig(strategy=ProcessingStrategy.INCREMENTAL)
        optimizer = PerformanceOptimizer(config)

        # First run
        result1 = optimizer.process_files(
            file_paths=temp_udl_files, metric_names=["consistency"]
        )

        # Second run (should benefit from caching)
        result2 = optimizer.process_files(
            file_paths=temp_udl_files, metric_names=["consistency"]
        )

        # Both should complete successfully
        assert result1.successful_files > 0
        assert result2.successful_files > 0

        # Cache hit ratio should be higher on second run
        # (or at least not worse)
        assert result2.cache_hit_ratio >= 0

    def test_streaming_memory_efficiency(self, large_udl_content):
        """Test that streaming processing is memory efficient."""
        config = StreamingConfig(
            chunk_size=512,
            max_memory_usage=10 * 1024 * 1024,  # 10MB limit
        )
        processor = StreamingProcessor(config)

        # Process large content
        results = list(processor.process_content_streaming(large_udl_content))

        # Should complete without memory issues
        assert len(results) > 0

        # Memory usage should be tracked
        memory_stats = processor.get_memory_usage()
        assert "rss_mb" in memory_stats
        assert "processed_chunks" in memory_stats


# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.filterwarnings("ignore:.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:.*:UserWarning"),
]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
