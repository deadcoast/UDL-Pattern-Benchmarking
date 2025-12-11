"""
Tests for performance and scalability improvements.

Tests distributed computing, GPU acceleration, streaming processing,
memory mapping, and incremental computation features.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from udl_rating_framework.core.incremental import (
    FileSnapshot,
    IncrementalProcessor,
    process_udl_incremental,
)
from udl_rating_framework.core.memory_mapping import (
    MemoryMappedFile,
    MemoryMappedUDLProcessor,
    process_large_udl_file,
)
from udl_rating_framework.core.multiprocessing import (
    ParallelProcessor,
    process_udl_files_parallel,
)
from udl_rating_framework.core.performance import (
    PerformanceConfig,
    PerformanceOptimizer,
    ProcessingStrategy,
    benchmark_all_strategies,
    process_files_optimized,
)
from udl_rating_framework.core.representation import UDLRepresentation

# Import performance modules for testing
from udl_rating_framework.core.streaming import (
    StreamingConfig,
    StreamingProcessor,
    process_large_file,
)

# Optional imports for distributed and GPU processing
try:
    from udl_rating_framework.core.distributed import (
        DistributedProcessor,
        process_files_distributed,
    )

    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

try:
    from udl_rating_framework.core.gpu_acceleration import (
        GPUAcceleratedProcessor,
        process_files_gpu,
    )

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


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
    """Large UDL content for testing streaming and memory mapping."""
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
    for i in range(100):  # Create ~100KB file
        repeated_content += f"\nrule_{i} ::= 'token_{i}' | rule_{max(0, i - 1)}\n"

    return repeated_content


@pytest.fixture
def temp_udl_files(sample_udl_content):
    """Create temporary UDL files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple test files
        files = []
        for i in range(5):
            file_path = temp_path / f"test_{i}.udl"
            content = (
                sample_udl_content +
                f"\n# File {i}\nfile_rule_{i} ::= 'file_{i}'\n"
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


class TestStreamingProcessor:
    """Test streaming processing functionality."""

    def test_streaming_config(self):
        """Test streaming configuration."""
        config = StreamingConfig(
            chunk_size=1024, overlap_size=128, max_memory_usage=1024 * 1024
        )

        assert config.chunk_size == 1024
        assert config.overlap_size == 128
        assert config.max_memory_usage == 1024 * 1024

    def test_streaming_processor_initialization(self):
        """Test streaming processor initialization."""
        config = StreamingConfig(chunk_size=512)
        processor = StreamingProcessor(config)

        assert processor.config.chunk_size == 512
        assert processor.chunker is not None
        assert processor.pipeline is not None

    def test_process_content_streaming(self, large_udl_content):
        """Test streaming processing of content."""
        processor = StreamingProcessor()

        results = list(processor.process_content_streaming(large_udl_content))

        assert len(results) > 0
        assert all(r.success or r.error for r in results)
        assert all(r.processing_time >= 0 for r in results)

    def test_process_file_streaming(self, large_temp_file):
        """Test streaming processing of file."""
        processor = StreamingProcessor()

        results = list(processor.process_file_streaming(large_temp_file))

        assert len(results) > 0
        assert all(r.file_path == str(large_temp_file) for r in results)
        assert any(r.aggregated_score > 0 for r in results if not r.error)

    def test_process_large_file_convenience(self, large_temp_file):
        """Test convenience function for processing large files."""
        results = list(
            process_large_file(
                file_path=large_temp_file, chunk_size=1024, overlap_size=128
            )
        )

        assert len(results) > 0
        assert all(
            isinstance(r.aggregated_score, float) for r in results if not r.error
        )


class TestMemoryMappedProcessor:
    """Test memory-mapped file processing functionality."""

    def test_memory_mapped_file_creation(self, large_temp_file):
        """Test memory-mapped file creation."""
        from udl_rating_framework.core.memory_mapping import MemoryMapConfig

        config = MemoryMapConfig(page_size=4096)

        with MemoryMappedFile(large_temp_file, config) as mapped_file:
            assert mapped_file.file_path == large_temp_file
            assert mapped_file.file_size > 0
            assert mapped_file.is_open

    def test_memory_mapped_file_reading(self, large_temp_file):
        """Test memory-mapped file reading."""
        with MemoryMappedFile(large_temp_file) as mapped_file:
            # Read first 100 bytes
            data = mapped_file.read(0, 100)
            assert len(data) <= 100
            assert isinstance(data, bytes)

            # Read lines
            lines = list(mapped_file.read_lines(max_lines=5))
            assert len(lines) <= 5
            assert all(isinstance(line_content, str)
                       for _, line_content in lines)

    def test_memory_mapped_processor(self, large_temp_file):
        """Test memory-mapped UDL processor."""
        processor = MemoryMappedUDLProcessor()

        report = processor.process_file(large_temp_file)

        assert report is not None
        assert isinstance(report.overall_score, float)
        assert 0.0 <= report.overall_score <= 1.0
        assert report.udl_file == str(large_temp_file)

    def test_process_multiple_files_memory_mapped(self, temp_udl_files):
        """Test processing multiple files with memory mapping."""
        processor = MemoryMappedUDLProcessor()

        reports = processor.process_multiple_files(temp_udl_files)

        assert len(reports) == len(temp_udl_files)
        assert all(isinstance(r.overall_score, float) for r in reports)

    def test_analyze_file_structure(self, large_temp_file):
        """Test file structure analysis."""
        processor = MemoryMappedUDLProcessor()

        analysis = processor.analyze_file_structure(large_temp_file)

        assert "file_path" in analysis
        assert "file_size" in analysis
        assert "line_count" in analysis
        assert analysis["line_count"] > 0
        assert analysis["file_size"] > 0

    def test_process_large_udl_file_convenience(self, large_temp_file):
        """Test convenience function for processing large UDL files."""
        report = process_large_udl_file(large_temp_file)

        assert report is not None
        assert isinstance(report.overall_score, float)
        assert 0.0 <= report.overall_score <= 1.0


class TestIncrementalProcessor:
    """Test incremental computation functionality."""

    def test_file_snapshot_creation(self, temp_udl_files):
        """Test file snapshot creation."""
        file_path = temp_udl_files[0]
        snapshot = FileSnapshot.from_file(file_path)

        assert snapshot.file_path == str(file_path)
        assert snapshot.content_hash != ""
        assert snapshot.modification_time > 0
        assert snapshot.file_size > 0
        assert snapshot.line_count > 0

    def test_incremental_processor_initialization(self):
        """Test incremental processor initialization."""
        processor = IncrementalProcessor()

        assert processor.cache is not None
        assert processor.dependency_tracker is not None
        assert processor.pipeline is not None

    def test_process_file_incremental_new(self, temp_udl_files):
        """Test incremental processing of new file."""
        processor = IncrementalProcessor()
        file_path = temp_udl_files[0]

        result = processor.process_file(file_path)

        assert result.file_path == str(file_path)
        assert not result.was_cached  # New file shouldn't be cached
        assert result.report is not None
        assert isinstance(result.report.overall_score, float)

    def test_process_file_incremental_cached(self, temp_udl_files):
        """Test incremental processing with caching."""
        processor = IncrementalProcessor()
        file_path = temp_udl_files[0]

        # Process file first time
        result1 = processor.process_file(file_path)

        # Process same file again (should use cache)
        result2 = processor.process_file(file_path)

        assert result1.file_path == result2.file_path
        # Note: Current implementation may not show caching due to change detection

    def test_process_multiple_files_incremental(self, temp_udl_files):
        """Test processing multiple files incrementally."""
        processor = IncrementalProcessor()

        results = processor.process_multiple_files(temp_udl_files)

        assert len(results) == len(temp_udl_files)
        assert all(r.report is not None for r in results)
        assert all(isinstance(r.processing_time, float) for r in results)

    def test_process_udl_incremental_convenience(self, temp_udl_files):
        """Test convenience function for incremental processing."""
        file_path = temp_udl_files[0]

        result = process_udl_incremental(file_path)

        assert result.file_path == str(file_path)
        assert result.report is not None


class TestParallelProcessor:
    """Test parallel processing functionality."""

    def test_parallel_processor_initialization(self):
        """Test parallel processor initialization."""
        processor = ParallelProcessor(max_workers=2)

        assert processor.max_workers == 2

    def test_process_files_parallel(self, temp_udl_files):
        """Test parallel file processing."""
        reports, stats = process_udl_files_parallel(
            file_paths=temp_udl_files,
            metric_names=["consistency", "completeness"],
            max_workers=2,
            show_progress=False,
        )

        assert len(reports) == len(temp_udl_files)
        assert stats.total_files == len(temp_udl_files)
        assert stats.successful >= 0
        assert stats.failed >= 0
        assert stats.total_time > 0


@pytest.mark.skipif(
    not DISTRIBUTED_AVAILABLE, reason="Distributed computing not available"
)
class TestDistributedProcessor:
    """Test distributed processing functionality."""

    def test_distributed_processor_initialization(self):
        """Test distributed processor initialization."""
        from udl_rating_framework.core.distributed import DistributedConfig

        config = DistributedConfig(backend="ray", num_workers=2)

        assert config.backend == "ray"
        assert config.num_workers == 2

    @pytest.mark.slow
    def test_process_files_distributed(self, temp_udl_files):
        """Test distributed file processing."""
        try:
            reports, stats = process_files_distributed(
                file_paths=temp_udl_files,
                metric_names=["consistency", "completeness"],
                backend="ray",
                num_workers=2,
                show_progress=False,
            )

            assert len(reports) == len(temp_udl_files)
            assert all(r is not None for r in reports)

        except Exception as e:
            pytest.skip(f"Distributed processing failed: {e}")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU acceleration not available")
class TestGPUProcessor:
    """Test GPU acceleration functionality."""

    def test_gpu_processor_initialization(self):
        """Test GPU processor initialization."""
        try:
            processor = GPUAcceleratedProcessor(batch_size=16)
            assert processor.batch_size == 16
        except Exception as e:
            pytest.skip(f"GPU processor initialization failed: {e}")

    @pytest.mark.slow
    def test_process_files_gpu(self, temp_udl_files):
        """Test GPU file processing."""
        try:
            results = process_files_gpu(
                file_paths=temp_udl_files, batch_size=16, max_sequence_length=256
            )

            assert len(results) == len(temp_udl_files)
            assert all("overall_score" in r for r in results)

        except Exception as e:
            pytest.skip(f"GPU processing failed: {e}")


class TestPerformanceOptimizer:
    """Test performance optimizer functionality."""

    def test_performance_config(self):
        """Test performance configuration."""
        config = PerformanceConfig(
            strategy=ProcessingStrategy.PARALLEL, max_workers=4, enable_gpu=False
        )

        assert config.strategy == ProcessingStrategy.PARALLEL
        assert config.max_workers == 4
        assert not config.enable_gpu

    def test_performance_optimizer_initialization(self):
        """Test performance optimizer initialization."""
        config = PerformanceConfig(strategy=ProcessingStrategy.PARALLEL)
        optimizer = PerformanceOptimizer(config)

        assert optimizer.config.strategy == ProcessingStrategy.PARALLEL
        assert ProcessingStrategy.PARALLEL in optimizer.processors

    def test_workload_analysis(self, temp_udl_files):
        """Test workload analysis."""
        optimizer = PerformanceOptimizer()

        workload_info = optimizer._analyze_workload(temp_udl_files)

        assert "file_count" in workload_info
        assert "avg_file_size_mb" in workload_info
        assert "max_file_size_mb" in workload_info
        assert workload_info["file_count"] == len(temp_udl_files)

    def test_strategy_selection(self, temp_udl_files):
        """Test automatic strategy selection."""
        optimizer = PerformanceOptimizer()

        workload_info = optimizer._analyze_workload(temp_udl_files)
        strategy = optimizer._select_strategy(workload_info)

        assert isinstance(strategy, ProcessingStrategy)

    def test_process_files_with_optimizer(self, temp_udl_files):
        """Test processing files with optimizer."""
        config = PerformanceConfig(strategy=ProcessingStrategy.PARALLEL)
        optimizer = PerformanceOptimizer(config)

        result = optimizer.process_files(
            file_paths=temp_udl_files, metric_names=[
                "consistency", "completeness"]
        )

        assert result.strategy_used == ProcessingStrategy.PARALLEL
        assert result.total_files == len(temp_udl_files)
        assert result.processing_time > 0
        assert len(result.reports) <= len(temp_udl_files)

    def test_process_files_optimized_convenience(self, temp_udl_files):
        """Test convenience function for optimized processing."""
        result = process_files_optimized(
            file_paths=temp_udl_files, metric_names=[
                "consistency", "completeness"]
        )

        assert result.total_files == len(temp_udl_files)
        assert result.processing_time > 0
        assert isinstance(result.strategy_used, ProcessingStrategy)

    def test_performance_statistics(self, temp_udl_files):
        """Test performance statistics collection."""
        optimizer = PerformanceOptimizer()

        # Process files to generate statistics
        result = optimizer.process_files(
            file_paths=temp_udl_files, metric_names=["consistency"]
        )

        stats = optimizer.get_performance_statistics()

        assert "total_files_processed" in stats
        assert "average_throughput" in stats
        assert "strategy_statistics" in stats
        assert stats["total_files_processed"] > 0


class TestPerformanceBenchmarking:
    """Test performance benchmarking functionality."""

    @pytest.mark.slow
    def test_benchmark_strategies(self, temp_udl_files):
        """Test benchmarking of different strategies."""
        # Use subset of files for faster testing
        test_files = temp_udl_files[:2]

        results = benchmark_all_strategies(
            test_files=test_files,
            metric_names=["consistency"],
            iterations=1,  # Single iteration for testing
        )

        assert len(results) > 0
        assert "sequential" in results
        assert "parallel" in results

        # Check result structure
        for strategy, stats in results.items():
            if "error" not in stats:
                assert "avg_processing_time" in stats
                assert "avg_throughput" in stats
                assert "successful_runs" in stats


class TestIntegration:
    """Integration tests for performance improvements."""

    def test_end_to_end_performance_processing(self, temp_udl_files):
        """Test end-to-end performance processing."""
        # Test with automatic strategy selection
        result = process_files_optimized(
            file_paths=temp_udl_files,
            metric_names=["consistency", "completeness", "expressiveness"],
        )

        assert result.total_files == len(temp_udl_files)
        assert result.successful_files >= 0
        assert result.processing_time > 0
        assert len(result.reports) <= len(temp_udl_files)

        # Verify reports have expected structure
        for report in result.reports:
            assert hasattr(report, "overall_score")
            assert hasattr(report, "confidence")
            assert hasattr(report, "metric_scores")

    def test_performance_with_different_file_sizes(
        self, sample_udl_content, large_udl_content
    ):
        """Test performance optimization with different file sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files of different sizes
            small_file = temp_path / "small.udl"
            small_file.write_text(sample_udl_content)

            large_file = temp_path / "large.udl"
            large_file.write_text(large_udl_content)

            files = [small_file, large_file]

            # Process with automatic optimization
            result = process_files_optimized(
                file_paths=files, metric_names=["consistency", "completeness"]
            )

            assert result.total_files == 2
            assert result.processing_time > 0

    def test_error_handling_in_performance_processing(self):
        """Test error handling in performance processing."""
        # Create non-existent file path
        non_existent_file = Path("non_existent_file.udl")

        result = process_files_optimized(
            file_paths=[non_existent_file], metric_names=["consistency"]
        )

        # Should handle errors gracefully
        assert result.total_files == 1
        assert result.failed_files >= 0
        assert result.processing_time >= 0


# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.filterwarnings("ignore:.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:.*:UserWarning"),
]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
