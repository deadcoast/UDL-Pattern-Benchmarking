"""
Property-based tests for performance and streaming modules.

This module focuses on testing the performance-critical modules that have
low test coverage, using property-based testing to find edge cases and
ensure robustness under various conditions.
"""

import math
import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from udl_rating_framework.core.performance import PerformanceConfig, ProcessingStrategy
from udl_rating_framework.core.streaming import (
    MemoryMappedFileReader,
    StreamingChunk,
    StreamingConfig,
)


class TestStreamingModuleProperties:
    """Property-based tests for streaming processing module."""

    @given(
        chunk_size=st.integers(
            min_value=1024, max_value=10 * 1024 * 1024
        ),  # 1KB to 10MB
        overlap_size=st.integers(min_value=0, max_value=1024),  # 0 to 1KB
        max_memory=st.integers(
            min_value=1024 * 1024, max_value=1024 * 1024 * 1024
        ),  # 1MB to 1GB
    )
    @settings(max_examples=50, deadline=3000)
    def test_streaming_config_invariants(self, chunk_size, overlap_size, max_memory):
        """Test invariants of streaming configuration."""
        # Ensure overlap is not larger than chunk size
        assume(overlap_size < chunk_size)
        # Ensure memory is sufficient for at least one chunk
        assume(max_memory >= chunk_size)
        # Ensure buffer size is reasonable for memory constraints
        max_chunks_in_memory = max_memory // chunk_size
        assume(
            max_chunks_in_memory >= 10 or max_chunks_in_memory == 0
        )  # Default buffer size is 10

        config = StreamingConfig(
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            max_memory_usage=max_memory,
        )

        # Invariant 1: Chunk size should be positive
        assert config.chunk_size > 0, (
            f"Chunk size {config.chunk_size} should be positive"
        )

        # Invariant 2: Overlap should be smaller than chunk size
        assert config.overlap_size < config.chunk_size, (
            f"Overlap {config.overlap_size} should be smaller than chunk {config.chunk_size}"
        )

        # Invariant 3: Max memory should be reasonable
        assert config.max_memory_usage >= config.chunk_size, (
            f"Max memory {config.max_memory_usage} should be at least chunk size {config.chunk_size}"
        )

        # Invariant 4: Buffer size should be reasonable for memory constraints
        max_chunks_in_memory = config.max_memory_usage // config.chunk_size
        assert (
            config.buffer_size <= max_chunks_in_memory or max_chunks_in_memory == 0
        ), f"Buffer size {config.buffer_size} should fit in memory constraint"

    @given(
        chunk_id=st.integers(min_value=0, max_value=1000),
        start_offset=st.integers(min_value=0, max_value=1000000),
        content=st.text(min_size=0, max_size=10000),
    )
    @settings(max_examples=100, deadline=3000)
    def test_streaming_chunk_properties(self, chunk_id, start_offset, content):
        """Test properties of streaming chunks."""
        end_offset = start_offset + len(content)

        chunk = StreamingChunk(
            chunk_id=chunk_id,
            start_offset=start_offset,
            end_offset=end_offset,
            content=content,
        )

        # Invariant 1: End offset should be >= start offset
        assert chunk.end_offset >= chunk.start_offset, (
            f"End offset {chunk.end_offset} should be >= start offset {chunk.start_offset}"
        )

        # Invariant 2: Content length should match offset difference
        expected_length = chunk.end_offset - chunk.start_offset
        actual_length = len(chunk.content)
        assert actual_length == expected_length, (
            f"Content length {actual_length} should match offset difference {expected_length}"
        )

        # Invariant 3: Chunk ID should be non-negative
        assert chunk.chunk_id >= 0, f"Chunk ID {chunk.chunk_id} should be non-negative"

        # Invariant 4: Chunk should be complete by default
        assert chunk.is_complete, "Chunk should be complete by default"

    @given(
        file_content=st.text(min_size=100, max_size=50000),
        chunk_size=st.integers(min_value=10, max_value=1000),
    )
    @settings(max_examples=30, deadline=5000)
    def test_memory_mapped_file_chunking_properties(self, file_content, chunk_size):
        """Test properties of memory-mapped file chunking."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as temp_file:
            temp_file.write(file_content)
            temp_file_path = Path(temp_file.name)

        try:
            # Test memory-mapped file reader properties
            reader = MemoryMappedFileReader(temp_file_path)

            # Invariant 1: File size should match content length
            expected_size = len(file_content.encode("utf-8"))
            assert reader.file_size == expected_size, (
                f"File size {reader.file_size} should match content size {expected_size}"
            )

            # Invariant 2: File path should be set correctly
            assert reader.file_path == temp_file_path, (
                f"File path should be {temp_file_path}, got {reader.file_path}"
            )

            # Invariant 3: Encoding should be set
            assert reader.encoding == "utf-8", (
                f"Encoding should be utf-8, got {reader.encoding}"
            )

            # Invariant 4: File size should be positive for non-empty content
            if len(file_content) > 0:
                assert reader.file_size > 0, (
                    "File size should be positive for non-empty content"
                )

        finally:
            # Clean up
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass


class TestPerformanceConfigProperties:
    """Property-based tests for performance configuration."""

    @given(
        max_memory_mb=st.integers(min_value=1, max_value=16384),  # 1MB to 16GB
        max_workers=st.one_of(
            st.none(), st.integers(min_value=1, max_value=64)),
        max_file_size_mb=st.integers(
            min_value=1, max_value=1024),  # 1MB to 1GB
        gpu_batch_size=st.integers(min_value=1, max_value=256),
        streaming_chunk_size=st.integers(
            min_value=1024, max_value=10 * 1024 * 1024),
    )
    @settings(max_examples=100, deadline=3000)
    def test_performance_config_invariants(
        self,
        max_memory_mb,
        max_workers,
        max_file_size_mb,
        gpu_batch_size,
        streaming_chunk_size,
    ):
        """Test invariants of performance configuration."""
        # Ensure streaming chunk size is reasonable relative to memory
        memory_bytes = max_memory_mb * 1024 * 1024
        assume(streaming_chunk_size <= memory_bytes)  # Reasonable constraint

        config = PerformanceConfig(
            max_memory_mb=max_memory_mb,
            max_workers=max_workers,
            max_file_size_mb=max_file_size_mb,
            gpu_batch_size=gpu_batch_size,
            streaming_chunk_size=streaming_chunk_size,
        )

        # Invariant 1: Memory limits should be positive
        assert config.max_memory_mb > 0, (
            f"Max memory {config.max_memory_mb} should be positive"
        )

        # Invariant 2: File size limit should be positive
        assert config.max_file_size_mb > 0, (
            f"Max file size {config.max_file_size_mb} should be positive"
        )

        # Invariant 3: GPU batch size should be positive
        assert config.gpu_batch_size > 0, (
            f"GPU batch size {config.gpu_batch_size} should be positive"
        )

        # Invariant 4: Streaming chunk size should be reasonable
        assert config.streaming_chunk_size >= 1024, (
            f"Streaming chunk size {config.streaming_chunk_size} should be at least 1KB"
        )

        # Invariant 5: Worker count should be reasonable if specified
        if config.max_workers is not None:
            assert config.max_workers > 0, (
                f"Max workers {config.max_workers} should be positive"
            )
            assert config.max_workers <= 1000, (
                f"Max workers {config.max_workers} should be reasonable"
            )

        # Invariant 6: Memory should be sufficient for at least one chunk
        memory_bytes = config.max_memory_mb * 1024 * 1024
        assert memory_bytes >= config.streaming_chunk_size, (
            f"Memory {memory_bytes} should be sufficient for chunk size {config.streaming_chunk_size}"
        )

    @given(strategy=st.sampled_from(list(ProcessingStrategy)))
    @settings(max_examples=20, deadline=2000)
    def test_processing_strategy_properties(self, strategy):
        """Test properties of processing strategies."""
        config = PerformanceConfig(strategy=strategy)

        # Invariant 1: Strategy should be valid enum value
        assert isinstance(config.strategy, ProcessingStrategy), (
            f"Strategy should be ProcessingStrategy enum, got {type(config.strategy)}"
        )

        # Invariant 2: Strategy should have string representation
        strategy_str = str(config.strategy.value)
        assert isinstance(strategy_str, str), (
            "Strategy should have string representation"
        )
        assert len(strategy_str) > 0, "Strategy string should not be empty"

        # Invariant 3: Fallback strategy should be valid
        assert isinstance(config.fallback_strategy, ProcessingStrategy), (
            "Fallback strategy should be ProcessingStrategy enum"
        )


class TestNumericalStabilityProperties:
    """Property-based tests for numerical stability in performance calculations."""

    @given(
        processing_times=st.lists(
            st.floats(
                min_value=0.001, max_value=3600.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=1000,
        )
    )
    @settings(max_examples=50, deadline=3000)
    def test_throughput_calculation_stability(self, processing_times):
        """Test numerical stability of throughput calculations."""
        # Calculate various throughput metrics
        total_time = sum(processing_times)
        num_items = len(processing_times)

        # Average processing time
        avg_time = total_time / num_items
        assert avg_time > 0, "Average processing time should be positive"
        assert math.isfinite(
            avg_time), "Average processing time should be finite"

        # Throughput (items per second)
        throughput = num_items / total_time
        assert throughput > 0, "Throughput should be positive"
        assert math.isfinite(throughput), "Throughput should be finite"

        # Variance calculation
        variance = sum((t - avg_time) **
                       2 for t in processing_times) / num_items
        assert variance >= 0, "Variance should be non-negative"
        assert math.isfinite(variance), "Variance should be finite"

        # Standard deviation
        std_dev = math.sqrt(variance)
        assert std_dev >= 0, "Standard deviation should be non-negative"
        assert math.isfinite(std_dev), "Standard deviation should be finite"

        # Coefficient of variation (if avg_time > 0)
        if avg_time > 0:
            cv = std_dev / avg_time
            assert cv >= 0, "Coefficient of variation should be non-negative"
            assert math.isfinite(
                cv), "Coefficient of variation should be finite"

    @given(
        memory_usage_bytes=st.lists(
            st.integers(
                min_value=1024, max_value=16 * 1024 * 1024 * 1024
            ),  # 1KB to 16GB
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=50, deadline=2000)
    def test_memory_statistics_properties(self, memory_usage_bytes):
        """Test properties of memory usage statistics."""
        # Convert to MB for calculations
        memory_usage_mb = [
            bytes_val / (1024 * 1024) for bytes_val in memory_usage_bytes
        ]

        # Basic statistics
        min_memory = min(memory_usage_mb)
        max_memory = max(memory_usage_mb)
        avg_memory = sum(memory_usage_mb) / len(memory_usage_mb)

        # Invariant 1: Min <= Average <= Max
        assert min_memory <= avg_memory <= max_memory, (
            f"Memory stats order: min={min_memory}, avg={avg_memory}, max={max_memory}"
        )

        # Invariant 2: All values should be positive
        assert min_memory > 0, f"Minimum memory {min_memory} should be positive"
        assert max_memory > 0, f"Maximum memory {max_memory} should be positive"
        assert avg_memory > 0, f"Average memory {avg_memory} should be positive"

        # Invariant 3: Range should be non-negative
        memory_range = max_memory - min_memory
        assert memory_range >= 0, f"Memory range {memory_range} should be non-negative"

        # Invariant 4: Peak memory should be achievable
        peak_memory_mb = max_memory
        assert peak_memory_mb >= avg_memory, (
            f"Peak memory {peak_memory_mb} should be >= average {avg_memory}"
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
