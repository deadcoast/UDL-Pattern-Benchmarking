"""
Streaming processing for large UDL files in UDL Rating Framework.

Provides memory-efficient streaming processing capabilities for very large
UDL files that cannot fit in memory, with chunking, overlap handling,
and incremental result aggregation.
"""

import hashlib
import json
import logging
import mmap
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

# Import metrics to ensure they are registered in the MetricRegistry
import udl_rating_framework.core.metrics  # noqa: F401
from udl_rating_framework.core.caching import get_udl_cache
from udl_rating_framework.core.pipeline import QualityReport, RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming processing."""

    chunk_size: int = 1024 * 1024  # 1MB chunks
    overlap_size: int = 1024  # 1KB overlap between chunks
    max_memory_usage: int = 100 * 1024 * 1024  # 100MB max memory
    buffer_size: int = 10  # Number of chunks to buffer
    enable_caching: bool = True
    cache_chunk_results: bool = True
    progress_callback: Optional[Callable[[int, int], None]] = None
    error_handling: str = "continue"  # 'continue', 'stop', 'skip'
    encoding: str = "utf-8"
    line_ending: str = "\n"


@dataclass
class StreamingChunk:
    """A chunk of data for streaming processing."""

    chunk_id: int
    start_offset: int
    end_offset: int
    content: str
    overlap_before: str = ""
    overlap_after: str = ""
    is_complete: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingResult:
    """Result from streaming processing."""

    chunk_id: int
    file_path: str
    chunk_start: int
    chunk_end: int
    partial_results: List[QualityReport]
    aggregated_score: float
    confidence: float
    processing_time: float
    error: Optional[str] = None
    success: bool = True


class MemoryMappedFileReader:
    """
    Memory-mapped file reader for efficient large file processing.

    Provides streaming access to large files without loading entire
    content into memory.
    """

    def __init__(self, file_path: Path, encoding: str = "utf-8"):
        """
        Initialize memory-mapped file reader.

        Args:
            file_path: Path to file to read
            encoding: File encoding
        """
        self.file_path = file_path
        self.encoding = encoding
        self.file_size = file_path.stat().st_size
        self.file_handle = None
        self.mmap_handle = None

    def __enter__(self):
        """Context manager entry."""
        self.file_handle = open(self.file_path, "rb")
        self.mmap_handle = mmap.mmap(
            self.file_handle.fileno(), 0, access=mmap.ACCESS_READ
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.mmap_handle:
            self.mmap_handle.close()
        if self.file_handle:
            self.file_handle.close()

    def read_chunk(self, start: int, size: int) -> bytes:
        """
        Read a chunk of data from the file.

        Args:
            start: Start offset
            size: Chunk size

        Returns:
            Chunk data as bytes
        """
        if not self.mmap_handle:
            raise RuntimeError("File not opened")

        end = min(start + size, self.file_size)
        return self.mmap_handle[start:end]

    def find_line_boundaries(
        self, start: int, size: int, line_ending: str = "\n"
    ) -> Tuple[int, int]:
        """
        Find line boundaries for a chunk to avoid splitting lines.

        Args:
            start: Start offset
            size: Chunk size
            line_ending: Line ending character

        Returns:
            Tuple of (adjusted_start, adjusted_end)
        """
        if not self.mmap_handle:
            raise RuntimeError("File not opened")

        # Adjust start to beginning of line (unless at file start)
        adjusted_start = start
        if start > 0:
            # Look backwards for line ending
            search_start = max(0, start - 1000)  # Search up to 1000 chars back
            chunk = self.mmap_handle[search_start:start]
            line_ending_bytes = line_ending.encode(self.encoding)

            last_line_end = chunk.rfind(line_ending_bytes)
            if last_line_end != -1:
                adjusted_start = search_start + \
                    last_line_end + len(line_ending_bytes)

        # Adjust end to end of line
        end = min(start + size, self.file_size)
        adjusted_end = end

        if end < self.file_size:
            # Look forward for line ending
            search_end = min(
                self.file_size, end + 1000
            )  # Search up to 1000 chars forward
            chunk = self.mmap_handle[end:search_end]
            line_ending_bytes = line_ending.encode(self.encoding)

            next_line_end = chunk.find(line_ending_bytes)
            if next_line_end != -1:
                adjusted_end = end + next_line_end + len(line_ending_bytes)
            else:
                adjusted_end = search_end

        return adjusted_start, adjusted_end


class StreamingChunker:
    """
    Chunker for breaking large files into processable chunks.

    Handles overlap management and boundary detection to ensure
    proper processing of language constructs that span chunks.
    """

    def __init__(self, config: StreamingConfig):
        """
        Initialize streaming chunker.

        Args:
            config: Streaming configuration
        """
        self.config = config

    def chunk_file(self, file_path: Path) -> Iterator[StreamingChunk]:
        """
        Chunk a file into streaming chunks.

        Args:
            file_path: Path to file to chunk

        Yields:
            StreamingChunk objects
        """
        logger.info(
            f"Chunking file: {file_path} (size: {file_path.stat().st_size} bytes)"
        )

        with MemoryMappedFileReader(file_path, self.config.encoding) as reader:
            file_size = reader.file_size
            chunk_id = 0
            current_offset = 0

            while current_offset < file_size:
                # Calculate chunk boundaries
                chunk_start, chunk_end = reader.find_line_boundaries(
                    current_offset, self.config.chunk_size, self.config.line_ending
                )

                # Read chunk content
                chunk_data = reader.read_chunk(
                    chunk_start, chunk_end - chunk_start)
                chunk_content = chunk_data.decode(
                    self.config.encoding, errors="replace"
                )

                # Handle overlap
                overlap_before = ""
                overlap_after = ""

                if chunk_start > 0 and self.config.overlap_size > 0:
                    # Read overlap before
                    overlap_start = max(
                        0, chunk_start - self.config.overlap_size)
                    overlap_data = reader.read_chunk(
                        overlap_start, chunk_start - overlap_start
                    )
                    overlap_before = overlap_data.decode(
                        self.config.encoding, errors="replace"
                    )

                if chunk_end < file_size and self.config.overlap_size > 0:
                    # Read overlap after
                    overlap_data = reader.read_chunk(
                        chunk_end, min(self.config.overlap_size,
                                       file_size - chunk_end)
                    )
                    overlap_after = overlap_data.decode(
                        self.config.encoding, errors="replace"
                    )

                # Create chunk
                chunk = StreamingChunk(
                    chunk_id=chunk_id,
                    start_offset=chunk_start,
                    end_offset=chunk_end,
                    content=chunk_content,
                    overlap_before=overlap_before,
                    overlap_after=overlap_after,
                    is_complete=True,
                    metadata={
                        "file_path": str(file_path),
                        "file_size": file_size,
                        "chunk_size": len(chunk_content),
                    },
                )

                yield chunk

                chunk_id += 1
                current_offset = chunk_end

                # Update progress
                if self.config.progress_callback:
                    progress = min(100, (current_offset / file_size) * 100)
                    self.config.progress_callback(int(progress), 100)

        logger.info(f"File chunked into {chunk_id} chunks")

    def chunk_content(
        self, content: str, file_path: str = "memory"
    ) -> Iterator[StreamingChunk]:
        """
        Chunk content string into streaming chunks.

        Args:
            content: Content to chunk
            file_path: Virtual file path for metadata

        Yields:
            StreamingChunk objects
        """
        content_size = len(content.encode(self.config.encoding))
        chunk_id = 0
        current_offset = 0

        while current_offset < len(content):
            # Calculate chunk end
            chunk_end = min(current_offset +
                            self.config.chunk_size, len(content))

            # Adjust to line boundaries
            if chunk_end < len(content):
                # Find next line ending
                next_line = content.find(self.config.line_ending, chunk_end)
                if next_line != -1 and next_line - chunk_end < 1000:  # Don't go too far
                    chunk_end = next_line + len(self.config.line_ending)

            # Extract chunk content
            chunk_content = content[current_offset:chunk_end]

            # Handle overlap
            overlap_before = ""
            overlap_after = ""

            if current_offset > 0 and self.config.overlap_size > 0:
                overlap_start = max(0, current_offset -
                                    self.config.overlap_size)
                overlap_before = content[overlap_start:current_offset]

            if chunk_end < len(content) and self.config.overlap_size > 0:
                overlap_end = min(len(content), chunk_end +
                                  self.config.overlap_size)
                overlap_after = content[chunk_end:overlap_end]

            # Create chunk
            chunk = StreamingChunk(
                chunk_id=chunk_id,
                start_offset=current_offset,
                end_offset=chunk_end,
                content=chunk_content,
                overlap_before=overlap_before,
                overlap_after=overlap_after,
                is_complete=True,
                metadata={
                    "file_path": file_path,
                    "content_size": content_size,
                    "chunk_size": len(chunk_content),
                },
            )

            yield chunk

            chunk_id += 1
            current_offset = chunk_end


class StreamingProcessor:
    """
    Streaming processor for large UDL files.

    Processes files in chunks with memory management and result aggregation.
    """

    def __init__(
        self, config: Optional[StreamingConfig] = None, metric_names: List[str] = None
    ):
        """
        Initialize streaming processor.

        Args:
            config: Streaming configuration
            metric_names: List of metrics to compute
        """
        self.config = config or StreamingConfig()
        self.metric_names = metric_names or [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]

        # Initialize components
        self.chunker = StreamingChunker(self.config)
        self.pipeline = RatingPipeline(metric_names=self.metric_names)
        self.cache = get_udl_cache() if self.config.enable_caching else None

        # Processing state
        self.processed_chunks = 0
        self.total_chunks = 0
        self.results_buffer = deque(maxlen=self.config.buffer_size)

    def process_file_streaming(self, file_path: Path) -> Iterator[StreamingResult]:
        """
        Process a file using streaming.

        Args:
            file_path: Path to file to process

        Yields:
            StreamingResult objects
        """
        logger.info(f"Starting streaming processing of: {file_path}")

        start_time = time.time()
        self.processed_chunks = 0

        # Check cache for entire file first
        if self.cache:
            cached_result = self.cache.get_udl(file_path)
            if cached_result:
                logger.info(f"Using cached result for: {file_path}")
                # Convert cached result to streaming format
                yield StreamingResult(
                    chunk_id=0,
                    file_path=str(file_path),
                    chunk_start=0,
                    chunk_end=file_path.stat().st_size,
                    partial_results=[cached_result],
                    aggregated_score=cached_result.overall_score,
                    confidence=cached_result.confidence,
                    processing_time=0.0,
                )
                return

        # Process file in chunks
        chunk_results = []

        for chunk in self.chunker.chunk_file(file_path):
            try:
                result = self._process_chunk(chunk)
                chunk_results.append(result)
                yield result

                self.processed_chunks += 1

            except Exception as e:
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")

                if self.config.error_handling == "stop":
                    raise
                elif self.config.error_handling == "skip":
                    continue

                # Continue with error result
                error_result = StreamingResult(
                    chunk_id=chunk.chunk_id,
                    file_path=str(file_path),
                    chunk_start=chunk.start_offset,
                    chunk_end=chunk.end_offset,
                    partial_results=[],
                    aggregated_score=0.0,
                    confidence=0.0,
                    processing_time=0.0,
                    error=str(e),
                    success=False,
                )
                chunk_results.append(error_result)
                yield error_result

        total_time = time.time() - start_time
        logger.info(
            f"Streaming processing completed: {self.processed_chunks} chunks in {total_time:.3f}s"
        )

        # Cache aggregated result if enabled
        if self.cache and chunk_results:
            aggregated_result = self._aggregate_chunk_results(
                chunk_results, str(file_path)
            )
            if aggregated_result:
                self.cache.put_udl(file_path, aggregated_result)

    def process_content_streaming(
        self, content: str, file_path: str = "memory"
    ) -> Iterator[StreamingResult]:
        """
        Process content string using streaming.

        Args:
            content: Content to process
            file_path: Virtual file path for metadata

        Yields:
            StreamingResult objects
        """
        logger.info(
            f"Starting streaming processing of content ({len(content)} chars)")

        start_time = time.time()
        self.processed_chunks = 0

        for chunk in self.chunker.chunk_content(content, file_path):
            try:
                result = self._process_chunk(chunk)
                yield result

                self.processed_chunks += 1

            except Exception as e:
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")

                if self.config.error_handling == "stop":
                    raise
                elif self.config.error_handling == "skip":
                    continue

                # Continue with error result
                error_result = StreamingResult(
                    chunk_id=chunk.chunk_id,
                    file_path=file_path,
                    chunk_start=chunk.start_offset,
                    chunk_end=chunk.end_offset,
                    partial_results=[],
                    aggregated_score=0.0,
                    confidence=0.0,
                    processing_time=0.0,
                    error=str(e),
                    success=False,
                )
                yield error_result

        total_time = time.time() - start_time
        logger.info(
            f"Content streaming processing completed: {self.processed_chunks} chunks in {total_time:.3f}s"
        )

    def _process_chunk(self, chunk: StreamingChunk) -> StreamingResult:
        """
        Process a single chunk.

        Args:
            chunk: Chunk to process

        Returns:
            StreamingResult
        """
        start_time = time.time()

        # Check chunk cache
        chunk_cache_key = None
        if self.config.cache_chunk_results:
            chunk_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            chunk_cache_key = f"chunk_{chunk_hash}"

            # TODO: Implement chunk-level caching if needed

        # Combine content with overlap for better context
        full_content = chunk.overlap_before + chunk.content + chunk.overlap_after

        # Create UDL representation
        udl = UDLRepresentation(
            full_content, chunk.metadata.get("file_path", "unknown")
        )

        # Rate the chunk
        report = self.pipeline.compute_rating(udl)

        processing_time = time.time() - start_time

        # Create result
        result = StreamingResult(
            chunk_id=chunk.chunk_id,
            file_path=chunk.metadata.get("file_path", "unknown"),
            chunk_start=chunk.start_offset,
            chunk_end=chunk.end_offset,
            partial_results=[report],
            aggregated_score=report.overall_score,
            confidence=report.confidence,
            processing_time=processing_time,
        )

        return result

    def _aggregate_chunk_results(
        self, chunk_results: List[StreamingResult], file_path: str
    ) -> Optional[QualityReport]:
        """
        Aggregate results from multiple chunks into a single report.

        Args:
            chunk_results: List of chunk results
            file_path: File path for the aggregated result

        Returns:
            Aggregated QualityReport or None if no valid results
        """
        valid_results = [
            r for r in chunk_results if r.error is None and r.partial_results
        ]

        if not valid_results:
            return None

        # Aggregate scores using weighted average by chunk size
        total_weight = 0
        weighted_score = 0.0
        weighted_confidence = 0.0
        all_metric_scores = {}

        for result in valid_results:
            chunk_size = result.chunk_end - result.chunk_start
            weight = chunk_size

            for report in result.partial_results:
                weighted_score += report.overall_score * weight
                weighted_confidence += report.confidence * weight

                # Aggregate metric scores
                for metric_name, score in report.metric_scores.items():
                    if metric_name not in all_metric_scores:
                        all_metric_scores[metric_name] = []
                    all_metric_scores[metric_name].append((score, weight))

            total_weight += weight

        if total_weight == 0:
            return None

        # Calculate final aggregated scores
        final_score = weighted_score / total_weight
        final_confidence = weighted_confidence / total_weight

        final_metric_scores = {}
        for metric_name, scores_weights in all_metric_scores.items():
            weighted_sum = sum(score * weight for score,
                               weight in scores_weights)
            total_metric_weight = sum(weight for _, weight in scores_weights)
            final_metric_scores[metric_name] = (
                weighted_sum / total_metric_weight if total_metric_weight > 0 else 0.0
            )

        # Create aggregated report
        aggregated_report = QualityReport(
            overall_score=final_score,
            confidence=final_confidence,
            metric_scores=final_metric_scores,
            metric_formulas={},  # TODO: Aggregate formulas if needed
            computation_trace=[],  # TODO: Aggregate traces if needed
            error_bounds={},  # TODO: Aggregate error bounds if needed
            timestamp=time.time(),
            udl_file=file_path,
            errors=[],
        )

        return aggregated_report

    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "buffer_size": len(self.results_buffer),
            "processed_chunks": self.processed_chunks,
        }


class AsyncStreamingProcessor:
    """
    Asynchronous streaming processor for concurrent chunk processing.

    Processes multiple chunks concurrently while maintaining memory limits.
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        metric_names: List[str] = None,
        max_concurrent_chunks: int = 4,
    ):
        """
        Initialize async streaming processor.

        Args:
            config: Streaming configuration
            metric_names: List of metrics to compute
            max_concurrent_chunks: Maximum concurrent chunks to process
        """
        self.config = config or StreamingConfig()
        self.metric_names = metric_names or [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]
        self.max_concurrent_chunks = max_concurrent_chunks

        # Initialize components
        self.chunker = StreamingChunker(self.config)
        self.pipeline = RatingPipeline(metric_names=self.metric_names)

        # Threading components
        self.chunk_queue = Queue(maxsize=self.config.buffer_size)
        self.result_queue = Queue()
        self.worker_threads = []
        self.stop_event = threading.Event()

    def process_file_async(self, file_path: Path) -> Iterator[StreamingResult]:
        """
        Process file asynchronously with concurrent chunk processing.

        Args:
            file_path: Path to file to process

        Yields:
            StreamingResult objects
        """
        logger.info(f"Starting async streaming processing of: {file_path}")

        # Start worker threads
        self._start_workers()

        try:
            # Producer: Generate chunks
            chunk_producer = threading.Thread(
                target=self._produce_chunks, args=(file_path,)
            )
            chunk_producer.start()

            # Consumer: Yield results
            results_yielded = 0
            while True:
                try:
                    result = self.result_queue.get(timeout=1.0)
                    if result is None:  # Sentinel value indicating completion
                        break

                    yield result
                    results_yielded += 1

                except Empty:
                    # Check if producer is still alive
                    if not chunk_producer.is_alive() and self.result_queue.empty():
                        break

            # Wait for producer to complete
            chunk_producer.join()

            logger.info(
                f"Async streaming processing completed: {results_yielded} results"
            )

        finally:
            # Stop workers
            self._stop_workers()

    def _start_workers(self) -> None:
        """Start worker threads for chunk processing."""
        self.stop_event.clear()

        for i in range(self.max_concurrent_chunks):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.start()
            self.worker_threads.append(worker)

        logger.debug(f"Started {len(self.worker_threads)} worker threads")

    def _stop_workers(self) -> None:
        """Stop worker threads."""
        self.stop_event.set()

        # Add sentinel values to wake up workers
        for _ in self.worker_threads:
            try:
                self.chunk_queue.put(None, timeout=1.0)
            except:
                pass

        # Wait for workers to complete
        for worker in self.worker_threads:
            worker.join(timeout=5.0)

        self.worker_threads.clear()
        logger.debug("Stopped worker threads")

    def _produce_chunks(self, file_path: Path) -> None:
        """Producer thread function to generate chunks."""
        try:
            for chunk in self.chunker.chunk_file(file_path):
                if self.stop_event.is_set():
                    break

                self.chunk_queue.put(chunk)

            # Signal completion
            self.result_queue.put(None)

        except Exception as e:
            logger.error(f"Error in chunk producer: {e}")
            self.result_queue.put(None)

    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread loop for processing chunks."""
        logger.debug(f"Worker {worker_id} started")

        while not self.stop_event.is_set():
            try:
                chunk = self.chunk_queue.get(timeout=1.0)
                if chunk is None:  # Sentinel value
                    break

                # Process chunk
                start_time = time.time()

                # Combine content with overlap
                full_content = (
                    chunk.overlap_before + chunk.content + chunk.overlap_after
                )

                # Create UDL representation
                udl = UDLRepresentation(
                    full_content, chunk.metadata.get("file_path", "unknown")
                )

                # Rate the chunk
                report = self.pipeline.compute_rating(udl)

                processing_time = time.time() - start_time

                # Create result
                result = StreamingResult(
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.metadata.get("file_path", "unknown"),
                    chunk_start=chunk.start_offset,
                    chunk_end=chunk.end_offset,
                    partial_results=[report],
                    aggregated_score=report.overall_score,
                    confidence=report.confidence,
                    processing_time=processing_time,
                )

                self.result_queue.put(result)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

                # Create error result if we have chunk info
                try:
                    error_result = StreamingResult(
                        chunk_id=getattr(chunk, "chunk_id", -1),
                        file_path=getattr(chunk, "metadata", {}).get(
                            "file_path", "unknown"
                        ),
                        chunk_start=getattr(chunk, "start_offset", 0),
                        chunk_end=getattr(chunk, "end_offset", 0),
                        partial_results=[],
                        aggregated_score=0.0,
                        confidence=0.0,
                        processing_time=0.0,
                        error=str(e),
                        success=False,
                    )
                    self.result_queue.put(error_result)
                except:
                    pass

        logger.debug(f"Worker {worker_id} stopped")


# Convenience functions
def process_large_file(
    file_path: Path,
    metric_names: List[str] = None,
    chunk_size: int = 1024 * 1024,
    overlap_size: int = 1024,
    enable_async: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Iterator[StreamingResult]:
    """
    Convenience function to process large UDL files with streaming.

    Args:
        file_path: Path to UDL file
        metric_names: List of metrics to compute
        chunk_size: Size of each chunk in bytes
        overlap_size: Overlap size between chunks
        enable_async: Whether to use async processing
        progress_callback: Optional progress callback

    Yields:
        StreamingResult objects
    """
    config = StreamingConfig(
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        progress_callback=progress_callback,
    )

    if enable_async:
        processor = AsyncStreamingProcessor(config, metric_names)
        yield from processor.process_file_async(file_path)
    else:
        processor = StreamingProcessor(config, metric_names)
        yield from processor.process_file_streaming(file_path)


def aggregate_streaming_results(
    results: List[StreamingResult],
) -> Optional[QualityReport]:
    """
    Aggregate streaming results into a single quality report.

    Args:
        results: List of streaming results

    Returns:
        Aggregated QualityReport or None
    """
    processor = StreamingProcessor()
    return processor._aggregate_chunk_results(results, "aggregated")
