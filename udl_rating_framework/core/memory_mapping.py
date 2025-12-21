"""
Memory-mapped file processing for UDL Rating Framework.

Provides efficient memory-mapped file access for large UDL files,
with optimized I/O operations and minimal memory footprint.
"""

import logging
import mmap
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from udl_rating_framework.core.pipeline import QualityReport
from udl_rating_framework.core.representation import UDLRepresentation

logger = logging.getLogger(__name__)


@dataclass
class MemoryMapConfig:
    """Configuration for memory-mapped file processing."""

    access_mode: int = mmap.ACCESS_READ
    page_size: int = 4096  # 4KB pages
    prefetch_size: int = 64 * 1024  # 64KB prefetch
    cache_size: int = 100  # Number of pages to cache
    enable_sequential_hint: bool = True
    enable_random_hint: bool = False
    enable_willneed_hint: bool = True
    thread_safe: bool = True


class MemoryMappedFile:
    """
    Memory-mapped file wrapper with optimized access patterns.

    Provides efficient access to large files using memory mapping
    with caching and access pattern optimization.
    """

    def __init__(self, file_path: Path, config: Optional[MemoryMapConfig] = None):
        """
        Initialize memory-mapped file.

        Args:
            file_path: Path to file to map
            config: Memory mapping configuration
        """
        self.file_path = file_path
        self.config = config or MemoryMapConfig()

        self.file_handle = None
        self.mmap_handle = None
        self.file_size = 0
        self.is_open = False

        # Thread safety
        self._lock = threading.RLock() if self.config.thread_safe else None

        # Access tracking
        self.access_count = 0
        self.last_access_time = 0.0
        self.access_pattern = []  # Track access patterns for optimization

        # Page cache
        self.page_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def open(self) -> None:
        """Open the memory-mapped file."""
        if self.is_open:
            return

        with self._get_lock():
            if self.is_open:  # Double-check after acquiring lock
                return

            try:
                self.file_handle = open(self.file_path, "rb")
                self.file_size = self.file_path.stat().st_size

                if self.file_size > 0:
                    self.mmap_handle = mmap.mmap(
                        self.file_handle.fileno(), 0, access=self.config.access_mode
                    )

                    # Set access hints if supported
                    self._set_access_hints()

                self.is_open = True
                logger.debug(
                    f"Opened memory-mapped file: {self.file_path} ({self.file_size} bytes)"
                )

            except Exception as e:
                self.close()
                raise RuntimeError(
                    f"Failed to open memory-mapped file {self.file_path}: {e}"
                )

    def close(self) -> None:
        """Close the memory-mapped file."""
        with self._get_lock():
            if self.mmap_handle:
                self.mmap_handle.close()
                self.mmap_handle = None

            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None

            self.is_open = False
            self.page_cache.clear()

            logger.debug(f"Closed memory-mapped file: {self.file_path}")

    def read(self, offset: int, size: int) -> bytes:
        """
        Read data from the memory-mapped file.

        Args:
            offset: Offset to read from
            size: Number of bytes to read

        Returns:
            Data bytes
        """
        if not self.is_open:
            self.open()

        with self._get_lock():
            self._track_access(offset, size)

            # Validate bounds
            if offset < 0 or offset >= self.file_size:
                raise ValueError(
                    f"Offset {offset} out of bounds (file size: {self.file_size})"
                )

            end_offset = min(offset + size, self.file_size)
            actual_size = end_offset - offset

            if actual_size <= 0:
                return b""

            # Check page cache first
            cached_data = self._get_from_cache(offset, actual_size)
            if cached_data is not None:
                self.cache_hits += 1
                return cached_data

            self.cache_misses += 1

            # Read from memory map
            data = self.mmap_handle[offset:end_offset]

            # Cache the data
            self._add_to_cache(offset, data)

            # Prefetch next data if sequential access detected
            if self._is_sequential_access():
                self._prefetch(end_offset)

            return data

    def read_lines(
        self,
        start_offset: int = 0,
        max_lines: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> Iterator[Tuple[int, str]]:
        """
        Read lines from the memory-mapped file.

        Args:
            start_offset: Starting offset
            max_lines: Maximum number of lines to read
            encoding: Text encoding

        Yields:
            Tuples of (line_offset, line_content)
        """
        if not self.is_open:
            self.open()

        current_offset = start_offset
        lines_read = 0

        while current_offset < self.file_size:
            if max_lines and lines_read >= max_lines:
                break

            # Find next line ending
            line_end = self._find_line_ending(current_offset)
            if line_end == -1:
                # Read rest of file as last line
                line_end = self.file_size

            # Read line data
            line_data = self.read(current_offset, line_end - current_offset)

            try:
                line_content = line_data.decode(encoding, errors="replace")
                # Remove line ending
                line_content = line_content.rstrip("\r\n")

                yield current_offset, line_content
                lines_read += 1

            except UnicodeDecodeError as e:
                logger.warning(
                    f"Unicode decode error at offset {current_offset}: {e}")

            current_offset = line_end + 1  # Skip line ending

    def find_pattern(
        self, pattern: bytes, start_offset: int = 0, max_matches: Optional[int] = None
    ) -> List[int]:
        """
        Find pattern occurrences in the memory-mapped file.

        Args:
            pattern: Pattern to search for
            start_offset: Starting offset for search
            max_matches: Maximum number of matches to find

        Returns:
            List of offsets where pattern was found
        """
        if not self.is_open:
            self.open()

        matches = []
        current_offset = start_offset

        while current_offset < self.file_size:
            if max_matches and len(matches) >= max_matches:
                break

            # Search in chunks to manage memory
            chunk_size = min(self.config.prefetch_size,
                             self.file_size - current_offset)
            chunk_data = self.read(current_offset, chunk_size)

            # Find pattern in chunk
            pattern_offset = chunk_data.find(pattern)

            if pattern_offset != -1:
                matches.append(current_offset + pattern_offset)
                current_offset += pattern_offset + len(pattern)
            else:
                # Move to next chunk with overlap to handle patterns spanning chunks
                overlap = len(pattern) - 1
                current_offset += chunk_size - overlap

        return matches

    def get_statistics(self) -> Dict[str, Any]:
        """Get access statistics."""
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0.0
        )

        return {
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "is_open": self.is_open,
            "access_count": self.access_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cached_pages": len(self.page_cache),
            "last_access_time": self.last_access_time,
        }

    def _get_lock(self):
        """Get thread lock or no-op context manager."""
        if self._lock:
            return self._lock
        else:

            @contextmanager
            def no_lock():
                yield

            return no_lock()

    def _set_access_hints(self) -> None:
        """Set memory access hints for optimization."""
        if not self.mmap_handle:
            return

        try:
            # Set sequential access hint if supported
            if self.config.enable_sequential_hint and hasattr(
                self.mmap_handle, "madvise"
            ):
                self.mmap_handle.madvise(mmap.MADV_SEQUENTIAL)

            # Set random access hint if supported
            if self.config.enable_random_hint and hasattr(self.mmap_handle, "madvise"):
                self.mmap_handle.madvise(mmap.MADV_RANDOM)

            # Set will-need hint if supported
            if self.config.enable_willneed_hint and hasattr(
                self.mmap_handle, "madvise"
            ):
                self.mmap_handle.madvise(mmap.MADV_WILLNEED)

        except (AttributeError, OSError) as e:
            logger.debug(f"Could not set memory access hints: {e}")

    def _track_access(self, offset: int, size: int) -> None:
        """Track access patterns for optimization."""
        self.access_count += 1
        self.last_access_time = time.time()

        # Keep limited history of access patterns
        self.access_pattern.append((offset, size, self.last_access_time))
        if len(self.access_pattern) > 100:  # Keep last 100 accesses
            self.access_pattern.pop(0)

    def _is_sequential_access(self) -> bool:
        """Detect if access pattern is sequential."""
        if len(self.access_pattern) < 2:
            return False

        # Check if last few accesses are sequential
        recent_accesses = self.access_pattern[-5:]  # Check last 5 accesses

        for i in range(1, len(recent_accesses)):
            prev_offset, prev_size, _ = recent_accesses[i - 1]
            curr_offset, _, _ = recent_accesses[i]

            # Check if current access starts near where previous ended
            if abs(curr_offset - (prev_offset + prev_size)) > self.config.page_size:
                return False

        return True

    def _prefetch(self, offset: int) -> None:
        """Prefetch data for sequential access optimization."""
        if not self.mmap_handle or offset >= self.file_size:
            return

        try:
            prefetch_size = min(self.config.prefetch_size,
                                self.file_size - offset)
            if prefetch_size > 0:
                # Touch the pages to bring them into memory
                _ = self.mmap_handle[
                    offset: offset + prefetch_size: self.config.page_size
                ]

        except Exception as e:
            logger.debug(f"Prefetch failed: {e}")

    def _get_from_cache(self, offset: int, size: int) -> Optional[bytes]:
        """Get data from page cache."""
        if not self.page_cache:
            return None

        # Simple cache lookup by offset
        cache_key = (offset, size)
        return self.page_cache.get(cache_key)

    def _add_to_cache(self, offset: int, data: bytes) -> None:
        """Add data to page cache."""
        if len(self.page_cache) >= self.config.cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = min(self.page_cache.keys())
            del self.page_cache[oldest_key]

        cache_key = (offset, len(data))
        self.page_cache[cache_key] = data

    def _find_line_ending(self, start_offset: int) -> int:
        """Find next line ending from given offset."""
        search_size = min(1024, self.file_size -
                          start_offset)  # Search in 1KB chunks

        while start_offset < self.file_size:
            chunk = self.read(start_offset, search_size)

            # Look for line endings
            for i, byte in enumerate(chunk):
                if byte in (ord("\n"), ord("\r")):
                    return start_offset + i

            start_offset += search_size
            search_size = min(1024, self.file_size - start_offset)

        return -1  # No line ending found

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MemoryMappedUDLProcessor:
    """
    UDL processor using memory-mapped files for efficient large file processing.

    Optimized for processing very large UDL files with minimal memory usage.
    """

    def __init__(
        self, config: Optional[MemoryMapConfig] = None, metric_names: List[str] = None
    ):
        """
        Initialize memory-mapped UDL processor.

        Args:
            config: Memory mapping configuration
            metric_names: List of metrics to compute
        """
        self.config = config or MemoryMapConfig()
        self.metric_names = metric_names or [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]

        # File cache
        self.file_cache = {}
        self.cache_lock = threading.RLock()

    def process_file(
        self, file_path: Path, chunk_size: int = 1024 * 1024, overlap_size: int = 1024
    ) -> QualityReport:
        """
        Process a UDL file using memory mapping.

        Args:
            file_path: Path to UDL file
            chunk_size: Size of processing chunks
            overlap_size: Overlap between chunks

        Returns:
            QualityReport for the file
        """
        logger.info(f"Processing file with memory mapping: {file_path}")

        start_time = time.time()

        with self._get_mapped_file(file_path) as mapped_file:
            # Process file in chunks
            chunk_reports = []
            current_offset = 0

            while current_offset < mapped_file.file_size:
                # Calculate chunk boundaries
                chunk_end = min(current_offset + chunk_size,
                                mapped_file.file_size)

                # Adjust to line boundaries to avoid splitting constructs
                if chunk_end < mapped_file.file_size:
                    # Find next line ending
                    line_end = mapped_file._find_line_ending(chunk_end)
                    if (
                        line_end != -1 and line_end - chunk_end < 1000
                    ):  # Don't go too far
                        chunk_end = line_end + 1

                # Read chunk with overlap
                chunk_start = max(0, current_offset - overlap_size)
                chunk_data = mapped_file.read(
                    chunk_start, chunk_end - chunk_start)

                # Process chunk
                try:
                    chunk_content = chunk_data.decode(
                        "utf-8", errors="replace")
                    chunk_report = self._process_chunk(
                        chunk_content, str(file_path))
                    chunk_reports.append(
                        (chunk_report, chunk_end - current_offset)
                    )  # Weight by size

                except Exception as e:
                    logger.error(
                        f"Error processing chunk at offset {current_offset}: {e}"
                    )

                current_offset = chunk_end

            # Aggregate chunk results
            final_report = self._aggregate_reports(
                chunk_reports, str(file_path))

            processing_time = time.time() - start_time
            logger.info(
                f"Memory-mapped processing completed in {processing_time:.3f}s")

            return final_report

    def process_multiple_files(
        self, file_paths: List[Path], max_concurrent: int = 4
    ) -> List[QualityReport]:
        """
        Process multiple files concurrently using memory mapping.

        Args:
            file_paths: List of file paths to process
            max_concurrent: Maximum concurrent files to process

        Returns:
            List of QualityReports
        """
        import concurrent.futures

        logger.info(f"Processing {len(file_paths)} files with memory mapping")

        reports = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent
        ) as executor:
            # Submit all files for processing
            future_to_path = {
                executor.submit(self.process_file, path): path for path in file_paths
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_path):
                file_path = future_to_path[future]

                try:
                    report = future.result()
                    reports.append(report)
                    logger.debug(f"Completed processing: {file_path}")

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

                    # Create error report
                    error_report = QualityReport(
                        overall_score=0.0,
                        confidence=0.0,
                        metric_scores={},
                        metric_formulas={},
                        computation_trace=[],
                        error_bounds={},
                        timestamp=time.time(),
                        udl_file=str(file_path),
                        errors=[str(e)],
                    )
                    reports.append(error_report)

        return reports

    def analyze_file_structure(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze file structure using memory mapping.

        Args:
            file_path: Path to file to analyze

        Returns:
            Dictionary with file structure analysis
        """
        with self._get_mapped_file(file_path) as mapped_file:
            analysis = {
                "file_path": str(file_path),
                "file_size": mapped_file.file_size,
                "line_count": 0,
                "non_empty_lines": 0,
                "max_line_length": 0,
                "avg_line_length": 0.0,
                "patterns": {},
                "encoding_issues": 0,
            }

            total_line_length = 0

            # Analyze line by line
            for line_offset, line_content in mapped_file.read_lines():
                analysis["line_count"] += 1

                if line_content.strip():
                    analysis["non_empty_lines"] += 1

                line_length = len(line_content)
                total_line_length += line_length
                analysis["max_line_length"] = max(
                    analysis["max_line_length"], line_length
                )

                # Look for common patterns
                if "::=" in line_content:
                    analysis["patterns"]["bnf_rules"] = (
                        analysis["patterns"].get("bnf_rules", 0) + 1
                    )
                if line_content.strip().startswith("#"):
                    analysis["patterns"]["comments"] = (
                        analysis["patterns"].get("comments", 0) + 1
                    )
                if "|" in line_content:
                    analysis["patterns"]["alternations"] = (
                        analysis["patterns"].get("alternations", 0) + 1
                    )

            if analysis["line_count"] > 0:
                analysis["avg_line_length"] = total_line_length / \
                    analysis["line_count"]

            return analysis

    def _get_mapped_file(self, file_path: Path) -> MemoryMappedFile:
        """Get cached memory-mapped file or create new one."""
        with self.cache_lock:
            cache_key = str(file_path)

            if cache_key not in self.file_cache:
                self.file_cache[cache_key] = MemoryMappedFile(
                    file_path, self.config)

            return self.file_cache[cache_key]

    def _process_chunk(self, content: str, file_path: str) -> QualityReport:
        """Process a single chunk of content."""
        from udl_rating_framework.core.pipeline import RatingPipeline

        # Create UDL representation
        udl = UDLRepresentation(content, file_path)

        # Create pipeline and rate
        pipeline = RatingPipeline(metric_names=self.metric_names)
        return pipeline.compute_rating(udl)

    def _aggregate_reports(
        self, weighted_reports: List[Tuple[QualityReport, int]], file_path: str
    ) -> QualityReport:
        """Aggregate multiple chunk reports into a single report."""
        if not weighted_reports:
            return QualityReport(
                overall_score=0.0,
                confidence=0.0,
                metric_scores={},
                metric_formulas={},
                computation_trace=[],
                error_bounds={},
                timestamp=time.time(),
                udl_file=file_path,
                errors=["No valid chunks processed"],
            )

        # Calculate weighted averages
        total_weight = sum(weight for _, weight in weighted_reports)

        if total_weight == 0:
            total_weight = len(weighted_reports)
            weighted_reports = [(report, 1) for report, _ in weighted_reports]

        # Aggregate scores
        weighted_overall = sum(
            report.overall_score * weight for report, weight in weighted_reports
        )
        weighted_confidence = sum(
            report.confidence * weight for report, weight in weighted_reports
        )

        overall_score = weighted_overall / total_weight
        confidence = weighted_confidence / total_weight

        # Aggregate metric scores
        all_metrics = set()
        for report, _ in weighted_reports:
            all_metrics.update(report.metric_scores.keys())

        aggregated_metrics = {}
        for metric in all_metrics:
            weighted_sum = sum(
                report.metric_scores.get(metric, 0.0) * weight
                for report, weight in weighted_reports
            )
            aggregated_metrics[metric] = weighted_sum / total_weight

        # Create aggregated report
        return QualityReport(
            overall_score=overall_score,
            confidence=confidence,
            metric_scores=aggregated_metrics,
            metric_formulas={},  # TODO: Aggregate formulas if needed
            computation_trace=[],  # TODO: Aggregate traces if needed
            error_bounds={},  # TODO: Aggregate error bounds if needed
            timestamp=time.time(),
            udl_file=file_path,
            errors=[],
        )

    def clear_cache(self) -> None:
        """Clear the file cache."""
        with self.cache_lock:
            for mapped_file in self.file_cache.values():
                mapped_file.close()
            self.file_cache.clear()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for all mapped files."""
        with self.cache_lock:
            stats = {
                "cached_files": len(self.file_cache),
                "total_cache_hits": 0,
                "total_cache_misses": 0,
                "total_access_count": 0,
                "files": {},
            }

            for file_path, mapped_file in self.file_cache.items():
                file_stats = mapped_file.get_statistics()
                stats["files"][file_path] = file_stats
                stats["total_cache_hits"] += file_stats["cache_hits"]
                stats["total_cache_misses"] += file_stats["cache_misses"]
                stats["total_access_count"] += file_stats["access_count"]

            total_accesses = stats["total_cache_hits"] + \
                stats["total_cache_misses"]
            stats["overall_cache_hit_rate"] = (
                stats["total_cache_hits"] / total_accesses
                if total_accesses > 0
                else 0.0
            )

            return stats


# Convenience functions
def process_large_udl_file(
    file_path: Path,
    metric_names: List[str] = None,
    chunk_size: int = 1024 * 1024,
    overlap_size: int = 1024,
) -> QualityReport:
    """
    Convenience function to process large UDL file with memory mapping.

    Args:
        file_path: Path to UDL file
        metric_names: List of metrics to compute
        chunk_size: Processing chunk size
        overlap_size: Overlap between chunks

    Returns:
        QualityReport for the file
    """
    processor = MemoryMappedUDLProcessor(metric_names=metric_names)
    return processor.process_file(file_path, chunk_size, overlap_size)


def analyze_udl_file_structure(file_path: Path) -> Dict[str, Any]:
    """
    Convenience function to analyze UDL file structure.

    Args:
        file_path: Path to UDL file

    Returns:
        File structure analysis
    """
    processor = MemoryMappedUDLProcessor()
    return processor.analyze_file_structure(file_path)
