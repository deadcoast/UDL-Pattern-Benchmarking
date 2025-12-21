"""
Batch processing workflows for large codebases.

Provides efficient batch processing capabilities for analyzing large numbers
of UDL files with parallel processing, progress tracking, and result aggregation.
"""

import json
import logging
import multiprocessing as mp
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from udl_rating_framework.core.pipeline import RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.io.file_discovery import FileDiscovery
from udl_rating_framework.io.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    max_workers: int = mp.cpu_count()
    chunk_size: int = 10
    timeout_per_file: float = 30.0
    memory_limit_mb: int = 1024
    enable_caching: bool = True
    cache_dir: Optional[Path] = None
    progress_callback: Optional[Callable[[int, int], None]] = None
    error_handling: str = "continue"  # 'continue', 'stop', 'retry'
    max_retries: int = 3
    output_format: str = "json"  # 'json', 'csv', 'html'
    include_detailed_metrics: bool = True
    generate_summary: bool = True


@dataclass
class BatchResult:
    """Result of batch processing."""

    total_files: int
    processed_files: int
    failed_files: int
    processing_time: float
    average_quality: float
    quality_distribution: Dict[str, int]
    file_results: Dict[str, Dict[str, Any]]
    errors: Dict[str, str]
    summary_stats: Dict[str, Any]


@dataclass
class ProcessingTask:
    """Individual processing task."""

    file_path: Path
    content: Optional[str] = None
    priority: int = 0
    retry_count: int = 0


class BatchProcessor:
    """
    Batch processor for large-scale UDL quality analysis.

    Features:
    - Parallel processing with configurable worker count
    - Progress tracking and reporting
    - Error handling and retry logic
    - Memory management and resource limits
    - Caching for improved performance
    - Multiple output formats
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()

        # Initialize components
        default_metrics = [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]
        self.pipeline = RatingPipeline(metric_names=default_metrics)
        self.file_discovery = FileDiscovery()
        self.report_generator = ReportGenerator()

        # Setup caching
        if self.config.enable_caching:
            if self.config.cache_dir is None:
                self.config.cache_dir = Path(
                    tempfile.gettempdir()) / "udl_rating_cache"
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Processing state
        self.processed_count = 0
        self.total_count = 0
        self.start_time = 0.0
        self.results: Dict[str, Dict[str, Any]] = {}
        self.errors: Dict[str, str] = {}

    def process_directory(
        self,
        directory: Path,
        patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> BatchResult:
        """
        Process all UDL files in a directory.

        Args:
            directory: Directory to process
            patterns: File patterns to include (e.g., ['*.udl', '*.dsl'])
            exclude_patterns: File patterns to exclude

        Returns:
            Batch processing results
        """
        logger.info(f"Starting batch processing of directory: {directory}")

        # Discover files
        if patterns is None:
            patterns = ["*.udl", "*.dsl", "*.grammar", "*.ebnf"]

        files = self.file_discovery.discover_files(
            directory, extensions=[p.replace("*", "") for p in patterns]
        )

        # Apply exclude patterns
        if exclude_patterns:
            filtered_files = []
            for file_path in files:
                exclude = False
                for pattern in exclude_patterns:
                    if file_path.match(pattern):
                        exclude = True
                        break
                if not exclude:
                    filtered_files.append(file_path)
            files = filtered_files

        return self.process_files(files)

    def process_files(self, file_paths: List[Path]) -> BatchResult:
        """
        Process a list of UDL files.

        Args:
            file_paths: List of file paths to process

        Returns:
            Batch processing results
        """
        self.start_time = time.time()
        self.total_count = len(file_paths)
        self.processed_count = 0
        self.results.clear()
        self.errors.clear()

        logger.info(
            f"Processing {self.total_count} files with {self.config.max_workers} workers"
        )

        # Create processing tasks
        tasks = [ProcessingTask(file_path=path) for path in file_paths]

        # Process tasks in parallel
        if self.config.max_workers == 1:
            # Sequential processing
            for task in tasks:
                self._process_single_task(task)
        else:
            # Parallel processing
            self._process_tasks_parallel(tasks)

        # Generate final results
        processing_time = time.time() - self.start_time
        return self._create_batch_result(processing_time)

    def process_files_streaming(
        self, file_paths: List[Path], output_file: Path
    ) -> Iterator[Dict[str, Any]]:
        """
        Process files with streaming output for very large datasets.

        Args:
            file_paths: List of file paths to process
            output_file: File to stream results to

        Yields:
            Individual file processing results
        """
        self.start_time = time.time()
        self.total_count = len(file_paths)
        self.processed_count = 0

        logger.info(
            f"Starting streaming processing of {self.total_count} files")

        with open(output_file, "w") as f:
            f.write("[\n")  # Start JSON array
            first_result = True

            # Process in chunks to manage memory
            for i in range(0, len(file_paths), self.config.chunk_size):
                chunk = file_paths[i: i + self.config.chunk_size]
                chunk_tasks = [ProcessingTask(file_path=path)
                               for path in chunk]

                # Process chunk
                chunk_results = self._process_chunk(chunk_tasks)

                # Stream results
                for result in chunk_results:
                    if not first_result:
                        f.write(",\n")
                    json.dump(result, f, indent=2)
                    first_result = False
                    yield result

            f.write("\n]")  # End JSON array

    def _process_tasks_parallel(self, tasks: List[ProcessingTask]) -> None:
        """Process tasks in parallel using ProcessPoolExecutor."""
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit tasks in chunks
            future_to_task = {}

            for i in range(0, len(tasks), self.config.chunk_size):
                chunk = tasks[i: i + self.config.chunk_size]
                future = executor.submit(self._process_chunk_worker, chunk)
                future_to_task[future] = chunk

            # Collect results
            for future in as_completed(future_to_task):
                chunk = future_to_task[future]
                try:
                    chunk_results = future.result(
                        timeout=self.config.timeout_per_file * len(chunk)
                    )

                    # Merge results
                    for file_path, result in chunk_results.items():
                        if "error" in result:
                            self.errors[str(file_path)] = result["error"]
                        else:
                            self.results[str(file_path)] = result

                        self.processed_count += 1
                        self._update_progress()

                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    for task in chunk:
                        self.errors[str(task.file_path)] = str(e)
                        self.processed_count += 1
                        self._update_progress()

    def _process_chunk(self, tasks: List[ProcessingTask]) -> List[Dict[str, Any]]:
        """Process a chunk of tasks."""
        results = []

        for task in tasks:
            result = self._process_single_task(task)
            results.append(
                {"file_path": str(task.file_path), "result": result})

        return results

    def _process_chunk_worker(
        self, tasks: List[ProcessingTask]
    ) -> Dict[str, Dict[str, Any]]:
        """Worker function for processing a chunk of tasks."""
        # Initialize pipeline in worker process
        default_metrics = [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]
        pipeline = RatingPipeline(metric_names=default_metrics)
        results = {}

        for task in tasks:
            try:
                result = self._process_task_with_pipeline(task, pipeline)
                results[str(task.file_path)] = result
            except Exception as e:
                logger.error(f"Error processing {task.file_path}: {e}")
                results[str(task.file_path)] = {"error": str(e)}

        return results

    def _process_single_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process a single task."""
        try:
            result = self._process_task_with_pipeline(task, self.pipeline)
            self.results[str(task.file_path)] = result
            self.processed_count += 1
            self._update_progress()
            return result

        except Exception as e:
            logger.error(f"Error processing {task.file_path}: {e}")

            # Handle retries
            if (
                self.config.error_handling == "retry"
                and task.retry_count < self.config.max_retries
            ):
                task.retry_count += 1
                logger.info(
                    f"Retrying {task.file_path} (attempt {task.retry_count})")
                return self._process_single_task(task)

            error_msg = str(e)
            self.errors[str(task.file_path)] = error_msg
            self.processed_count += 1
            self._update_progress()

            if self.config.error_handling == "stop":
                raise

            return {"error": error_msg}

    def _process_task_with_pipeline(
        self, task: ProcessingTask, pipeline: RatingPipeline
    ) -> Dict[str, Any]:
        """Process a task using the provided pipeline."""
        # Check cache first
        if self.config.enable_caching:
            cached_result = self._get_cached_result(task.file_path)
            if cached_result:
                return cached_result

        # Read file content
        if task.content is None:
            task.content = task.file_path.read_text(encoding="utf-8")

        # Create UDL representation
        udl_repr = UDLRepresentation(task.content, str(task.file_path))

        # Rate UDL
        report = pipeline.rate_udl(udl_repr)

        # Create result
        result = {
            "file_path": str(task.file_path),
            "overall_score": report.overall_score,
            "confidence": report.confidence,
            "timestamp": time.time(),
        }

        if self.config.include_detailed_metrics:
            result["metric_scores"] = report.metric_scores
            result["computation_trace"] = [
                asdict(step) for step in report.computation_trace
            ]

        # Cache result
        if self.config.enable_caching:
            self._cache_result(task.file_path, result)

        return result

    def _get_cached_result(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get cached result for file."""
        if not self.config.cache_dir:
            return None

        cache_file = self.config.cache_dir / f"{file_path.name}.json"

        if cache_file.exists():
            try:
                # Check if file has been modified since cache
                file_mtime = file_path.stat().st_mtime
                cache_mtime = cache_file.stat().st_mtime

                if cache_mtime > file_mtime:
                    with open(cache_file) as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache for {file_path}: {e}")

        return None

    def _cache_result(self, file_path: Path, result: Dict[str, Any]) -> None:
        """Cache result for file."""
        if not self.config.cache_dir:
            return

        cache_file = self.config.cache_dir / f"{file_path.name}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"Error caching result for {file_path}: {e}")

    def _update_progress(self) -> None:
        """Update progress and call callback if provided."""
        if self.config.progress_callback:
            self.config.progress_callback(
                self.processed_count, self.total_count)

        # Log progress periodically
        if self.processed_count % 100 == 0 or self.processed_count == self.total_count:
            elapsed = time.time() - self.start_time
            rate = self.processed_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {self.processed_count}/{self.total_count} "
                f"({self.processed_count / self.total_count * 100:.1f}%) "
                f"Rate: {rate:.1f} files/sec"
            )

    def _create_batch_result(self, processing_time: float) -> BatchResult:
        """Create final batch result."""
        # Calculate statistics
        successful_results = [
            r for r in self.results.values() if "error" not in r]

        if successful_results:
            scores = [r["overall_score"] for r in successful_results]
            average_quality = sum(scores) / len(scores)

            # Quality distribution
            quality_distribution = defaultdict(int)
            for score in scores:
                if score >= 0.8:
                    quality_distribution["excellent"] += 1
                elif score >= 0.6:
                    quality_distribution["good"] += 1
                elif score >= 0.4:
                    quality_distribution["fair"] += 1
                else:
                    quality_distribution["poor"] += 1
        else:
            average_quality = 0.0
            quality_distribution = {}

        # Summary statistics
        summary_stats = {
            "processing_rate": (
                self.processed_count / processing_time if processing_time > 0 else 0
            ),
            "success_rate": (
                len(successful_results) / self.total_count
                if self.total_count > 0
                else 0
            ),
            "error_rate": (
                len(self.errors) / self.total_count if self.total_count > 0 else 0
            ),
            "average_processing_time": (
                processing_time / self.total_count if self.total_count > 0 else 0
            ),
        }

        if successful_results:
            summary_stats.update(
                {
                    "min_quality": min(r["overall_score"] for r in successful_results),
                    "max_quality": max(r["overall_score"] for r in successful_results),
                    "median_quality": sorted(
                        [r["overall_score"] for r in successful_results]
                    )[len(successful_results) // 2],
                }
            )

        return BatchResult(
            total_files=self.total_count,
            processed_files=self.processed_count,
            failed_files=len(self.errors),
            processing_time=processing_time,
            average_quality=average_quality,
            quality_distribution=dict(quality_distribution),
            file_results=self.results,
            errors=self.errors,
            summary_stats=summary_stats,
        )

    def save_results(
        self, result: BatchResult, output_file: Path, format: Optional[str] = None
    ) -> None:
        """
        Save batch results to file.

        Args:
            result: Batch processing results
            output_file: Output file path
            format: Output format ('json', 'csv', 'html')
        """
        if format is None:
            format = self.config.output_format

        if format == "json":
            with open(output_file, "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)

        elif format == "csv":
            import csv

            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(
                    ["file_path", "overall_score", "confidence", "status"])

                # Write results
                for file_path, file_result in result.file_results.items():
                    if "error" in file_result:
                        writer.writerow(
                            [file_path, "", "",
                                f"Error: {file_result['error']}"]
                        )
                    else:
                        writer.writerow(
                            [
                                file_path,
                                file_result["overall_score"],
                                file_result["confidence"],
                                "Success",
                            ]
                        )

        elif format == "html":
            html_content = self._generate_html_report(result)
            with open(output_file, "w") as f:
                f.write(html_content)

        else:
            raise ValueError(f"Unsupported output format: {format}")

        logger.info(f"Results saved to {output_file}")

    def _generate_html_report(self, result: BatchResult) -> str:
        """Generate HTML report from batch results."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>UDL Batch Processing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
                .stat {{ background: white; padding: 10px; border-radius: 5px; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .error {{ color: red; }}
                .success {{ color: green; }}
            </style>
        </head>
        <body>
            <h1>UDL Batch Processing Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="stats">
                    <div class="stat">
                        <strong>Total Files:</strong> {result.total_files}
                    </div>
                    <div class="stat">
                        <strong>Processed:</strong> {result.processed_files}
                    </div>
                    <div class="stat">
                        <strong>Failed:</strong> {result.failed_files}
                    </div>
                    <div class="stat">
                        <strong>Average Quality:</strong> {result.average_quality:.3f}
                    </div>
                    <div class="stat">
                        <strong>Processing Time:</strong> {result.processing_time:.1f}s
                    </div>
                </div>
            </div>
            
            <h2>Quality Distribution</h2>
            <table>
                <tr><th>Quality Level</th><th>Count</th><th>Percentage</th></tr>
        """

        total_successful = sum(result.quality_distribution.values())
        for level, count in result.quality_distribution.items():
            percentage = (count / total_successful *
                          100) if total_successful > 0 else 0
            html += f"<tr><td>{level.title()}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"

        html += """
            </table>
            
            <h2>File Results</h2>
            <table>
                <tr><th>File Path</th><th>Overall Score</th><th>Confidence</th><th>Status</th></tr>
        """

        for file_path, file_result in result.file_results.items():
            if "error" in file_result:
                html += f'<tr><td>{file_path}</td><td>-</td><td>-</td><td class="error">Error: {file_result["error"]}</td></tr>'
            else:
                html += f"""<tr>
                    <td>{file_path}</td>
                    <td>{file_result["overall_score"]:.3f}</td>
                    <td>{file_result["confidence"]:.3f}</td>
                    <td class="success">Success</td>
                </tr>"""

        html += """
            </table>
        </body>
        </html>
        """

        return html

    def clear_cache(self) -> None:
        """Clear processing cache."""
        if self.config.cache_dir and self.config.cache_dir.exists():
            shutil.rmtree(self.config.cache_dir)
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")


def main():
    """CLI entry point for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(description="UDL Batch Processor")
    parser.add_argument("input_path", type=Path,
                        help="Input directory or file list")
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "html"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(), help="Number of worker processes"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10, help="Chunk size for parallel processing"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Timeout per file in seconds"
    )
    parser.add_argument("--cache-dir", type=Path, help="Cache directory path")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching")
    parser.add_argument(
        "--include-patterns",
        nargs="+",
        default=["*.udl", "*.dsl", "*.grammar", "*.ebnf"],
        help="File patterns to include",
    )
    parser.add_argument(
        "--exclude-patterns", nargs="+", help="File patterns to exclude"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming processing for large datasets",
    )

    args = parser.parse_args()

    # Create configuration
    config = BatchConfig(
        max_workers=args.workers,
        chunk_size=args.chunk_size,
        timeout_per_file=args.timeout,
        enable_caching=not args.no_cache,
        cache_dir=args.cache_dir,
        output_format=args.format,
    )

    # Progress callback
    def progress_callback(processed: int, total: int):
        print(
            f"Progress: {processed}/{total} ({processed / total * 100:.1f}%)")

    config.progress_callback = progress_callback

    # Create processor
    processor = BatchProcessor(config)

    try:
        if args.input_path.is_dir():
            # Process directory
            if args.streaming:
                print("Streaming processing not supported for directory input")
                return

            result = processor.process_directory(
                args.input_path,
                patterns=args.include_patterns,
                exclude_patterns=args.exclude_patterns,
            )
        else:
            # Process file list
            with open(args.input_path) as f:
                file_paths = [Path(line.strip()) for line in f if line.strip()]

            if args.streaming:
                print("Starting streaming processing...")
                for file_result in processor.process_files_streaming(
                    file_paths, args.output
                ):
                    print(f"Processed: {file_result['file_path']}")
                return
            else:
                result = processor.process_files(file_paths)

        # Save results
        processor.save_results(result, args.output, args.format)

        # Print summary
        print("\nBatch processing completed:")
        print(f"  Total files: {result.total_files}")
        print(f"  Processed: {result.processed_files}")
        print(f"  Failed: {result.failed_files}")
        print(f"  Average quality: {result.average_quality:.3f}")
        print(f"  Processing time: {result.processing_time:.1f}s")
        print(f"  Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
