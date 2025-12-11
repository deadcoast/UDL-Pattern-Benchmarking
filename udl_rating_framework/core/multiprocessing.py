"""
Multiprocessing module for UDL Rating Framework.

Provides parallel processing capabilities for batch file processing and metric computation
to improve performance on multi-core systems.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from pathlib import Path
import traceback
from dataclasses import dataclass
import time

from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.core.pipeline import QualityReport, RatingPipeline

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of parallel processing operation."""
    
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: Optional[int] = None


@dataclass
class BatchProcessingStats:
    """Statistics for batch processing operation."""
    
    total_files: int
    successful: int
    failed: int
    total_time: float
    average_time_per_file: float
    max_time: float
    min_time: float
    worker_count: int


def _process_single_udl(args: Tuple[str, str, List[str], Optional[Dict[str, float]], bool]) -> ProcessingResult:
    """
    Process a single UDL file (worker function).
    
    Args:
        args: Tuple of (file_path, content, metric_names, weights, enable_tracing)
        
    Returns:
        ProcessingResult with rating report or error
    """
    file_path, content, metric_names, weights, enable_tracing = args
    worker_id = mp.current_process().pid
    start_time = time.time()
    
    try:
        # Create UDL representation
        udl = UDLRepresentation(content, file_path)
        
        # Create rating pipeline
        pipeline = RatingPipeline(
            metric_names=metric_names,
            weights=weights,
            enable_tracing=enable_tracing
        )
        
        # Compute rating
        report = pipeline.compute_rating(udl)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            success=True,
            result=report,
            processing_time=processing_time,
            worker_id=worker_id
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error processing {file_path}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        return ProcessingResult(
            success=False,
            error=error_msg,
            processing_time=processing_time,
            worker_id=worker_id
        )


def _compute_single_metric(args: Tuple[str, str, str]) -> ProcessingResult:
    """
    Compute a single metric for a UDL (worker function).
    
    Args:
        args: Tuple of (udl_content, file_path, metric_name)
        
    Returns:
        ProcessingResult with metric value or error
    """
    udl_content, file_path, metric_name = args
    worker_id = mp.current_process().pid
    start_time = time.time()
    
    try:
        # Import here to avoid issues with multiprocessing
        from udl_rating_framework.core.metrics.base import MetricRegistry
        
        # Create UDL representation
        udl = UDLRepresentation(udl_content, file_path)
        
        # Get metric class and compute
        metric_class = MetricRegistry.get_metric(metric_name)
        metric = metric_class()
        value = metric.compute(udl)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            success=True,
            result=value,
            processing_time=processing_time,
            worker_id=worker_id
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error computing {metric_name} for {file_path}: {str(e)}"
        logger.error(error_msg)
        
        return ProcessingResult(
            success=False,
            error=error_msg,
            processing_time=processing_time,
            worker_id=worker_id
        )


class ParallelProcessor:
    """
    Parallel processor for UDL rating operations.
    
    Provides multiprocessing capabilities for batch file processing and metric computation.
    """
    
    def __init__(self, max_workers: Optional[int] = None, chunk_size: int = 1):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes (default: CPU count)
            chunk_size: Number of tasks per chunk for load balancing
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        
        logger.info(f"Initialized parallel processor with {self.max_workers} workers")
    
    def process_files_parallel(
        self,
        file_contents: List[Tuple[str, str]],  # (file_path, content)
        metric_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        enable_tracing: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[QualityReport], BatchProcessingStats]:
        """
        Process multiple UDL files in parallel.
        
        Args:
            file_contents: List of (file_path, content) tuples
            metric_names: List of metric names to compute
            weights: Optional metric weights
            enable_tracing: Whether to enable computation tracing
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (reports, statistics)
        """
        start_time = time.time()
        total_files = len(file_contents)
        
        logger.info(f"Starting parallel processing of {total_files} files with {self.max_workers} workers")
        
        # Prepare arguments for worker processes
        args_list = [
            (file_path, content, metric_names, weights, enable_tracing)
            for file_path, content in file_contents
        ]
        
        reports = []
        processing_times = []
        successful = 0
        failed = 0
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(_process_single_udl, args): i
                for i, args in enumerate(args_list)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                file_path = file_contents[index][0]
                
                try:
                    result = future.result()
                    processing_times.append(result.processing_time)
                    
                    if result.success:
                        reports.append(result.result)
                        successful += 1
                        logger.debug(f"Successfully processed {file_path} in {result.processing_time:.3f}s")
                    else:
                        # Create error report
                        error_report = QualityReport(
                            overall_score=0.0,
                            confidence=0.0,
                            metric_scores={},
                            metric_formulas={},
                            computation_trace=[],
                            error_bounds={},
                            timestamp=time.time(),
                            udl_file=file_path,
                            errors=[result.error] if result.error else []
                        )
                        reports.append(error_report)
                        failed += 1
                        logger.warning(f"Failed to process {file_path}: {result.error}")
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(successful + failed, total_files)
                        
                except Exception as e:
                    # Handle unexpected errors
                    error_msg = f"Unexpected error processing {file_path}: {str(e)}"
                    logger.error(error_msg)
                    
                    error_report = QualityReport(
                        overall_score=0.0,
                        confidence=0.0,
                        metric_scores={},
                        metric_formulas={},
                        computation_trace=[],
                        error_bounds={},
                        timestamp=time.time(),
                        udl_file=file_path,
                        errors=[error_msg]
                    )
                    reports.append(error_report)
                    failed += 1
                    processing_times.append(0.0)
                    
                    if progress_callback:
                        progress_callback(successful + failed, total_files)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = BatchProcessingStats(
            total_files=total_files,
            successful=successful,
            failed=failed,
            total_time=total_time,
            average_time_per_file=sum(processing_times) / len(processing_times) if processing_times else 0.0,
            max_time=max(processing_times) if processing_times else 0.0,
            min_time=min(processing_times) if processing_times else 0.0,
            worker_count=self.max_workers
        )
        
        logger.info(
            f"Parallel processing completed: {successful} successful, {failed} failed, "
            f"total time: {total_time:.3f}s, avg per file: {stats.average_time_per_file:.3f}s"
        )
        
        return reports, stats
    
    def compute_metrics_parallel(
        self,
        udl_contents: List[Tuple[str, str]],  # (file_path, content)
        metric_names: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, List[Tuple[str, Union[float, str]]]]:
        """
        Compute multiple metrics for multiple UDLs in parallel.
        
        Args:
            udl_contents: List of (file_path, content) tuples
            metric_names: List of metric names to compute
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping metric names to lists of (file_path, value_or_error) tuples
        """
        start_time = time.time()
        total_tasks = len(udl_contents) * len(metric_names)
        
        logger.info(f"Starting parallel metric computation: {len(udl_contents)} UDLs × {len(metric_names)} metrics = {total_tasks} tasks")
        
        # Prepare arguments for worker processes
        args_list = []
        for file_path, content in udl_contents:
            for metric_name in metric_names:
                args_list.append((content, file_path, metric_name))
        
        results = {metric_name: [] for metric_name in metric_names}
        completed_tasks = 0
        
        # Process metrics in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(_compute_single_metric, args): args
                for args in args_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_args):
                args = future_to_args[future]
                udl_content, file_path, metric_name = args
                
                try:
                    result = future.result()
                    
                    if result.success:
                        results[metric_name].append((file_path, result.result))
                        logger.debug(f"Computed {metric_name} for {file_path}: {result.result}")
                    else:
                        results[metric_name].append((file_path, result.error))
                        logger.warning(f"Failed to compute {metric_name} for {file_path}: {result.error}")
                    
                    completed_tasks += 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(completed_tasks, total_tasks)
                        
                except Exception as e:
                    error_msg = f"Unexpected error computing {metric_name} for {file_path}: {str(e)}"
                    logger.error(error_msg)
                    results[metric_name].append((file_path, error_msg))
                    completed_tasks += 1
                    
                    if progress_callback:
                        progress_callback(completed_tasks, total_tasks)
        
        total_time = time.time() - start_time
        logger.info(f"Parallel metric computation completed in {total_time:.3f}s")
        
        return results
    
    def get_optimal_worker_count(self, task_count: int, task_complexity: str = "medium") -> int:
        """
        Get optimal worker count based on task characteristics.
        
        Args:
            task_count: Number of tasks to process
            task_complexity: Task complexity ("low", "medium", "high")
            
        Returns:
            Optimal number of workers
        """
        cpu_count = mp.cpu_count()
        
        # Adjust based on task complexity
        complexity_factors = {
            "low": 1.5,      # I/O bound tasks can use more workers
            "medium": 1.0,   # Balanced tasks use CPU count
            "high": 0.75     # CPU intensive tasks use fewer workers
        }
        
        factor = complexity_factors.get(task_complexity, 1.0)
        optimal_workers = int(cpu_count * factor)
        
        # Don't exceed task count (no point having more workers than tasks)
        optimal_workers = min(optimal_workers, task_count)
        
        # Ensure at least 1 worker
        optimal_workers = max(1, optimal_workers)
        
        return optimal_workers


def create_progress_callback(description: str = "Processing") -> Callable[[int, int], None]:
    """
    Create a simple progress callback function.
    
    Args:
        description: Description to show in progress messages
        
    Returns:
        Progress callback function
    """
    def progress_callback(completed: int, total: int):
        percentage = (completed / total) * 100 if total > 0 else 0
        logger.info(f"{description}: {completed}/{total} ({percentage:.1f}%)")
    
    return progress_callback


# Convenience functions for common operations

def process_udl_files_parallel(
    file_paths: List[Union[str, Path]],
    metric_names: List[str],
    max_workers: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    enable_tracing: bool = False,
    show_progress: bool = True
) -> Tuple[List[QualityReport], BatchProcessingStats]:
    """
    Convenience function to process UDL files in parallel.
    
    Args:
        file_paths: List of file paths to process
        metric_names: List of metric names to compute
        max_workers: Maximum number of worker processes
        weights: Optional metric weights
        enable_tracing: Whether to enable computation tracing
        show_progress: Whether to show progress messages
        
    Returns:
        Tuple of (reports, statistics)
    """
    # Read file contents
    file_contents = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            file_contents.append((str(file_path), content))
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            # Add empty content to maintain order
            file_contents.append((str(file_path), ""))
    
    # Create processor and process files
    processor = ParallelProcessor(max_workers=max_workers)
    
    progress_callback = None
    if show_progress:
        progress_callback = create_progress_callback("Processing UDL files")
    
    return processor.process_files_parallel(
        file_contents=file_contents,
        metric_names=metric_names,
        weights=weights,
        enable_tracing=enable_tracing,
        progress_callback=progress_callback
    )


def compute_metrics_for_udls_parallel(
    udl_representations: List[UDLRepresentation],
    metric_names: List[str],
    max_workers: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, List[Tuple[str, Union[float, str]]]]:
    """
    Convenience function to compute metrics for UDL representations in parallel.
    
    Args:
        udl_representations: List of UDL representations
        metric_names: List of metric names to compute
        max_workers: Maximum number of worker processes
        show_progress: Whether to show progress messages
        
    Returns:
        Dictionary mapping metric names to lists of (file_path, value_or_error) tuples
    """
    # Extract content from UDL representations
    udl_contents = [
        (udl.file_path, udl.source_text)
        for udl in udl_representations
    ]
    
    # Create processor and compute metrics
    processor = ParallelProcessor(max_workers=max_workers)
    
    progress_callback = None
    if show_progress:
        progress_callback = create_progress_callback("Computing metrics")
    
    return processor.compute_metrics_parallel(
        udl_contents=udl_contents,
        metric_names=metric_names,
        progress_callback=progress_callback
    )