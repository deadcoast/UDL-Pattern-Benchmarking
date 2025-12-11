"""
Distributed computing support for UDL Rating Framework.

Provides distributed processing capabilities using Ray and Dask for large-scale
UDL analysis across multiple machines and clusters.
"""

import asyncio
import logging
import os
import pickle
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

try:
    import ray
    from ray import remote

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    remote = lambda *args, **kwargs: lambda f: f  # No-op decorator

try:
    import dask
    from dask import delayed
    from dask.distributed import Client, Future, as_completed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

    def delayed(f):
        return f  # No-op decorator


from udl_rating_framework.core.multiprocessing import (
    BatchProcessingStats,
    ProcessingResult,
)
from udl_rating_framework.core.pipeline import QualityReport, RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed processing."""

    backend: str = "ray"  # 'ray', 'dask', or 'auto'
    cluster_address: Optional[str] = None
    num_workers: Optional[int] = None
    resources_per_worker: Dict[str, float] = None
    batch_size: int = 10
    max_retries: int = 3
    timeout_seconds: float = 300.0
    enable_progress_tracking: bool = True
    checkpoint_interval: int = 100
    result_serialization: str = "pickle"  # 'pickle', 'json'

    def __post_init__(self):
        if self.resources_per_worker is None:
            self.resources_per_worker = {
                "cpu": 1.0,
                "memory": 1024 * 1024 * 1024,
            }  # 1GB


@dataclass
class DistributedTask:
    """Distributed processing task."""

    task_id: str
    file_path: str
    content: Optional[str] = None
    metric_names: List[str] = None
    weights: Optional[Dict[str, float]] = None
    priority: int = 0
    retry_count: int = 0


@dataclass
class DistributedResult:
    """Result from distributed processing."""

    task_id: str
    success: bool
    result: Optional[QualityReport] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: Optional[str] = None
    node_id: Optional[str] = None


class DistributedProcessor:
    """
    Distributed processor for large-scale UDL analysis.

    Supports both Ray and Dask backends for distributed computing across
    multiple machines and clusters.
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        """
        Initialize distributed processor.

        Args:
            config: Distributed processing configuration
        """
        self.config = config or DistributedConfig()
        self.backend = None
        self.client = None
        self.is_initialized = False

        # Auto-detect backend if not specified
        if self.config.backend == "auto":
            if RAY_AVAILABLE:
                self.config.backend = "ray"
            elif DASK_AVAILABLE:
                self.config.backend = "dask"
            else:
                raise RuntimeError(
                    "No distributed computing backend available. Install Ray or Dask."
                )

        # Validate backend availability
        if self.config.backend == "ray" and not RAY_AVAILABLE:
            raise RuntimeError(
                "Ray backend requested but not available. Install Ray.")
        if self.config.backend == "dask" and not DASK_AVAILABLE:
            raise RuntimeError(
                "Dask backend requested but not available. Install Dask."
            )

    def initialize(self) -> None:
        """Initialize the distributed computing backend."""
        if self.is_initialized:
            return

        logger.info(f"Initializing {self.config.backend} distributed backend")

        if self.config.backend == "ray":
            self._initialize_ray()
        elif self.config.backend == "dask":
            self._initialize_dask()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        self.is_initialized = True
        logger.info(
            f"Distributed backend {self.config.backend} initialized successfully"
        )

    def shutdown(self) -> None:
        """Shutdown the distributed computing backend."""
        if not self.is_initialized:
            return

        logger.info(f"Shutting down {self.config.backend} distributed backend")

        if self.config.backend == "ray":
            ray.shutdown()
        elif self.config.backend == "dask" and self.client:
            self.client.close()

        self.is_initialized = False

    def process_files_distributed(
        self,
        file_contents: List[tuple],  # (file_path, content)
        metric_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple:
        """
        Process files using distributed computing.

        Args:
            file_contents: List of (file_path, content) tuples
            metric_names: List of metric names to compute
            weights: Optional metric weights
            progress_callback: Optional progress callback

        Returns:
            Tuple of (reports, statistics)
        """
        if not self.is_initialized:
            self.initialize()

        start_time = time.time()
        total_files = len(file_contents)

        logger.info(f"Starting distributed processing of {total_files} files")

        # Create tasks
        tasks = []
        for i, (file_path, content) in enumerate(file_contents):
            task = DistributedTask(
                task_id=f"task_{i}",
                file_path=file_path,
                content=content,
                metric_names=metric_names,
                weights=weights,
            )
            tasks.append(task)

        # Process tasks based on backend
        if self.config.backend == "ray":
            results = self._process_with_ray(tasks, progress_callback)
        elif self.config.backend == "dask":
            results = self._process_with_dask(tasks, progress_callback)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        # Convert results to expected format
        reports = []
        successful = 0
        failed = 0
        processing_times = []

        for result in results:
            processing_times.append(result.processing_time)

            if result.success and result.result:
                reports.append(result.result)
                successful += 1
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
                    udl_file=result.task_id,
                    errors=[result.error] if result.error else [],
                )
                reports.append(error_report)
                failed += 1

        total_time = time.time() - start_time

        # Create statistics
        stats = BatchProcessingStats(
            total_files=total_files,
            successful=successful,
            failed=failed,
            total_time=total_time,
            average_time_per_file=(
                sum(processing_times) / len(processing_times)
                if processing_times
                else 0.0
            ),
            max_time=max(processing_times) if processing_times else 0.0,
            min_time=min(processing_times) if processing_times else 0.0,
            worker_count=self.config.num_workers or 1,
        )

        logger.info(
            f"Distributed processing completed: {successful} successful, {failed} failed, "
            f"total time: {total_time:.3f}s"
        )

        return reports, stats

    def _initialize_ray(self) -> None:
        """Initialize Ray backend."""
        ray_config = {"ignore_reinit_error": True, "log_to_driver": False}

        if self.config.cluster_address:
            ray_config["address"] = self.config.cluster_address

        if self.config.num_workers:
            ray_config["num_cpus"] = self.config.num_workers

        ray.init(**ray_config)

        # Get cluster info
        cluster_resources = ray.cluster_resources()
        logger.info(f"Ray cluster resources: {cluster_resources}")

    def _initialize_dask(self) -> None:
        """Initialize Dask backend."""
        if self.config.cluster_address:
            self.client = Client(self.config.cluster_address)
        else:
            # Start local cluster
            from dask.distributed import LocalCluster

            cluster_config = {}
            if self.config.num_workers:
                cluster_config["n_workers"] = self.config.num_workers

            cluster = LocalCluster(**cluster_config)
            self.client = Client(cluster)

        logger.info(f"Dask client: {self.client}")

    def _process_with_ray(
        self,
        tasks: List[DistributedTask],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DistributedResult]:
        """Process tasks using Ray."""

        # Submit tasks to Ray
        futures = []
        for task in tasks:
            future = process_udl_task_ray.remote(task)
            futures.append((future, task))

        # Collect results
        results = []
        completed = 0

        while futures:
            # Wait for at least one task to complete
            ready_futures, remaining_futures = ray.wait(
                [f for f, _ in futures],
                num_returns=1,
                timeout=self.config.timeout_seconds,
            )

            # Process completed tasks
            for ready_future in ready_futures:
                # Find corresponding task
                task = None
                for f, t in futures:
                    if f == ready_future:
                        task = t
                        break

                try:
                    result = ray.get(ready_future)
                    results.append(result)
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, len(tasks))

                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    error_result = DistributedResult(
                        task_id=task.task_id, success=False, error=str(e)
                    )
                    results.append(error_result)
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, len(tasks))

            # Update futures list
            futures = [(f, t) for f, t in futures if f not in ready_futures]

        return results

    def _process_with_dask(
        self,
        tasks: List[DistributedTask],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DistributedResult]:
        """Process tasks using Dask."""

        # Submit tasks to Dask
        futures = []
        for task in tasks:
            future = self.client.submit(process_udl_task_dask, task)
            futures.append(future)

        # Collect results
        results = []
        completed = 0

        for future in as_completed(futures, timeout=self.config.timeout_seconds):
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(tasks))

            except Exception as e:
                logger.error(f"Task failed: {e}")
                error_result = DistributedResult(
                    task_id=f"unknown_{completed}", success=False, error=str(e)
                )
                results.append(error_result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(tasks))

        return results

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the distributed cluster."""
        if not self.is_initialized:
            return {}

        if self.config.backend == "ray":
            return {
                "backend": "ray",
                "cluster_resources": ray.cluster_resources(),
                "available_resources": ray.available_resources(),
                "nodes": [node for node in ray.nodes()],
                "is_connected": ray.is_initialized(),
            }

        elif self.config.backend == "dask":
            if self.client:
                return {
                    "backend": "dask",
                    "scheduler_info": self.client.scheduler_info(),
                    "workers": list(self.client.scheduler_info()["workers"].keys()),
                    "total_cores": sum(
                        w["ncores"]
                        for w in self.client.scheduler_info()["workers"].values()
                    ),
                    "total_memory": sum(
                        w["memory_limit"]
                        for w in self.client.scheduler_info()["workers"].values()
                    ),
                }

        return {}


# Ray remote functions
if RAY_AVAILABLE:

    @remote
    def process_udl_task_ray(task: DistributedTask) -> DistributedResult:
        """Ray remote function to process a UDL task."""
        return _process_udl_task_impl(task)


# Dask functions
def process_udl_task_dask(task: DistributedTask) -> DistributedResult:
    """Dask function to process a UDL task."""
    return _process_udl_task_impl(task)


def _process_udl_task_impl(task: DistributedTask) -> DistributedResult:
    """Implementation of UDL task processing."""
    start_time = time.time()

    try:
        # Create UDL representation
        udl = UDLRepresentation(task.content, task.file_path)

        # Create rating pipeline
        pipeline = RatingPipeline(
            metric_names=task.metric_names, weights=task.weights)

        # Compute rating
        report = pipeline.compute_rating(udl)

        processing_time = time.time() - start_time

        return DistributedResult(
            task_id=task.task_id,
            success=True,
            result=report,
            processing_time=processing_time,
            worker_id=os.getpid(),
            node_id=os.uname().nodename if hasattr(os, "uname") else "unknown",
        )

    except Exception as e:
        processing_time = time.time() - start_time

        return DistributedResult(
            task_id=task.task_id,
            success=False,
            error=str(e),
            processing_time=processing_time,
            worker_id=os.getpid(),
            node_id=os.uname().nodename if hasattr(os, "uname") else "unknown",
        )


class DistributedBatchProcessor:
    """
    High-level distributed batch processor.

    Provides easy-to-use interface for distributed UDL processing
    with automatic scaling and fault tolerance.
    """

    def __init__(
        self,
        backend: str = "auto",
        cluster_address: Optional[str] = None,
        num_workers: Optional[int] = None,
    ):
        """
        Initialize distributed batch processor.

        Args:
            backend: Distributed backend ('ray', 'dask', 'auto')
            cluster_address: Address of existing cluster
            num_workers: Number of workers to use
        """
        config = DistributedConfig(
            backend=backend, cluster_address=cluster_address, num_workers=num_workers
        )

        self.processor = DistributedProcessor(config)

    def process_directory(
        self,
        directory: Path,
        metric_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        file_patterns: List[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple:
        """
        Process all UDL files in a directory using distributed computing.

        Args:
            directory: Directory containing UDL files
            metric_names: List of metrics to compute
            weights: Optional metric weights
            file_patterns: File patterns to match
            progress_callback: Optional progress callback

        Returns:
            Tuple of (reports, statistics)
        """
        from udl_rating_framework.io.file_discovery import FileDiscovery

        # Discover files
        if file_patterns is None:
            file_patterns = ["*.udl", "*.dsl", "*.grammar", "*.ebnf"]

        discovery = FileDiscovery()
        extensions = [p.replace("*", "") for p in file_patterns]
        files = discovery.discover_files(directory, extensions)

        # Read file contents
        file_contents = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                file_contents.append((str(file_path), content))
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        # Process files
        return self.processor.process_files_distributed(
            file_contents=file_contents,
            metric_names=metric_names,
            weights=weights,
            progress_callback=progress_callback,
        )

    def process_files(
        self,
        file_paths: List[Path],
        metric_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple:
        """
        Process a list of UDL files using distributed computing.

        Args:
            file_paths: List of file paths to process
            metric_names: List of metrics to compute
            weights: Optional metric weights
            progress_callback: Optional progress callback

        Returns:
            Tuple of (reports, statistics)
        """
        # Read file contents
        file_contents = []
        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding="utf-8")
                file_contents.append((str(file_path), content))
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        # Process files
        return self.processor.process_files_distributed(
            file_contents=file_contents,
            metric_names=metric_names,
            weights=weights,
            progress_callback=progress_callback,
        )

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of the distributed cluster."""
        return self.processor.get_cluster_info()

    def shutdown(self) -> None:
        """Shutdown the distributed processor."""
        self.processor.shutdown()

    def __enter__(self):
        """Context manager entry."""
        self.processor.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.processor.shutdown()


# Convenience functions
def process_files_distributed(
    file_paths: List[Path],
    metric_names: List[str],
    backend: str = "auto",
    cluster_address: Optional[str] = None,
    num_workers: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    show_progress: bool = True,
) -> tuple:
    """
    Convenience function to process files with distributed computing.

    Args:
        file_paths: List of file paths to process
        metric_names: List of metrics to compute
        backend: Distributed backend to use
        cluster_address: Address of existing cluster
        num_workers: Number of workers to use
        weights: Optional metric weights
        show_progress: Whether to show progress

    Returns:
        Tuple of (reports, statistics)
    """
    progress_callback = None
    if show_progress:

        def progress_callback(completed: int, total: int):
            percentage = (completed / total) * 100 if total > 0 else 0
            logger.info(
                f"Distributed processing: {completed}/{total} ({percentage:.1f}%)"
            )

    with DistributedBatchProcessor(
        backend=backend, cluster_address=cluster_address, num_workers=num_workers
    ) as processor:
        return processor.process_files(
            file_paths=file_paths,
            metric_names=metric_names,
            weights=weights,
            progress_callback=progress_callback,
        )


def process_directory_distributed(
    directory: Path,
    metric_names: List[str],
    backend: str = "auto",
    cluster_address: Optional[str] = None,
    num_workers: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    file_patterns: List[str] = None,
    show_progress: bool = True,
) -> tuple:
    """
    Convenience function to process directory with distributed computing.

    Args:
        directory: Directory containing UDL files
        metric_names: List of metrics to compute
        backend: Distributed backend to use
        cluster_address: Address of existing cluster
        num_workers: Number of workers to use
        weights: Optional metric weights
        file_patterns: File patterns to match
        show_progress: Whether to show progress

    Returns:
        Tuple of (reports, statistics)
    """
    progress_callback = None
    if show_progress:

        def progress_callback(completed: int, total: int):
            percentage = (completed / total) * 100 if total > 0 else 0
            logger.info(
                f"Distributed processing: {completed}/{total} ({percentage:.1f}%)"
            )

    with DistributedBatchProcessor(
        backend=backend, cluster_address=cluster_address, num_workers=num_workers
    ) as processor:
        return processor.process_directory(
            directory=directory,
            metric_names=metric_names,
            weights=weights,
            file_patterns=file_patterns,
            progress_callback=progress_callback,
        )
