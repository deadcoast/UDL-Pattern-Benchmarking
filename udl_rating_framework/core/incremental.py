"""
Incremental computation for updated UDLs in UDL Rating Framework.

Provides efficient incremental processing that only recomputes metrics
for changed parts of UDL files, with dependency tracking and smart
invalidation strategies.
"""

import difflib
import hashlib
import json
import logging
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from udl_rating_framework.core.caching import get_metric_cache, get_udl_cache
from udl_rating_framework.core.pipeline import QualityReport, RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation

logger = logging.getLogger(__name__)


@dataclass
class FileSnapshot:
    """Snapshot of a file for change detection."""

    file_path: str
    content_hash: str
    modification_time: float
    file_size: int
    line_count: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, file_path: Path) -> "FileSnapshot":
        """Create snapshot from file."""
        try:
            stat = file_path.stat()
            content = file_path.read_text(encoding="utf-8")

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            checksum = hashlib.md5(content.encode("utf-8")).hexdigest()
            line_count = content.count("\n") + 1

            return cls(
                file_path=str(file_path),
                content_hash=content_hash,
                modification_time=stat.st_mtime,
                file_size=stat.st_size,
                line_count=line_count,
                checksum=checksum,
            )
        except Exception as e:
            logger.error(f"Failed to create snapshot for {file_path}: {e}")
            return cls(
                file_path=str(file_path),
                content_hash="",
                modification_time=0.0,
                file_size=0,
                line_count=0,
                checksum="",
            )


@dataclass
class ChangeSet:
    """Set of changes detected in a file."""

    file_path: str
    change_type: str  # 'modified', 'added', 'deleted', 'renamed'
    old_snapshot: Optional[FileSnapshot] = None
    new_snapshot: Optional[FileSnapshot] = None
    affected_lines: Set[int] = field(default_factory=set)
    affected_constructs: Set[str] = field(default_factory=set)
    change_summary: Dict[str, Any] = field(default_factory=dict)

    def get_change_ratio(self) -> float:
        """Get ratio of changed content (0.0 to 1.0)."""
        if not self.old_snapshot or not self.new_snapshot:
            return 1.0  # Complete change

        if self.old_snapshot.line_count == 0:
            return 1.0

        return len(self.affected_lines) / self.old_snapshot.line_count


@dataclass
class IncrementalResult:
    """Result of incremental computation."""

    file_path: str
    was_cached: bool
    was_incremental: bool
    change_ratio: float
    processing_time: float
    cache_hit_ratio: float
    recomputed_metrics: Set[str]
    report: QualityReport
    change_set: Optional[ChangeSet] = None


class DependencyTracker:
    """
    Tracks dependencies between UDL constructs and metrics.

    Enables fine-grained invalidation when only specific parts
    of a UDL file change.
    """

    def __init__(self):
        """Initialize dependency tracker."""
        self.construct_dependencies = defaultdict(
            set)  # construct -> set of metrics
        self.metric_dependencies = defaultdict(
            set)  # metric -> set of constructs
        self.construct_locations = {}  # construct -> (file, line_range)
        self.invalidation_rules = {}  # metric -> invalidation_function

        self._setup_default_dependencies()

    def _setup_default_dependencies(self) -> None:
        """Setup default dependency rules for standard metrics."""
        # Consistency metric depends on rule definitions and references
        self.add_dependency(
            "consistency", ["rule_definition", "rule_reference", "constraint"]
        )

        # Completeness metric depends on construct definitions
        self.add_dependency(
            "completeness", ["construct_definition",
                             "terminal", "non_terminal"]
        )

        # Expressiveness metric depends on grammar structure
        self.add_dependency(
            "expressiveness", ["grammar_rule", "production", "alternation"]
        )

        # Structural coherence depends on overall structure
        self.add_dependency(
            "structural_coherence", ["grammar_rule", "dependency", "hierarchy"]
        )

    def add_dependency(self, metric_name: str, construct_types: List[str]) -> None:
        """
        Add dependency between metric and construct types.

        Args:
            metric_name: Name of the metric
            construct_types: List of construct types the metric depends on
        """
        for construct_type in construct_types:
            self.construct_dependencies[construct_type].add(metric_name)
            self.metric_dependencies[metric_name].add(construct_type)

    def get_affected_metrics(self, changed_constructs: Set[str]) -> Set[str]:
        """
        Get metrics affected by changed constructs.

        Args:
            changed_constructs: Set of changed construct types

        Returns:
            Set of affected metric names
        """
        affected_metrics = set()

        for construct in changed_constructs:
            affected_metrics.update(
                self.construct_dependencies.get(construct, set()))

        return affected_metrics

    def analyze_construct_changes(
        self, old_udl: Optional[UDLRepresentation], new_udl: UDLRepresentation
    ) -> Set[str]:
        """
        Analyze what types of constructs have changed.

        Args:
            old_udl: Previous UDL representation (None if new file)
            new_udl: New UDL representation

        Returns:
            Set of changed construct types
        """
        if old_udl is None:
            # New file - all constructs are "changed"
            return self._extract_construct_types(new_udl)

        # Compare constructs between old and new
        old_constructs = self._extract_constructs(old_udl)
        new_constructs = self._extract_constructs(new_udl)

        changed_types = set()

        # Check for added/removed constructs
        old_keys = set(old_constructs.keys())
        new_keys = set(new_constructs.keys())

        added_constructs = new_keys - old_keys
        removed_constructs = old_keys - new_keys

        # Check for modified constructs
        common_constructs = old_keys & new_keys
        modified_constructs = set()

        for construct_name in common_constructs:
            if old_constructs[construct_name] != new_constructs[construct_name]:
                modified_constructs.add(construct_name)

        # Map construct changes to types
        all_changed = added_constructs | removed_constructs | modified_constructs

        for construct_name in all_changed:
            construct_type = self._get_construct_type(construct_name, new_udl)
            if construct_type:
                changed_types.add(construct_type)

        return changed_types

    def _extract_construct_types(self, udl: UDLRepresentation) -> Set[str]:
        """Extract all construct types from UDL."""
        types = set()

        # Analyze tokens to identify construct types
        tokens = udl.get_tokens()

        for token in tokens:
            if token.text == "::=":
                types.add("rule_definition")
            elif "|" in token.text:
                types.add("alternation")
            elif token.text.startswith('"') or token.text.startswith("'"):
                types.add("terminal")
            elif token.text.isupper():
                types.add("non_terminal")
            elif token.text.startswith("#"):
                types.add("comment")

        # Analyze grammar graph
        try:
            graph = udl.get_grammar_graph()
            if graph.number_of_nodes() > 0:
                types.add("grammar_rule")
            if graph.number_of_edges() > 0:
                types.add("dependency")
        except:
            pass

        return types

    def _extract_constructs(self, udl: UDLRepresentation) -> Dict[str, str]:
        """Extract constructs with their definitions."""
        constructs = {}

        try:
            # Extract from grammar graph
            graph = udl.get_grammar_graph()

            for node in graph.nodes():
                constructs[str(node)] = str(graph.nodes[node])

            for edge in graph.edges():
                edge_key = f"{edge[0]}->{edge[1]}"
                constructs[edge_key] = str(graph.edges[edge])

        except Exception as e:
            logger.debug(f"Failed to extract constructs: {e}")

        return constructs

    def _get_construct_type(
        self, construct_name: str, udl: UDLRepresentation
    ) -> Optional[str]:
        """Get the type of a construct."""
        # Simple heuristics for construct type detection
        if "->" in construct_name:
            return "dependency"
        elif construct_name.isupper():
            return "non_terminal"
        elif construct_name.startswith('"') or construct_name.startswith("'"):
            return "terminal"
        else:
            return "construct_definition"


class IncrementalCache:
    """
    Cache for incremental computation results.

    Stores file snapshots, metric results, and dependency information
    to enable efficient incremental updates.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize incremental cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir or Path.home() / ".udl_rating_cache" / "incremental"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self.file_snapshots = {}  # file_path -> FileSnapshot
        self.metric_results = {}  # (file_path, metric_name) -> result
        self.dependency_cache = {}  # file_path -> construct dependencies

        # Cache metadata
        self.cache_stats = {"hits": 0, "misses": 0,
                            "invalidations": 0, "updates": 0}

        self._lock = threading.RLock()

        # Load existing cache
        self._load_cache()

    def get_file_snapshot(self, file_path: Path) -> Optional[FileSnapshot]:
        """Get cached file snapshot."""
        with self._lock:
            return self.file_snapshots.get(str(file_path))

    def update_file_snapshot(self, file_path: Path, snapshot: FileSnapshot) -> None:
        """Update file snapshot in cache."""
        with self._lock:
            self.file_snapshots[str(file_path)] = snapshot
            self.cache_stats["updates"] += 1
            self._save_snapshot(snapshot)

    def get_metric_result(self, file_path: Path, metric_name: str) -> Optional[float]:
        """Get cached metric result."""
        with self._lock:
            cache_key = (str(file_path), metric_name)
            result = self.metric_results.get(cache_key)

            if result is not None:
                self.cache_stats["hits"] += 1
                return result
            else:
                self.cache_stats["misses"] += 1
                return None

    def update_metric_result(
        self, file_path: Path, metric_name: str, result: float
    ) -> None:
        """Update metric result in cache."""
        with self._lock:
            cache_key = (str(file_path), metric_name)
            self.metric_results[cache_key] = result
            self.cache_stats["updates"] += 1

    def invalidate_file(
        self, file_path: Path, affected_metrics: Optional[Set[str]] = None
    ) -> None:
        """
        Invalidate cache entries for a file.

        Args:
            file_path: Path to file to invalidate
            affected_metrics: Specific metrics to invalidate (all if None)
        """
        with self._lock:
            file_path_str = str(file_path)

            if affected_metrics is None:
                # Invalidate all metrics for this file
                keys_to_remove = [
                    key for key in self.metric_results.keys() if key[0] == file_path_str
                ]
            else:
                # Invalidate specific metrics
                keys_to_remove = [
                    (file_path_str, metric)
                    for metric in affected_metrics
                    if (file_path_str, metric) in self.metric_results
                ]

            for key in keys_to_remove:
                del self.metric_results[key]
                self.cache_stats["invalidations"] += 1

            logger.debug(
                f"Invalidated {len(keys_to_remove)} cache entries for {file_path}"
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.cache_stats["hits"] + \
                self.cache_stats["misses"]
            hit_rate = (
                self.cache_stats["hits"] /
                total_requests if total_requests > 0 else 0.0
            )

            return {
                **self.cache_stats,
                "hit_rate": hit_rate,
                "cached_files": len(self.file_snapshots),
                "cached_metrics": len(self.metric_results),
            }

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.file_snapshots.clear()
            self.metric_results.clear()
            self.dependency_cache.clear()

            # Clear cache files
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()

            logger.info("Incremental cache cleared")

    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            # Load snapshots
            snapshots_file = self.cache_dir / "snapshots.json"
            if snapshots_file.exists():
                with open(snapshots_file) as f:
                    snapshots_data = json.load(f)

                for file_path, snapshot_data in snapshots_data.items():
                    snapshot = FileSnapshot(**snapshot_data)
                    self.file_snapshots[file_path] = snapshot

            # Load metric results
            metrics_file = self.cache_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics_data = json.load(f)

                for key_str, result in metrics_data.items():
                    file_path, metric_name = key_str.split("|", 1)
                    self.metric_results[(file_path, metric_name)] = result

            logger.debug(
                f"Loaded incremental cache: {len(self.file_snapshots)} snapshots, "
                f"{len(self.metric_results)} metric results"
            )

        except Exception as e:
            logger.warning(f"Failed to load incremental cache: {e}")

    def _save_snapshot(self, snapshot: FileSnapshot) -> None:
        """Save individual snapshot to disk."""
        try:
            snapshots_file = self.cache_dir / "snapshots.json"

            # Load existing snapshots
            snapshots_data = {}
            if snapshots_file.exists():
                with open(snapshots_file) as f:
                    snapshots_data = json.load(f)

            # Update with new snapshot
            snapshots_data[snapshot.file_path] = asdict(snapshot)

            # Save back to disk
            with open(snapshots_file, "w") as f:
                json.dump(snapshots_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")

    def save_cache(self) -> None:
        """Save cache to disk."""
        with self._lock:
            try:
                # Save snapshots
                snapshots_file = self.cache_dir / "snapshots.json"
                snapshots_data = {
                    file_path: asdict(snapshot)
                    for file_path, snapshot in self.file_snapshots.items()
                }

                with open(snapshots_file, "w") as f:
                    json.dump(snapshots_data, f, indent=2)

                # Save metric results
                metrics_file = self.cache_dir / "metrics.json"
                metrics_data = {
                    f"{file_path}|{metric_name}": result
                    for (file_path, metric_name), result in self.metric_results.items()
                }

                with open(metrics_file, "w") as f:
                    json.dump(metrics_data, f, indent=2)

                logger.debug("Incremental cache saved to disk")

            except Exception as e:
                logger.warning(f"Failed to save incremental cache: {e}")


class IncrementalProcessor:
    """
    Incremental processor for UDL files.

    Provides efficient processing that only recomputes changed parts
    of UDL files with smart caching and dependency tracking.
    """

    def __init__(
        self, cache_dir: Optional[Path] = None, metric_names: List[str] = None
    ):
        """
        Initialize incremental processor.

        Args:
            cache_dir: Directory for cache storage
            metric_names: List of metrics to compute
        """
        self.metric_names = metric_names or [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]

        # Initialize components
        self.cache = IncrementalCache(cache_dir)
        self.dependency_tracker = DependencyTracker()
        self.pipeline = RatingPipeline(metric_names=self.metric_names)

        # Processing statistics
        self.stats = {
            "files_processed": 0,
            "cache_hits": 0,
            "incremental_updates": 0,
            "full_recomputes": 0,
            "total_time_saved": 0.0,
        }

    def process_file(self, file_path: Path) -> IncrementalResult:
        """
        Process file with incremental computation.

        Args:
            file_path: Path to UDL file

        Returns:
            IncrementalResult with processing information
        """
        start_time = time.time()

        logger.debug(f"Processing file incrementally: {file_path}")

        # Create new snapshot
        new_snapshot = FileSnapshot.from_file(file_path)
        old_snapshot = self.cache.get_file_snapshot(file_path)

        # Detect changes
        change_set = self._detect_changes(old_snapshot, new_snapshot)

        # Determine processing strategy
        if change_set.change_type == "added" or old_snapshot is None:
            # New file - full computation
            result = self._process_full(file_path, new_snapshot, change_set)
        elif change_set.change_type == "deleted":
            # File deleted - invalidate cache
            self.cache.invalidate_file(file_path)
            raise FileNotFoundError(f"File not found: {file_path}")
        elif change_set.get_change_ratio() > 0.5:
            # Major changes - full recomputation
            result = self._process_full(file_path, new_snapshot, change_set)
        else:
            # Minor changes - incremental update
            result = self._process_incremental(
                file_path, new_snapshot, change_set)

        # Update cache
        self.cache.update_file_snapshot(file_path, new_snapshot)

        # Update statistics
        self.stats["files_processed"] += 1
        if result.was_cached:
            self.stats["cache_hits"] += 1
        if result.was_incremental:
            self.stats["incremental_updates"] += 1
        else:
            self.stats["full_recomputes"] += 1

        processing_time = time.time() - start_time
        result.processing_time = processing_time

        logger.debug(
            f"Incremental processing completed: {file_path} "
            f"(cached: {result.was_cached}, incremental: {result.was_incremental}, "
            f"time: {processing_time:.3f}s)"
        )

        return result

    def process_multiple_files(self, file_paths: List[Path]) -> List[IncrementalResult]:
        """
        Process multiple files with incremental computation.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of IncrementalResults
        """
        logger.info(f"Processing {len(file_paths)} files incrementally")

        results = []

        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

                # Create error result
                error_result = IncrementalResult(
                    file_path=str(file_path),
                    was_cached=False,
                    was_incremental=False,
                    change_ratio=1.0,
                    processing_time=0.0,
                    cache_hit_ratio=0.0,
                    recomputed_metrics=set(),
                    report=QualityReport(
                        overall_score=0.0,
                        confidence=0.0,
                        metric_scores={},
                        metric_formulas={},
                        computation_trace=[],
                        error_bounds={},
                        timestamp=time.time(),
                        udl_file=str(file_path),
                        errors=[str(e)],
                    ),
                )
                results.append(error_result)

        return results

    def _detect_changes(
        self, old_snapshot: Optional[FileSnapshot], new_snapshot: FileSnapshot
    ) -> ChangeSet:
        """Detect changes between file snapshots."""
        if old_snapshot is None:
            return ChangeSet(
                file_path=new_snapshot.file_path,
                change_type="added",
                new_snapshot=new_snapshot,
            )

        if old_snapshot.content_hash == new_snapshot.content_hash:
            return ChangeSet(
                file_path=new_snapshot.file_path,
                change_type="unchanged",
                old_snapshot=old_snapshot,
                new_snapshot=new_snapshot,
            )

        # Analyze specific changes
        change_set = ChangeSet(
            file_path=new_snapshot.file_path,
            change_type="modified",
            old_snapshot=old_snapshot,
            new_snapshot=new_snapshot,
        )

        # TODO: Implement detailed change analysis
        # For now, assume all lines are potentially affected
        change_set.affected_lines = set(range(1, new_snapshot.line_count + 1))

        return change_set

    def _process_full(
        self, file_path: Path, snapshot: FileSnapshot, change_set: ChangeSet
    ) -> IncrementalResult:
        """Process file with full computation."""
        logger.debug(f"Full recomputation for: {file_path}")

        # Read file and create UDL representation
        content = file_path.read_text(encoding="utf-8")
        udl = UDLRepresentation(content, str(file_path))

        # Compute all metrics
        report = self.pipeline.compute_rating(udl)

        # Cache metric results
        for metric_name, score in report.metric_scores.items():
            self.cache.update_metric_result(file_path, metric_name, score)

        return IncrementalResult(
            file_path=str(file_path),
            was_cached=False,
            was_incremental=False,
            change_ratio=change_set.get_change_ratio(),
            processing_time=0.0,  # Will be set by caller
            cache_hit_ratio=0.0,
            recomputed_metrics=set(report.metric_scores.keys()),
            report=report,
            change_set=change_set,
        )

    def _process_incremental(
        self, file_path: Path, snapshot: FileSnapshot, change_set: ChangeSet
    ) -> IncrementalResult:
        """Process file with incremental computation."""
        logger.debug(f"Incremental update for: {file_path}")

        # Read file and create UDL representation
        content = file_path.read_text(encoding="utf-8")
        new_udl = UDLRepresentation(content, str(file_path))

        # Try to load old UDL representation (simplified - would need better caching)
        old_udl = None  # TODO: Implement old UDL caching if needed

        # Analyze construct changes
        changed_constructs = self.dependency_tracker.analyze_construct_changes(
            old_udl, new_udl
        )
        affected_metrics = self.dependency_tracker.get_affected_metrics(
            changed_constructs
        )

        # Get cached metric results
        cached_metrics = {}
        recomputed_metrics = set()

        for metric_name in self.metric_names:
            if metric_name in affected_metrics:
                # Recompute affected metric
                recomputed_metrics.add(metric_name)
            else:
                # Try to use cached result
                cached_result = self.cache.get_metric_result(
                    file_path, metric_name)
                if cached_result is not None:
                    cached_metrics[metric_name] = cached_result
                else:
                    # Cache miss - need to recompute
                    recomputed_metrics.add(metric_name)

        # Recompute only affected metrics
        if recomputed_metrics:
            # Create pipeline with only affected metrics
            partial_pipeline = RatingPipeline(
                metric_names=list(recomputed_metrics))
            partial_report = partial_pipeline.compute_rating(new_udl)

            # Update cache with new results
            for metric_name, score in partial_report.metric_scores.items():
                self.cache.update_metric_result(file_path, metric_name, score)
                cached_metrics[metric_name] = score

        # Combine cached and recomputed results
        all_metric_scores = cached_metrics

        # Create aggregated report
        from udl_rating_framework.core.aggregation import MetricAggregator

        aggregator = MetricAggregator(
            {name: 1.0 / len(all_metric_scores) for name in all_metric_scores}
        )
        overall_score = aggregator.aggregate(all_metric_scores)

        # Calculate confidence (simplified)
        confidence = (
            0.8 if recomputed_metrics else 0.9
        )  # Lower confidence for partial recomputation

        report = QualityReport(
            overall_score=overall_score,
            confidence=confidence,
            metric_scores=all_metric_scores,
            metric_formulas={},  # TODO: Include formulas
            computation_trace=[],  # TODO: Include trace
            error_bounds={},  # TODO: Include error bounds
            timestamp=time.time(),
            udl_file=str(file_path),
            errors=[],
        )

        cache_hit_ratio = (
            len(cached_metrics) /
            len(self.metric_names) if self.metric_names else 0.0
        )

        return IncrementalResult(
            file_path=str(file_path),
            was_cached=len(cached_metrics) > 0,
            was_incremental=True,
            change_ratio=change_set.get_change_ratio(),
            processing_time=0.0,  # Will be set by caller
            cache_hit_ratio=cache_hit_ratio,
            recomputed_metrics=recomputed_metrics,
            report=report,
            change_set=change_set,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        cache_stats = self.cache.get_cache_stats()

        return {**self.stats, "cache_statistics": cache_stats}

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear_cache()
        self.stats = {
            "files_processed": 0,
            "cache_hits": 0,
            "incremental_updates": 0,
            "full_recomputes": 0,
            "total_time_saved": 0.0,
        }


# Convenience functions
def process_udl_incremental(
    file_path: Path, metric_names: List[str] = None, cache_dir: Optional[Path] = None
) -> IncrementalResult:
    """
    Convenience function to process UDL file incrementally.

    Args:
        file_path: Path to UDL file
        metric_names: List of metrics to compute
        cache_dir: Cache directory

    Returns:
        IncrementalResult
    """
    processor = IncrementalProcessor(cache_dir, metric_names)
    return processor.process_file(file_path)


def process_udl_directory_incremental(
    directory: Path,
    metric_names: List[str] = None,
    cache_dir: Optional[Path] = None,
    file_patterns: List[str] = None,
) -> List[IncrementalResult]:
    """
    Convenience function to process UDL directory incrementally.

    Args:
        directory: Directory containing UDL files
        metric_names: List of metrics to compute
        cache_dir: Cache directory
        file_patterns: File patterns to match

    Returns:
        List of IncrementalResults
    """
    from udl_rating_framework.io.file_discovery import FileDiscovery

    # Discover files
    if file_patterns is None:
        file_patterns = ["*.udl", "*.dsl", "*.grammar", "*.ebnf"]

    discovery = FileDiscovery()
    extensions = [p.replace("*", "") for p in file_patterns]
    files = discovery.discover_files(directory, extensions)

    # Process incrementally
    processor = IncrementalProcessor(cache_dir, metric_names)
    return processor.process_multiple_files(files)
