"""
Caching module for UDL Rating Framework.

Provides caching functionality for parsed UDL representations and computed metrics
to improve performance during batch processing and repeated evaluations.
"""

import hashlib
import logging
import pickle
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

from udl_rating_framework.core.representation import UDLRepresentation

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    data: Any
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    file_hash: Optional[str] = None
    file_mtime: Optional[float] = None


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.

    Provides automatic eviction of least recently used items when capacity is exceeded.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: Dict[str, datetime] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                self._access_order[key] = entry.last_accessed
                return entry.data
            return None

    def put(
        self,
        key: str,
        value: Any,
        file_hash: Optional[str] = None,
        file_mtime: Optional[float] = None,
    ) -> None:
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
            file_hash: Optional file hash for invalidation
            file_mtime: Optional file modification time
        """
        with self._lock:
            now = datetime.now()

            # If cache is full, evict least recently used item
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            entry = CacheEntry(
                data=value,
                created_at=now,
                access_count=1,
                last_accessed=now,
                file_hash=file_hash,
                file_mtime=file_mtime,
            )

            self._cache[key] = entry
            self._access_order[key] = now

    def invalidate(self, key: str) -> bool:
        """
        Remove item from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if item was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_order[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_order:
            return

        # Find least recently used key
        lru_key = min(self._access_order.keys(),
                      key=lambda k: self._access_order[k])

        # Remove from cache
        del self._cache[lru_key]
        del self._access_order[lru_key]

        logger.debug(f"Evicted LRU cache entry: {lru_key}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(
                entry.access_count for entry in self._cache.values())

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "average_accesses": (
                    total_accesses / len(self._cache) if self._cache else 0
                ),
                "oldest_entry": min(
                    (entry.created_at for entry in self._cache.values()), default=None
                ),
                "newest_entry": max(
                    (entry.created_at for entry in self._cache.values()), default=None
                ),
            }


class UDLRepresentationCache:
    """
    Cache for parsed UDL representations.

    Provides file-based invalidation and automatic cache management.
    """

    def __init__(self, max_size: int = 500, ttl_hours: int = 24):
        """
        Initialize UDL representation cache.

        Args:
            max_size: Maximum number of UDL representations to cache
            ttl_hours: Time-to-live in hours for cache entries
        """
        self.cache = LRUCache(max_size)
        self.ttl = timedelta(hours=ttl_hours)

    def get_udl(
        self, file_path: Union[str, Path], content: Optional[str] = None
    ) -> Optional[UDLRepresentation]:
        """
        Get cached UDL representation.

        Args:
            file_path: Path to UDL file
            content: Optional content for hash-based caching

        Returns:
            Cached UDLRepresentation or None if not found/invalid
        """
        file_path = Path(file_path)
        cache_key = self._get_cache_key(file_path, content)

        entry = self.cache.get(cache_key)
        if entry is None:
            return None

        # Check if entry is still valid
        if not self._is_entry_valid(entry, file_path):
            self.cache.invalidate(cache_key)
            return None

        logger.debug(f"Cache hit for UDL: {file_path}")
        return entry

    def put_udl(
        self,
        file_path: Union[str, Path],
        udl: UDLRepresentation,
        content: Optional[str] = None,
    ) -> None:
        """
        Cache UDL representation.

        Args:
            file_path: Path to UDL file
            udl: UDL representation to cache
            content: Optional content for hash-based caching
        """
        file_path = Path(file_path)
        cache_key = self._get_cache_key(file_path, content)

        # Get file metadata for invalidation
        file_hash = None
        file_mtime = None

        if content:
            file_hash = self._compute_content_hash(content)

        if file_path.exists():
            try:
                file_mtime = file_path.stat().st_mtime
            except OSError:
                pass

        self.cache.put(cache_key, udl, file_hash=file_hash,
                       file_mtime=file_mtime)
        logger.debug(f"Cached UDL representation: {file_path}")

    def invalidate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Invalidate cache entry for specific file.

        Args:
            file_path: Path to file to invalidate

        Returns:
            True if entry was invalidated
        """
        file_path = Path(file_path)
        cache_key = self._get_cache_key(file_path)
        return self.cache.invalidate(cache_key)

    def clear(self) -> None:
        """Clear all cached UDL representations."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def _get_cache_key(self, file_path: Path, content: Optional[str] = None) -> str:
        """
        Generate cache key for file.

        Args:
            file_path: Path to file
            content: Optional content for hash-based key

        Returns:
            Cache key string
        """
        if content:
            # Use content hash for in-memory content
            content_hash = self._compute_content_hash(content)
            return f"content:{content_hash}"
        else:
            # Use file path for file-based content
            return f"file:{str(file_path.resolve())}"

    def _compute_content_hash(self, content: str) -> str:
        """
        Compute hash of content.

        Args:
            content: Content to hash

        Returns:
            SHA-256 hash of content
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _is_entry_valid(self, entry: Any, file_path: Path) -> bool:
        """
        Check if cache entry is still valid.

        Args:
            entry: Cache entry to validate
            file_path: Path to original file

        Returns:
            True if entry is valid
        """
        # Get the actual cache entry
        cache_entry = None
        cache_key = self._get_cache_key(file_path)

        with self.cache._lock:
            if cache_key in self.cache._cache:
                cache_entry = self.cache._cache[cache_key]

        if cache_entry is None:
            return False

        # Check TTL
        if datetime.now() - cache_entry.created_at > self.ttl:
            logger.debug(f"Cache entry expired for: {file_path}")
            return False

        # Check file modification time if available
        if cache_entry.file_mtime is not None and file_path.exists():
            try:
                current_mtime = file_path.stat().st_mtime
                if current_mtime != cache_entry.file_mtime:
                    logger.debug(
                        f"File modified, cache invalid for: {file_path}")
                    return False
            except OSError:
                # If we can't check file stats, assume invalid
                return False

        return True


class MetricCache:
    """
    Cache for computed metric values.

    Provides caching of expensive metric computations with automatic invalidation.
    """

    def __init__(self, max_size: int = 1000, ttl_hours: int = 12):
        """
        Initialize metric cache.

        Args:
            max_size: Maximum number of metric results to cache
            ttl_hours: Time-to-live in hours for cache entries
        """
        self.cache = LRUCache(max_size)
        self.ttl = timedelta(hours=ttl_hours)

    def get_metric(self, udl_hash: str, metric_name: str) -> Optional[float]:
        """
        Get cached metric value.

        Args:
            udl_hash: Hash of UDL representation
            metric_name: Name of metric

        Returns:
            Cached metric value or None if not found/invalid
        """
        cache_key = f"{metric_name}:{udl_hash}"

        entry = self.cache.get(cache_key)
        if entry is None:
            return None

        # Check TTL
        with self.cache._lock:
            if cache_key in self.cache._cache:
                cache_entry = self.cache._cache[cache_key]
                if datetime.now() - cache_entry.created_at > self.ttl:
                    self.cache.invalidate(cache_key)
                    return None

        logger.debug(f"Cache hit for metric {metric_name}")
        return entry

    def put_metric(self, udl_hash: str, metric_name: str, value: float) -> None:
        """
        Cache metric value.

        Args:
            udl_hash: Hash of UDL representation
            metric_name: Name of metric
            value: Metric value to cache
        """
        cache_key = f"{metric_name}:{udl_hash}"
        self.cache.put(cache_key, value)
        logger.debug(f"Cached metric {metric_name}")

    def invalidate_udl(self, udl_hash: str) -> int:
        """
        Invalidate all metrics for a UDL.

        Args:
            udl_hash: Hash of UDL representation

        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        keys_to_remove = []

        with self.cache._lock:
            for key in self.cache._cache.keys():
                if key.endswith(f":{udl_hash}"):
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            if self.cache.invalidate(key):
                invalidated += 1

        return invalidated

    def clear(self) -> None:
        """Clear all cached metrics."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


# Global cache instances
_udl_cache = None
_metric_cache = None
_cache_lock = threading.Lock()


def get_udl_cache() -> UDLRepresentationCache:
    """Get global UDL representation cache instance."""
    global _udl_cache
    if _udl_cache is None:
        with _cache_lock:
            if _udl_cache is None:
                _udl_cache = UDLRepresentationCache()
    return _udl_cache


def get_metric_cache() -> MetricCache:
    """Get global metric cache instance."""
    global _metric_cache
    if _metric_cache is None:
        with _cache_lock:
            if _metric_cache is None:
                _metric_cache = MetricCache()
    return _metric_cache


def clear_all_caches() -> None:
    """Clear all global caches."""
    global _udl_cache, _metric_cache
    with _cache_lock:
        if _udl_cache:
            _udl_cache.clear()
        if _metric_cache:
            _metric_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    stats = {}

    if _udl_cache:
        stats["udl_cache"] = _udl_cache.get_stats()

    if _metric_cache:
        stats["metric_cache"] = _metric_cache.get_stats()

    return stats
