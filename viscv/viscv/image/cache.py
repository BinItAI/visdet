"""On-disk cache for downsized images to avoid repeated decoding and resizing."""

import hashlib
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np


class ImageCache:
    """On-disk cache for downsized images.

    This cache stores downsized images on disk after the first load, avoiding
    repeated JPEG decoding and resizing operations in subsequent training epochs.

    The cache uses:
    - NPY format for fast numpy array serialization
    - MD5 hashing for cache keys (based on image path, target size, and modification time)
    - SQLite for metadata and LRU tracking
    - Automatic eviction when cache size exceeds max_size_gb

    Args:
        cache_dir: Directory to store cached images. Defaults to ~/.cache/viscv/image_cache
        max_size_gb: Maximum cache size in gigabytes. When exceeded, least recently used
            images are evicted. Defaults to 10.0 GB.
        enabled: Whether caching is enabled. Defaults to True.
    """

    def __init__(
        self,
        cache_dir: Path | str = Path.home() / ".cache" / "viscv" / "image_cache",
        max_size_gb: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        if not self.enabled:
            return

        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for cache metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table for cache entries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                img_path TEXT NOT NULL,
                target_width INTEGER NOT NULL,
                target_height INTEGER NOT NULL,
                mtime REAL NOT NULL,
                file_size INTEGER NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 1
            )
        """)

        conn.commit()
        conn.close()

    def _generate_cache_key(self, img_path: str, target_size: tuple[int, int], mtime: float) -> str:
        """Generate unique cache key for an image.

        Args:
            img_path: Path to original image
            target_size: Target size (width, height) after resize
            mtime: Modification time of original image

        Returns:
            MD5 hash as cache key
        """
        key_str = f"{img_path}:{target_size[0]}x{target_size[1]}:{mtime:.6f}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached image.

        Args:
            cache_key: Cache key hash

        Returns:
            Path to NPY file
        """
        # Use first 2 chars of hash for subdirectory (256 buckets)
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.npy"

    def get(self, img_path: str, target_size: tuple[int, int]) -> np.ndarray | None:
        """Get cached image if available.

        Args:
            img_path: Path to original image
            target_size: Target size (width, height) after resize

        Returns:
            Cached image array, or None if not in cache or cache is disabled
        """
        if not self.enabled:
            return None

        # Check if original file exists and get mtime
        img_file = Path(img_path)
        if not img_file.exists():
            return None

        mtime = img_file.stat().st_mtime
        cache_key = self._generate_cache_key(img_path, target_size, mtime)
        cache_path = self._get_cache_path(cache_key)

        # Check if cached file exists
        if not cache_path.exists():
            return None

        # Verify cache entry is valid (check mtime hasn't changed)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mtime FROM cache_entries WHERE cache_key = ?",
            (cache_key,),
        )
        result = cursor.fetchone()

        if result is None:
            # Cache file exists but no metadata - clean up
            cache_path.unlink(missing_ok=True)
            conn.close()
            return None

        cached_mtime = result[0]
        if abs(cached_mtime - mtime) > 0.001:  # Allow small floating point diff
            # Image has been modified - invalidate cache
            self._remove_entry(cache_key, cache_path, cursor)
            conn.commit()
            conn.close()
            return None

        # Load cached image
        try:
            img = np.load(cache_path)

            # Update access time and count
            now = time.time()
            cursor.execute(
                """
                UPDATE cache_entries
                SET last_accessed = ?, access_count = access_count + 1
                WHERE cache_key = ?
                """,
                (now, cache_key),
            )
            conn.commit()
            conn.close()

            return img

        except Exception:
            # Failed to load - clean up corrupted cache
            self._remove_entry(cache_key, cache_path, cursor)
            conn.commit()
            conn.close()
            return None

    def put(self, img_path: str, target_size: tuple[int, int], img: np.ndarray) -> None:
        """Store image in cache.

        Args:
            img_path: Path to original image
            target_size: Target size (width, height) after resize
            img: Image array to cache
        """
        if not self.enabled:
            return

        # Get original file mtime
        img_file = Path(img_path)
        if not img_file.exists():
            return

        mtime = img_file.stat().st_mtime
        cache_key = self._generate_cache_key(img_path, target_size, mtime)
        cache_path = self._get_cache_path(cache_key)

        # Save to disk
        try:
            np.save(cache_path, img)
            file_size = cache_path.stat().st_size

            # Add metadata
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            now = time.time()
            cursor.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (cache_key, img_path, target_width, target_height, mtime, file_size, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    cache_key,
                    img_path,
                    target_size[0],
                    target_size[1],
                    mtime,
                    file_size,
                    now,
                ),
            )

            conn.commit()
            conn.close()

            # Check if we need to evict
            self._maybe_evict()

        except Exception:
            # Failed to save - clean up
            cache_path.unlink(missing_ok=True)

    def _remove_entry(self, cache_key: str, cache_path: Path, cursor: sqlite3.Cursor) -> None:
        """Remove a cache entry.

        Args:
            cache_key: Cache key to remove
            cache_path: Path to cached file
            cursor: SQLite cursor (must be committed by caller)
        """
        cache_path.unlink(missing_ok=True)
        cursor.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))

    def _maybe_evict(self) -> None:
        """Evict least recently used entries if cache size exceeds max_size_bytes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get total cache size
        cursor.execute("SELECT SUM(file_size) FROM cache_entries")
        result = cursor.fetchone()
        total_size = result[0] if result[0] is not None else 0

        if total_size <= self.max_size_bytes:
            conn.close()
            return

        # Evict LRU entries until under limit
        bytes_to_free = total_size - self.max_size_bytes

        cursor.execute(
            """
            SELECT cache_key, file_size
            FROM cache_entries
            ORDER BY last_accessed ASC
            """,
        )

        freed_bytes = 0
        for cache_key, file_size in cursor.fetchall():
            cache_path = self._get_cache_path(cache_key)
            self._remove_entry(cache_key, cache_path, cursor)
            freed_bytes += file_size

            if freed_bytes >= bytes_to_free:
                break

        conn.commit()
        conn.close()

    def clear(self) -> None:
        """Clear all cached images."""
        if not self.enabled:
            return

        # Remove all cache files
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Reinitialize database
        self._init_db()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            - total_entries: Number of cached images
            - total_size_mb: Total cache size in megabytes
            - hit_rate: Estimated cache hit rate (based on access counts)
        """
        if not self.enabled:
            return {"enabled": False}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                COUNT(*) as total_entries,
                SUM(file_size) as total_size,
                AVG(access_count) as avg_access_count
            FROM cache_entries
            """,
        )

        result = cursor.fetchone()
        conn.close()

        total_entries = result[0] if result[0] is not None else 0
        total_size = result[1] if result[1] is not None else 0
        avg_access = result[2] if result[2] is not None else 0

        return {
            "enabled": True,
            "total_entries": total_entries,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "avg_access_count": float(avg_access),
            "cache_dir": str(self.cache_dir),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        if not stats["enabled"]:
            return f"{self.__class__.__name__}(enabled=False)"

        return (
            f"{self.__class__.__name__}("
            f"cache_dir={stats['cache_dir']}, "
            f"entries={stats['total_entries']}, "
            f"size={stats['total_size_gb']:.2f}GB)"
        )
