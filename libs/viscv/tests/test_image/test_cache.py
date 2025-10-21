"""Unit tests for ImageCache class."""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from viscv.image import ImageCache


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary test image file."""
    img_path = tmp_path / "test_image.jpg"
    # Just create the path - we'll use it for mtime testing
    img_path.touch()
    return img_path


class TestImageCache:
    """Test suite for ImageCache class."""

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization creates necessary directories and database."""
        cache = ImageCache(cache_dir=temp_cache_dir, max_size_gb=1.0, enabled=True)

        assert cache.cache_dir.exists()
        assert cache.db_path.exists()
        assert cache.enabled is True
        assert cache.max_size_bytes == 1024 * 1024 * 1024

    def test_cache_disabled(self):
        """Test that disabled cache doesn't create any files."""
        cache = ImageCache(enabled=False)

        assert cache.enabled is False
        # get() should return None when disabled
        result = cache.get("any_path.jpg", (100, 100))
        assert result is None

    def test_cache_key_generation(self, temp_cache_dir, temp_image_file):
        """Test cache key generation is consistent."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        img_path = str(temp_image_file)
        target_size = (224, 224)
        mtime = temp_image_file.stat().st_mtime

        key1 = cache._generate_cache_key(img_path, target_size, mtime)
        key2 = cache._generate_cache_key(img_path, target_size, mtime)

        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length

    def test_cache_miss(self, temp_cache_dir, temp_image_file):
        """Test cache miss returns None."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        result = cache.get(str(temp_image_file), (224, 224))
        assert result is None

    def test_cache_put_and_get(self, temp_cache_dir, temp_image_file):
        """Test basic cache put and get operations."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        # Create test image
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_path = str(temp_image_file)
        target_size = (224, 224)

        # Put image in cache
        cache.put(img_path, target_size, test_img)

        # Get image from cache
        cached_img = cache.get(img_path, target_size)

        assert cached_img is not None
        assert np.array_equal(cached_img, test_img)
        assert cached_img.shape == test_img.shape
        assert cached_img.dtype == test_img.dtype

    def test_cache_different_sizes(self, temp_cache_dir, temp_image_file):
        """Test caching same image at different sizes."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        img_path = str(temp_image_file)
        test_img_224 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_img_512 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Cache at two different sizes
        cache.put(img_path, (224, 224), test_img_224)
        cache.put(img_path, (512, 512), test_img_512)

        # Retrieve both
        cached_224 = cache.get(img_path, (224, 224))
        cached_512 = cache.get(img_path, (512, 512))

        assert np.array_equal(cached_224, test_img_224)
        assert np.array_equal(cached_512, test_img_512)

    def test_cache_invalidation_on_mtime_change(self, temp_cache_dir, temp_image_file):
        """Test cache invalidates when file mtime changes."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        img_path = str(temp_image_file)
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Cache the image
        cache.put(img_path, (224, 224), test_img)
        assert cache.get(img_path, (224, 224)) is not None

        # Modify file mtime
        time.sleep(0.01)  # Ensure mtime changes
        temp_image_file.touch()

        # Cache should miss now
        result = cache.get(img_path, (224, 224))
        assert result is None

    def test_cache_access_count(self, temp_cache_dir, temp_image_file):
        """Test cache tracks access count."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        img_path = str(temp_image_file)
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Cache the image
        cache.put(img_path, (224, 224), test_img)

        # Access multiple times
        for _ in range(5):
            cache.get(img_path, (224, 224))

        # Check stats
        stats = cache.get_stats()
        assert stats["enabled"] is True
        assert stats["total_entries"] == 1
        # Access count should be at least 5 (may be more due to put operation)
        assert stats["avg_access_count"] >= 5

    def test_cache_eviction(self, temp_cache_dir, temp_image_file):
        """Test LRU eviction when cache size exceeds limit."""
        # Create cache with very small size limit (1 MB)
        cache = ImageCache(cache_dir=temp_cache_dir, max_size_gb=0.001, enabled=True)

        img_path = str(temp_image_file)

        # Create several large images to exceed cache size
        images = []
        for i in range(5):
            # Each image is ~500KB
            img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            images.append(img)
            cache.put(img_path, (400 + i, 400 + i), img)

        # Check that eviction occurred
        stats = cache.get_stats()
        # Should have fewer than 5 entries due to eviction
        assert stats["total_entries"] < 5

    def test_cache_clear(self, temp_cache_dir, temp_image_file):
        """Test clearing all cache entries."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        img_path = str(temp_image_file)
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Add some entries
        cache.put(img_path, (224, 224), test_img)
        cache.put(img_path, (512, 512), test_img)

        assert cache.get_stats()["total_entries"] == 2

        # Clear cache
        cache.clear()

        # Verify cache is empty
        assert cache.get_stats()["total_entries"] == 0
        assert cache.get(img_path, (224, 224)) is None

    def test_cache_corrupted_file_handling(self, temp_cache_dir, temp_image_file):
        """Test cache handles corrupted cache files gracefully."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        img_path = str(temp_image_file)
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Cache the image
        cache.put(img_path, (224, 224), test_img)

        # Get cache key and corrupt the file
        mtime = temp_image_file.stat().st_mtime
        cache_key = cache._generate_cache_key(img_path, (224, 224), mtime)
        cache_path = cache._get_cache_path(cache_key)

        # Corrupt the cache file
        with open(cache_path, "wb") as f:
            f.write(b"corrupted data")

        # Should return None and clean up corrupted file
        result = cache.get(img_path, (224, 224))
        assert result is None

    def test_cache_nonexistent_file(self, temp_cache_dir):
        """Test cache handles non-existent image files gracefully."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        # Try to get cache for non-existent file
        result = cache.get("/nonexistent/path.jpg", (224, 224))
        assert result is None

        # Try to put cache for non-existent file (should be no-op)
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cache.put("/nonexistent/path.jpg", (224, 224), test_img)

        # Verify nothing was cached
        assert cache.get_stats()["total_entries"] == 0

    def test_cache_stats(self, temp_cache_dir, temp_image_file):
        """Test cache statistics are accurate."""
        cache = ImageCache(cache_dir=temp_cache_dir, max_size_gb=1.0, enabled=True)

        stats = cache.get_stats()
        assert stats["enabled"] is True
        assert stats["total_entries"] == 0
        assert stats["total_size_mb"] == 0
        assert "cache_dir" in stats

        # Add some entries
        img_path = str(temp_image_file)
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cache.put(img_path, (224, 224), test_img)

        stats = cache.get_stats()
        assert stats["total_entries"] == 1
        assert stats["total_size_mb"] > 0

    def test_cache_repr(self, temp_cache_dir):
        """Test cache string representation."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)
        repr_str = repr(cache)

        assert "ImageCache" in repr_str
        assert str(temp_cache_dir) in repr_str

        # Test disabled cache repr
        disabled_cache = ImageCache(enabled=False)
        disabled_repr = repr(disabled_cache)
        assert "enabled=False" in disabled_repr

    def test_cache_subdirectory_structure(self, temp_cache_dir, temp_image_file):
        """Test cache creates subdirectories for better filesystem performance."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        img_path = str(temp_image_file)
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Cache the image
        cache.put(img_path, (224, 224), test_img)

        # Verify subdirectory structure exists
        # Cache should create subdirectories based on first 2 chars of hash
        subdirs = [d for d in temp_cache_dir.iterdir() if d.is_dir()]
        assert len(subdirs) > 0  # At least one subdirectory should exist

    def test_cache_concurrent_access(self, temp_cache_dir, temp_image_file):
        """Test cache handles multiple access patterns."""
        cache = ImageCache(cache_dir=temp_cache_dir, enabled=True)

        img_path = str(temp_image_file)
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Cache the image
        cache.put(img_path, (224, 224), test_img)

        # Multiple reads
        results = [cache.get(img_path, (224, 224)) for _ in range(10)]

        # All should return the same image
        assert all(r is not None for r in results)
        assert all(np.array_equal(r, test_img) for r in results)
