"""Integration tests for image cache with transform pipeline."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from viscv.transforms.loading import LoadImageFromFile
from viscv.transforms.processing import Resize


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image file."""
    img_path = tmp_path / "test_image.jpg"
    # Create and save a test image
    test_img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    Image.fromarray(test_img).save(img_path)
    return str(img_path)


class TestCacheIntegration:
    """Test suite for cache integration with transforms."""

    def test_load_and_resize_with_cache(self, temp_cache_dir, temp_image):
        """Test complete pipeline: load image with cache, resize, cache resized."""
        # Create transforms with cache enabled
        load_transform = LoadImageFromFile(
            enable_cache=True,
            cache_dir=str(temp_cache_dir),
            cache_max_size_gb=1.0,
        )
        resize_transform = Resize(scale=(224, 224), keep_ratio=True, enable_cache=True)

        # First pass - should be cache miss
        results = {"img_path": temp_image}
        results = load_transform(results)
        assert results is not None
        assert results["_cache_hit"] is False
        assert "img" in results

        # Resize
        results = resize_transform(results)
        assert results is not None
        assert "img" in results
        assert results["img_shape"][0] <= 224
        assert results["img_shape"][1] <= 224

        # Second pass - should be cache hit
        # Need to provide target size from first run
        results2 = {
            "img_path": temp_image,
            "_cache_target_size": results["_cache_target_size"],
        }
        results2 = load_transform(results2)
        assert results2 is not None
        assert results2["_cache_hit"] is True  # Should hit cache this time

        # Verify cached image is correct
        assert results2["img"].shape[:2] == results["img"].shape[:2]

    def test_cache_disabled_by_default_in_resize(self, temp_cache_dir, temp_image):
        """Test that resize still works when cache is disabled."""
        load_transform = LoadImageFromFile()  # Cache disabled
        resize_transform = Resize(scale=(224, 224), enable_cache=True)

        results = {"img_path": temp_image}
        results = load_transform(results)
        results = resize_transform(results)

        assert results is not None
        assert "img" in results

    def test_multiple_sizes_cached_separately(self, temp_cache_dir, temp_image):
        """Test that different resize sizes are cached separately."""
        load_transform = LoadImageFromFile(
            enable_cache=True,
            cache_dir=str(temp_cache_dir),
        )
        resize_224 = Resize(scale=(224, 224), enable_cache=True)
        resize_512 = Resize(scale=(512, 512), enable_cache=True)

        # Process at 224x224
        results_224 = {"img_path": temp_image}
        results_224 = load_transform(results_224)
        results_224 = resize_224(results_224)

        # Process at 512x512
        results_512 = {"img_path": temp_image}
        results_512 = load_transform(results_512)
        results_512 = resize_512(results_512)

        # Verify both are cached
        stats = load_transform.cache.get_stats()
        assert stats["total_entries"] == 2

        # Verify second load hits cache for 224 (need to provide target size)
        results_224_cached = {
            "img_path": temp_image,
            "_cache_target_size": results_224["_cache_target_size"],
        }
        results_224_cached = load_transform(results_224_cached)
        assert results_224_cached["_cache_hit"] is True
        assert results_224_cached["img"].shape[:2] == results_224["img"].shape[:2]

    def test_cache_instance_passed_through_pipeline(self, temp_cache_dir, temp_image):
        """Test that cache instance is properly passed through results dict."""
        load_transform = LoadImageFromFile(
            enable_cache=True,
            cache_dir=str(temp_cache_dir),
        )

        results = {"img_path": temp_image}
        results = load_transform(results)

        # Verify cache instance is in results
        assert "_image_cache" in results
        assert results["_image_cache"] is load_transform.cache

    def test_cache_not_passed_when_disabled(self, temp_image):
        """Test that cache is not passed when disabled."""
        load_transform = LoadImageFromFile(enable_cache=False)

        results = {"img_path": temp_image}
        results = load_transform(results)

        # Verify cache instance is NOT in results
        assert "_image_cache" not in results

    def test_resize_saves_to_cache_after_first_load(self, temp_cache_dir, temp_image):
        """Test that resize saves to cache only after first load."""
        load_transform = LoadImageFromFile(
            enable_cache=True,
            cache_dir=str(temp_cache_dir),
        )
        resize_transform = Resize(scale=(224, 224), enable_cache=True)

        # First pass - load and resize
        results = {"img_path": temp_image}
        results = load_transform(results)
        initial_cache_entries = load_transform.cache.get_stats()["total_entries"]

        results = resize_transform(results)

        # After resize, cache should have the resized image
        final_cache_entries = load_transform.cache.get_stats()["total_entries"]
        assert final_cache_entries > initial_cache_entries

    def test_cache_with_keep_ratio_false(self, temp_cache_dir, temp_image):
        """Test cache works with keep_ratio=False."""
        load_transform = LoadImageFromFile(
            enable_cache=True,
            cache_dir=str(temp_cache_dir),
        )
        resize_transform = Resize(scale=(224, 224), keep_ratio=False, enable_cache=True)

        # First pass
        results = {"img_path": temp_image}
        results = load_transform(results)
        results = resize_transform(results)
        assert results["img"].shape[:2] == (224, 224)

        # Second pass - should hit cache (need to provide target size)
        results2 = {
            "img_path": temp_image,
            "_cache_target_size": results["_cache_target_size"],
        }
        results2 = load_transform(results2)
        assert results2["_cache_hit"] is True

    def test_cache_stats_accuracy(self, temp_cache_dir, temp_image):
        """Test that cache statistics are accurate after transform pipeline."""
        load_transform = LoadImageFromFile(
            enable_cache=True,
            cache_dir=str(temp_cache_dir),
            cache_max_size_gb=1.0,
        )
        resize_transform = Resize(scale=(224, 224), enable_cache=True)

        # Run pipeline multiple times
        results = {"img_path": temp_image}
        results = load_transform(results)
        results = resize_transform(results)
        target_size = results["_cache_target_size"]

        # Subsequent runs with target size
        for _ in range(2):
            results = {"img_path": temp_image, "_cache_target_size": target_size}
            results = load_transform(results)
            results = resize_transform(results)

        stats = load_transform.cache.get_stats()
        assert stats["enabled"] is True
        assert stats["total_entries"] == 1  # Same image, same size
        assert stats["avg_access_count"] >= 2  # At least 2 accesses (1 write, 2+ reads)

    def test_pipeline_without_resize(self, temp_cache_dir, temp_image):
        """Test that cache works even without resize transform."""
        load_transform = LoadImageFromFile(
            enable_cache=True,
            cache_dir=str(temp_cache_dir),
        )

        # Load without resize (cache won't save anything since no target size)
        results = {"img_path": temp_image}
        results = load_transform(results)

        assert results is not None
        assert "_cache_hit" in results

    def test_cache_repr_in_transform(self, temp_cache_dir):
        """Test transform repr includes cache status."""
        load_transform = LoadImageFromFile(
            enable_cache=True,
            cache_dir=str(temp_cache_dir),
        )

        repr_str = repr(load_transform)
        assert "cache_enabled=True" in repr_str

    def test_multiple_images_cached(self, temp_cache_dir, tmp_path):
        """Test that multiple different images are cached correctly."""
        # Create multiple test images
        images = []
        for i in range(3):
            img_path = tmp_path / f"test_image_{i}.jpg"
            test_img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
            Image.fromarray(test_img).save(img_path)
            images.append(str(img_path))

        load_transform = LoadImageFromFile(
            enable_cache=True,
            cache_dir=str(temp_cache_dir),
        )
        resize_transform = Resize(scale=(224, 224), enable_cache=True)

        # Process all images and store target sizes
        target_sizes = {}
        for img_path in images:
            results = {"img_path": img_path}
            results = load_transform(results)
            results = resize_transform(results)
            target_sizes[img_path] = results["_cache_target_size"]

        # Verify all are cached
        stats = load_transform.cache.get_stats()
        assert stats["total_entries"] == 3

        # Verify cache hits on second pass (with target sizes)
        for img_path in images:
            results = {
                "img_path": img_path,
                "_cache_target_size": target_sizes[img_path],
            }
            results = load_transform(results)
            assert results["_cache_hit"] is True
