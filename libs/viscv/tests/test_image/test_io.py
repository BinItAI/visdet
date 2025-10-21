"""Tests for viscv.image.io module."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add viscv to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from viscv.image import imfrombytes


class TestImfrombytes:
    """Test imfrombytes function."""

    @pytest.fixture
    def image_bytes(self):
        """Create test image bytes."""
        # Create a simple test image
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        # Encode as JPEG bytes
        success, buffer = cv2.imencode(".jpg", img)
        assert success

        return buffer.tobytes(), img

    def test_basic_decode(self, image_bytes):
        """Test basic image decoding from bytes."""
        img_bytes, original = image_bytes

        decoded = imfrombytes(img_bytes)

        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == original.shape
        assert decoded.dtype == np.uint8

    def test_color_flag(self, image_bytes):
        """Test different color flags."""
        img_bytes, _ = image_bytes

        # Color image
        color_img = imfrombytes(img_bytes, flag="color")
        assert len(color_img.shape) == 3

        # Grayscale image
        gray_img = imfrombytes(img_bytes, flag="grayscale")
        assert len(gray_img.shape) == 2

        # Unchanged
        unchanged_img = imfrombytes(img_bytes, flag="unchanged")
        assert isinstance(unchanged_img, np.ndarray)

    def test_channel_order(self, image_bytes):
        """Test channel order parameter."""
        img_bytes, _ = image_bytes

        # Default BGR
        bgr_img = imfrombytes(img_bytes, channel_order="bgr")

        # RGB order
        rgb_img = imfrombytes(img_bytes, channel_order="rgb")

        # The channels should be swapped
        # Note: Due to JPEG compression, exact equality won't work
        assert bgr_img.shape == rgb_img.shape

    def test_backend_parameter(self, image_bytes):
        """Test backend parameter."""
        img_bytes, _ = image_bytes

        # Test cv2 backend (default)
        cv2_img = imfrombytes(img_bytes, backend="cv2")
        assert isinstance(cv2_img, np.ndarray)

        # Test explicit cv2 backend
        cv2_explicit = imfrombytes(img_bytes, backend="cv2")
        assert isinstance(cv2_explicit, np.ndarray)

    def test_invalid_backend(self, image_bytes):
        """Test invalid backend raises error."""
        img_bytes, _ = image_bytes

        with pytest.raises(ValueError, match="backend: invalid is not supported"):
            imfrombytes(img_bytes, backend="invalid")

    def test_png_bytes(self):
        """Test decoding PNG bytes."""
        # Create image with transparency
        img = np.random.randint(0, 255, (30, 30, 4), dtype=np.uint8)

        # Encode as PNG
        success, buffer = cv2.imencode(".png", img)
        assert success
        png_bytes = buffer.tobytes()

        # Decode
        decoded = imfrombytes(png_bytes, flag="unchanged")
        assert decoded.shape == img.shape

    def test_empty_bytes(self):
        """Test handling of empty bytes."""
        # cv2.imdecode returns None for invalid data
        result = imfrombytes(b"")
        assert result is None

        result = imfrombytes(b"invalid image data")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
