"""Tests for viscv.transforms.loading module."""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add viscv to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from viscv.transforms import TRANSFORMS, LoadImageFromFile


class TestLoadImageFromFile:
    """Test LoadImageFromFile transform."""

    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image."""
        # Create a simple 100x100 RGB image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            yield f.name, img

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_init(self):
        """Test LoadImageFromFile initialization."""
        loader = LoadImageFromFile()
        assert loader.to_float32 is False
        assert loader.color_type == "color"
        assert loader.imdecode_backend == "cv2"
        assert loader.ignore_empty is False
        assert loader.file_client_args is None
        assert loader.backend_args is None

        # Test with custom parameters
        loader = LoadImageFromFile(
            to_float32=True,
            color_type="grayscale",
            imdecode_backend="pillow",
            ignore_empty=True,
        )
        assert loader.to_float32 is True
        assert loader.color_type == "grayscale"
        assert loader.imdecode_backend == "pillow"
        assert loader.ignore_empty is True

    def test_transform_basic(self, temp_image):
        """Test basic image loading."""
        img_path, original_img = temp_image

        loader = LoadImageFromFile()
        results = {"img_path": img_path}

        output = loader.transform(results)

        assert output is not None
        assert "img" in output
        assert "img_shape" in output
        assert "ori_shape" in output

        # Check image is loaded correctly
        assert isinstance(output["img"], np.ndarray)
        assert output["img"].shape == original_img.shape
        assert output["img_shape"] == original_img.shape[:2]
        assert output["ori_shape"] == original_img.shape[:2]

        # Check data type
        assert output["img"].dtype == np.uint8

    def test_transform_float32(self, temp_image):
        """Test loading image as float32."""
        img_path, _ = temp_image

        loader = LoadImageFromFile(to_float32=True)
        results = {"img_path": img_path}

        output = loader.transform(results)

        assert output["img"].dtype == np.float32

    def test_transform_grayscale(self, temp_image):
        """Test loading image as grayscale."""
        img_path, original_img = temp_image

        loader = LoadImageFromFile(color_type="grayscale")
        results = {"img_path": img_path}

        output = loader.transform(results)

        # Grayscale image should have 2 dimensions
        assert len(output["img"].shape) == 2
        assert output["img_shape"] == original_img.shape[:2]

    def test_transform_missing_file(self):
        """Test behavior with missing file."""
        loader = LoadImageFromFile(ignore_empty=False)
        results = {"img_path": "/nonexistent/path/image.jpg"}

        with pytest.raises(Exception):
            loader.transform(results)

        # Test with ignore_empty=True
        loader = LoadImageFromFile(ignore_empty=True)
        output = loader.transform(results)
        assert output is None

    def test_registry(self):
        """Test that LoadImageFromFile is registered."""
        assert TRANSFORMS.get("LoadImageFromFile") == LoadImageFromFile

        # Test building from config
        config = {
            "type": "LoadImageFromFile",
            "to_float32": True,
            "color_type": "color",
        }
        loader = TRANSFORMS.build(config)
        assert isinstance(loader, LoadImageFromFile)
        assert loader.to_float32 is True
        assert loader.color_type == "color"

    def test_repr(self):
        """Test string representation."""
        loader = LoadImageFromFile(to_float32=True, ignore_empty=True, backend_args={"backend": "disk"})
        repr_str = repr(loader)
        assert "LoadImageFromFile" in repr_str
        assert "to_float32=True" in repr_str
        assert "ignore_empty=True" in repr_str
        assert "backend_args=" in repr_str

    def test_deprecated_file_client_args(self):
        """Test deprecated file_client_args parameter."""
        with pytest.warns(DeprecationWarning):
            loader = LoadImageFromFile(file_client_args={"backend": "disk"})
        assert loader.file_client_args == {"backend": "disk"}
        assert loader.backend_args is None

        # Test that both args cannot be set
        with pytest.raises(ValueError):
            LoadImageFromFile(file_client_args={"backend": "disk"}, backend_args={"backend": "disk"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
