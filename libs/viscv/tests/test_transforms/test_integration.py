"""Integration tests for viscv transforms."""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Add viscv to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from viscv.transforms import TRANSFORMS, LoadImageFromFile


def test_end_to_end_image_loading():
    """Test complete image loading workflow."""
    # Create a test image
    test_img = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # RGB pixels
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
            [[128, 128, 128], [0, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, test_img)
        temp_path = f.name

    try:
        # Method 1: Direct instantiation
        print("Testing direct instantiation...")
        loader = LoadImageFromFile(to_float32=False)
        result = loader({"img_path": temp_path})

        assert result is not None
        assert "img" in result
        assert np.array_equal(result["img"], test_img)
        print("✓ Direct instantiation works")

        # Method 2: Using registry
        print("\nTesting registry build...")
        config = {
            "type": "LoadImageFromFile",
            "to_float32": True,
            "color_type": "color",
        }
        loader2 = TRANSFORMS.build(config)
        result2 = loader2({"img_path": temp_path})

        assert result2 is not None
        assert result2["img"].dtype == np.float32
        print("✓ Registry build works")

        # Method 3: Pipeline simulation
        print("\nTesting pipeline simulation...")
        pipeline_config = [
            {"type": "LoadImageFromFile", "to_float32": False},
            # Could add more transforms here
        ]

        data = {"img_path": temp_path}
        for cfg in pipeline_config:
            transform = TRANSFORMS.build(cfg)
            data = transform(data)
            if data is None:
                break

        assert data is not None
        assert "img" in data
        print("✓ Pipeline simulation works")

        print("\n✅ All integration tests passed!")

    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_end_to_end_image_loading()
