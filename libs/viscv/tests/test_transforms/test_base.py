"""Tests for viscv.transforms.base module."""

import sys
from pathlib import Path

import pytest

# Add viscv to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from viscv.transforms.base import BaseTransform


class TestBaseTransform:
    """Test BaseTransform abstract class."""

    def test_abstract_class(self):
        """Test that BaseTransform cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseTransform()

    def test_concrete_implementation(self):
        """Test concrete implementation of BaseTransform."""

        class ConcreteTransform(BaseTransform):
            def __init__(self, value=1):
                self.value = value

            def transform(self, results: dict) -> dict | None:
                results["transformed"] = True
                results["value"] = self.value
                return results

        # Should be able to instantiate concrete class
        transform = ConcreteTransform(value=42)
        assert transform.value == 42

        # Test __call__ method
        input_dict = {"data": "test"}
        output = transform(input_dict)

        assert output is not None
        assert output["data"] == "test"
        assert output["transformed"] is True
        assert output["value"] == 42

    def test_none_return(self):
        """Test transform that returns None."""

        class NoneTransform(BaseTransform):
            def transform(self, results: dict) -> dict | None:
                if results.get("skip", False):
                    return None
                return results

        transform = NoneTransform()

        # Normal case
        output = transform({"data": "test"})
        assert output == {"data": "test"}

        # Skip case
        output = transform({"skip": True})
        assert output is None

    def test_transform_pipeline(self):
        """Test chaining multiple transforms."""

        class AddOneTransform(BaseTransform):
            def transform(self, results: dict) -> dict | None:
                results["value"] = results.get("value", 0) + 1
                return results

        class MultiplyTwoTransform(BaseTransform):
            def transform(self, results: dict) -> dict | None:
                results["value"] = results.get("value", 1) * 2
                return results

        # Create pipeline
        transforms = [AddOneTransform(), MultiplyTwoTransform(), AddOneTransform()]

        # Apply transforms
        data = {"value": 5}
        for t in transforms:
            data = t(data)
            if data is None:
                break

        # (5 + 1) * 2 + 1 = 13
        assert data["value"] == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
