"""Tests for viscv.transforms.builder module."""

import sys
from pathlib import Path

import pytest

# Add viscv to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from viscv.transforms.base import BaseTransform
from viscv.transforms.builder import TRANSFORMS, Registry


class TestRegistry:
    """Test Registry class."""

    def test_init(self):
        """Test Registry initialization."""
        registry = Registry("test_registry")
        assert registry._name == "test_registry"
        assert registry._module_dict == {}

    def test_register_module_direct(self):
        """Test direct module registration."""
        registry = Registry("test")

        class DummyTransform(BaseTransform):
            def transform(self, results):
                return results

        # Register with explicit name
        registry.register_module(name="Dummy", module=DummyTransform)
        assert registry.get("Dummy") == DummyTransform

        # Register without name (uses class name)
        class AnotherTransform(BaseTransform):
            def transform(self, results):
                return results

        registry.register_module(module=AnotherTransform)
        assert registry.get("AnotherTransform") == AnotherTransform

    def test_register_module_decorator(self):
        """Test decorator-based registration."""
        registry = Registry("test")

        @registry.register_module()
        class DecoratedTransform(BaseTransform):
            def transform(self, results):
                return results

        assert registry.get("DecoratedTransform") == DecoratedTransform

        # Test with custom name
        @registry.register_module(name="CustomName")
        class AnotherDecoratedTransform(BaseTransform):
            def transform(self, results):
                return results

        assert registry.get("CustomName") == AnotherDecoratedTransform
        assert registry.get("AnotherDecoratedTransform") is None

    def test_register_duplicate(self):
        """Test registering duplicate names."""
        registry = Registry("test")

        class Transform1(BaseTransform):
            def transform(self, results):
                return results

        class Transform2(BaseTransform):
            def transform(self, results):
                return results

        registry.register_module(name="Transform", module=Transform1)

        # Should raise error without force
        with pytest.raises(KeyError, match="Transform is already registered"):
            registry.register_module(name="Transform", module=Transform2)

        # Should work with force=True
        registry.register_module(name="Transform", module=Transform2, force=True)
        assert registry.get("Transform") == Transform2

    def test_build(self):
        """Test building modules from config."""
        registry = Registry("test")

        class ConfigurableTransform(BaseTransform):
            def __init__(self, param1=1, param2="default"):
                self.param1 = param1
                self.param2 = param2

            def transform(self, results):
                return results

        registry.register_module(module=ConfigurableTransform)

        # Test basic build
        config = {"type": "ConfigurableTransform"}
        instance = registry.build(config)
        assert isinstance(instance, ConfigurableTransform)
        assert instance.param1 == 1
        assert instance.param2 == "default"

        # Test build with parameters
        config = {"type": "ConfigurableTransform", "param1": 42, "param2": "custom"}
        instance = registry.build(config)
        assert instance.param1 == 42
        assert instance.param2 == "custom"

    def test_build_errors(self):
        """Test error cases in build."""
        registry = Registry("test")

        # Test non-dict config
        with pytest.raises(TypeError, match="cfg must be a dict"):
            registry.build("not a dict")

        # Test missing type
        with pytest.raises(KeyError, match='cfg must contain the key "type"'):
            registry.build({})

        # Test unregistered type
        with pytest.raises(KeyError, match="UnknownType is not in the test registry"):
            registry.build({"type": "UnknownType"})

    def test_transforms_registry(self):
        """Test the global TRANSFORMS registry."""
        assert isinstance(TRANSFORMS, Registry)
        assert TRANSFORMS._name == "transforms"

        # Check that LoadImageFromFile is registered
        from viscv.transforms.loading import LoadImageFromFile

        assert TRANSFORMS.get("LoadImageFromFile") == LoadImageFromFile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
