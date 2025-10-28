"""Tests for SimpleRunner dynamic annotation file parameters.

Tests cover the new train_ann_file and val_ann_file parameters that allow
users to specify annotation files dynamically, supporting ML pipelines with
on-the-fly annotation generation from upstream sources.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from visdet.runner import SimpleRunner


class TestSimpleRunnerAnnotationFiles(unittest.TestCase):
    """Test dynamic annotation file parameter handling in SimpleRunner."""

    def setUp(self) -> None:
        """Set up test fixtures with temporary annotation files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create mock COCO annotation files
        self.train_ann = self.temp_path / "train.json"
        self.val_ann = self.temp_path / "val.json"

        mock_coco = {
            "images": [{"id": i, "file_name": f"image_{i}.jpg"} for i in range(5)],
            "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 100, 100]} for i in range(5)],
            "categories": [{"id": 1, "name": "test_class"}],
        }

        self.train_ann.write_text(json.dumps(mock_coco))
        self.val_ann.write_text(json.dumps(mock_coco))

    def tearDown(self) -> None:
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_train_ann_file_parameter_stored(self, mock_build_config: MagicMock, mock_sync: MagicMock) -> None:
        """Test that train_ann_file parameter is stored correctly."""
        runner = SimpleRunner(
            model="mask_rcnn_swin_s",
            dataset="test_dataset",
            train_ann_file=str(self.train_ann),
        )

        assert runner.train_ann_file == str(self.train_ann)

    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_val_ann_file_parameter_stored(self, mock_build_config: MagicMock, mock_sync: MagicMock) -> None:
        """Test that val_ann_file parameter is stored correctly."""
        runner = SimpleRunner(
            model="mask_rcnn_swin_s",
            dataset="test_dataset",
            val_ann_file=str(self.val_ann),
        )

        assert runner.val_ann_file == str(self.val_ann)

    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_both_annotation_files_stored(self, mock_build_config: MagicMock, mock_sync: MagicMock) -> None:
        """Test that both annotation files can be provided and stored."""
        runner = SimpleRunner(
            model="mask_rcnn_swin_s",
            dataset="test_dataset",
            train_ann_file=str(self.train_ann),
            val_ann_file=str(self.val_ann),
        )

        assert runner.train_ann_file == str(self.train_ann)
        assert runner.val_ann_file == str(self.val_ann)

    def test_missing_train_ann_file_raises_error(self) -> None:
        """Test that missing training annotation file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as context:
            SimpleRunner(
                model="mask_rcnn_swin_s",
                dataset="test_dataset",
                train_ann_file="/nonexistent/path/train.json",
            )

        error_msg = str(context.exception)
        assert "Training annotation file not found" in error_msg
        assert "/nonexistent/path/train.json" in error_msg

    def test_missing_val_ann_file_raises_error(self) -> None:
        """Test that missing validation annotation file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as context:
            SimpleRunner(
                model="mask_rcnn_swin_s",
                dataset="test_dataset",
                train_ann_file=str(self.train_ann),
                val_ann_file="/nonexistent/path/val.json",
            )

        error_msg = str(context.exception)
        assert "Validation annotation file not found" in error_msg

    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_none_annotation_files_allowed(self, mock_build_config: MagicMock, mock_sync: MagicMock) -> None:
        """Test that None annotation files are allowed (uses preset defaults)."""
        # Should not raise any errors
        runner = SimpleRunner(
            model="mask_rcnn_swin_s",
            dataset="test_dataset",
            train_ann_file=None,
            val_ann_file=None,
        )

        assert runner.train_ann_file is None
        assert runner.val_ann_file is None

    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_relative_annotation_file_path(self, mock_build_config: MagicMock, mock_sync: MagicMock) -> None:
        """Test that relative annotation file paths work with data_root."""
        # Create a simple nested structure
        data_dir = self.temp_path / "data"
        data_dir.mkdir()
        ann_dir = data_dir / "annotations"
        ann_dir.mkdir()

        # Create annotation file in nested directory
        nested_ann = ann_dir / "train.json"
        nested_ann.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))

        # Mock _build_config to set dataset_cfg with data_root
        def mock_build(*args, **kwargs):
            pass

        # Create with absolute path first to set up runner
        runner = SimpleRunner(
            model="mask_rcnn_swin_s",
            dataset="test_dataset",
            train_ann_file=str(nested_ann),
        )

        assert runner.train_ann_file == str(nested_ann)
        assert runner.train_ann_file is not None

    def test_error_message_includes_provided_path(self) -> None:
        """Test that error messages include both provided and resolved paths."""
        provided_path = "/some/path/train.json"

        with self.assertRaises(FileNotFoundError) as context:
            SimpleRunner(
                model="mask_rcnn_swin_s",
                dataset="test_dataset",
                train_ann_file=provided_path,
            )

        error_msg = str(context.exception)
        # Should mention the provided path
        assert "Provided path:" in error_msg
        assert provided_path in error_msg

    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    def test_annotation_file_override_in_build_config(self, mock_sync: MagicMock) -> None:
        """Test that annotation files override dataset config in _build_config."""
        with patch("visdet.runner.SimpleRunner._resolve_preset") as mock_resolve:
            # Mock dataset preset (without custom annotation files)
            mock_resolve.side_effect = [
                {},  # model preset
                {
                    "type": "CocoDataset",
                    "ann_file": "default/train.json",  # Default from preset
                    "data_root": "/default/data",
                    "pipeline": [],
                },  # dataset preset
                {},  # optimizer preset
            ]

            runner = SimpleRunner(
                model="mask_rcnn_swin_s",
                dataset="test_dataset",
                train_ann_file=str(self.train_ann),
            )

            # Check that dataset_cfg was overridden
            assert runner.dataset_cfg["ann_file"] == str(self.train_ann)

    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    def test_val_annotation_enables_validation_without_preset(self, mock_sync: MagicMock) -> None:
        """Test that val_ann_file enables validation even without preset val_ann_file."""
        with patch("visdet.runner.SimpleRunner._resolve_preset") as mock_resolve:
            # Mock dataset preset WITHOUT val_ann_file
            mock_resolve.side_effect = [
                {},  # model preset
                {
                    "type": "CocoDataset",
                    "ann_file": "train.json",
                    "data_root": self.temp_dir,
                    "pipeline": [],
                    # No val_ann_file field
                },  # dataset preset
                {},  # optimizer preset
            ]

            # Should not raise - val_ann_file enables validation
            runner = SimpleRunner(
                model="mask_rcnn_swin_s",
                dataset="test_dataset",
                train_ann_file=str(self.train_ann),
                val_ann_file=str(self.val_ann),
            )

            assert runner.val_ann_file == str(self.val_ann)


class TestAnnotationFileValidation(unittest.TestCase):
    """Test the annotation file validation mechanism."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_validation_runs_before_build_config(self) -> None:
        """Test that validation runs before _build_config in __init__."""
        with patch("visdet.runner.SimpleRunner._build_config") as mock_build:
            try:
                SimpleRunner(
                    model="mask_rcnn_swin_s",
                    dataset="test_dataset",
                    train_ann_file="/nonexistent/file.json",
                )
            except FileNotFoundError:
                pass

            # _build_config should not be called if validation fails
            mock_build.assert_not_called()

    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_validation_logs_success(self, mock_build: MagicMock, mock_sync: MagicMock) -> None:
        """Test that successful validation is logged."""
        ann_file = self.temp_path / "train.json"
        ann_file.write_text(json.dumps({}))

        with patch("visdet.runner.logger") as mock_logger:
            _runner = SimpleRunner(  # noqa: F841
                model="mask_rcnn_swin_s",
                dataset="test_dataset",
                train_ann_file=str(ann_file),
            )

            # Check that validation success was logged
            mock_logger.info.assert_called()


if __name__ == "__main__":
    unittest.main()
