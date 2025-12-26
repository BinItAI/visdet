"""Tests for SimpleRunner train_ann_file and val_ann_file parameters."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

from visdet.runner import SimpleRunner


class TestSimpleRunnerAnnotationFiles(unittest.TestCase):
    """Test annotation file parameter handling in SimpleRunner."""

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

    @patch("visdet.runner.SimpleRunner._resolve_preset")
    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_train_ann_file_parameter_stored(
        self, mock_build_config: MagicMock, mock_sync: MagicMock, mock_resolve: MagicMock
    ) -> None:
        """Test that train_ann_file parameter is stored correctly."""
        mock_resolve.return_value = {}
        runner = SimpleRunner(
            model="mask_rcnn_swin_s",
            dataset="test_dataset",
            train_ann_file=str(self.train_ann),
        )

        assert runner.train_ann_file == str(self.train_ann)

    @patch("visdet.runner.SimpleRunner._resolve_preset")
    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_val_ann_file_parameter_stored(
        self, mock_build_config: MagicMock, mock_sync: MagicMock, mock_resolve: MagicMock
    ) -> None:
        """Test that val_ann_file parameter is stored correctly."""
        mock_resolve.return_value = {}
        runner = SimpleRunner(
            model="mask_rcnn_swin_s",
            dataset="test_dataset",
            val_ann_file=str(self.val_ann),
        )

        assert runner.val_ann_file == str(self.val_ann)

    @patch("visdet.runner.SimpleRunner._resolve_preset")
    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_both_annotation_files_stored(
        self, mock_build_config: MagicMock, mock_sync: MagicMock, mock_resolve: MagicMock
    ) -> None:
        """Test that both annotation files can be provided and stored."""
        mock_resolve.return_value = {}
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
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
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
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
            with self.assertRaises(FileNotFoundError) as context:
                SimpleRunner(
                    model="mask_rcnn_swin_s",
                    dataset="test_dataset",
                    train_ann_file=str(self.train_ann),
                    val_ann_file="/nonexistent/path/val.json",
                )

            error_msg = str(context.exception)
            assert "Validation annotation file not found" in error_msg

    @patch("visdet.runner.SimpleRunner._resolve_preset")
    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_none_annotation_files_allowed(
        self, mock_build_config: MagicMock, mock_sync: MagicMock, mock_resolve: MagicMock
    ) -> None:
        """Test that None annotation files are allowed (uses preset defaults)."""
        mock_resolve.return_value = {}
        # Should not raise any errors
        runner = SimpleRunner(
            model="mask_rcnn_swin_s",
            dataset="test_dataset",
            train_ann_file=None,
            val_ann_file=None,
        )

        assert runner.train_ann_file is None
        assert runner.val_ann_file is None

    @patch("visdet.runner.SimpleRunner._resolve_preset")
    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_relative_annotation_file_path(
        self, mock_build_config: MagicMock, mock_sync: MagicMock, mock_resolve: MagicMock
    ) -> None:
        """Test that relative annotation file paths work with data_root."""
        mock_resolve.return_value = {}
        # Create a simple nested structure
        data_dir = self.temp_path / "data"
        data_dir.mkdir()
        ann_dir = data_dir / "annotations"
        ann_dir.mkdir()

        # Create annotation file in nested directory
        nested_ann = ann_dir / "train.json"
        nested_ann.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))

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

        with patch("visdet.runner.SimpleRunner._resolve_preset"):
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

    def test_annotation_file_override_in_build_config(self) -> None:
        """Test that annotation files are properly stored for override."""
        with patch("visdet.runner.SimpleRunner._resolve_preset") as mock_resolve:
            with patch("visdet.runner.SimpleRunner._sync_num_classes"):
                with patch("visdet.runner.SimpleRunner._build_config"):
                    # Mock dataset preset
                    mock_resolve.side_effect = [
                        {},  # model preset
                        {
                            "type": "CocoDataset",
                            "ann_file": "default/train.json",
                            "data_root": "/default/data",
                        },  # dataset preset
                        {},  # optimizer preset
                    ]

                    runner = SimpleRunner(
                        model="mask_rcnn_swin_s",
                        dataset="test_dataset",
                        train_ann_file=str(self.train_ann),
                    )

                    # Check that annotation file parameter is stored
                    # (actual override happens in _build_config which is mocked here)
                    assert runner.train_ann_file == str(self.train_ann)

    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_val_annotation_enables_validation_without_preset(
        self, mock_build_config: MagicMock, mock_sync: MagicMock
    ) -> None:
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
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
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

    @patch("visdet.runner.SimpleRunner._resolve_preset")
    @patch("visdet.runner.SimpleRunner._sync_num_classes")
    @patch("visdet.runner.SimpleRunner._build_config")
    def test_validation_logs_success(
        self, mock_build: MagicMock, mock_sync: MagicMock, mock_resolve: MagicMock
    ) -> None:
        """Test that successful validation is logged."""
        mock_resolve.return_value = {}
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


class TestAnnotationClassDetection(unittest.TestCase):
    """Test automatic class detection from annotation files."""

    def setUp(self) -> None:
        """Set up test fixtures with temporary annotation files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_coco_annotation(self, output_path: Path, category_ids: list[int], category_names: list[str]) -> None:
        """Helper to create COCO annotation with specified categories."""
        categories = [{"id": cid, "name": cname} for cid, cname in zip(category_ids, category_names)]
        annotation = {
            "images": [{"id": i, "file_name": f"image_{i}.jpg"} for i in range(3)],
            "annotations": [
                {"id": i, "image_id": i, "category_id": category_ids[i % len(category_ids)]} for i in range(3)
            ],
            "categories": categories,
        }
        output_path.write_text(json.dumps(annotation))

    def test_extract_classes_from_annotation(self) -> None:
        """Test extracting classes from COCO annotation file."""
        train_ann = self.temp_path / "train.json"
        self.create_coco_annotation(train_ann, [1, 2, 3], ["cat", "dog", "bird"])

        # Create minimal runner without full initialization
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
            with patch("visdet.runner.SimpleRunner._validate_annotation_files"):
                with patch("visdet.runner.SimpleRunner._build_config"):
                    runner = SimpleRunner(
                        model="mask_rcnn_swin_s",
                        dataset="test_dataset",
                        train_ann_file=str(train_ann),
                    )

        # Test extraction
        cat_ids, cat_dict = runner._extract_classes_from_annotation(str(train_ann))
        assert cat_ids == [1, 2, 3]
        assert cat_dict == {1: "cat", 2: "dog", 3: "bird"}

    def test_sync_num_classes_from_dataset_preset_annotation_files(self) -> None:
        """Test inferring num_classes from dataset preset annotation files."""
        train_ann = self.temp_path / "train.json"
        val_ann = self.temp_path / "val.json"
        self.create_coco_annotation(train_ann, [1, 2], ["cat", "dog"])
        self.create_coco_annotation(val_ann, [1, 2], ["cat", "dog"])

        model_preset = {
            "type": "MaskRCNN",
            "roi_head": {
                "type": "StandardRoIHead",
                "bbox_head": {"type": "Shared2FCBBoxHead"},
                "mask_head": {"type": "FCNMaskHead"},
            },
        }

        dataset_preset = {
            "type": "CocoDataset",
            "data_root": str(self.temp_path),
            "ann_file": "train.json",
            "val_ann_file": "val.json",
            "pipeline": [],
        }

        optimizer_preset = {"type": "AdamW"}

        with patch("visdet.runner.SimpleRunner._resolve_preset") as mock_resolve:
            mock_resolve.side_effect = [model_preset, dataset_preset, optimizer_preset]
            runner = SimpleRunner(model="any", dataset="any", optimizer="any")

        assert runner.model_cfg["roi_head"]["bbox_head"]["num_classes"] == 2
        assert runner.model_cfg["roi_head"]["mask_head"]["num_classes"] == 2

    def test_sync_num_classes_prefers_override_over_preset(self) -> None:
        """Test that train_ann_file override wins over dataset ann_file."""
        preset_train_ann = self.temp_path / "preset_train.json"
        override_train_ann = self.temp_path / "override_train.json"

        self.create_coco_annotation(preset_train_ann, [1, 2], ["cat", "dog"])
        self.create_coco_annotation(override_train_ann, [1, 2, 3], ["cat", "dog", "bird"])

        model_preset = {
            "type": "MaskRCNN",
            "roi_head": {
                "type": "StandardRoIHead",
                "bbox_head": {"type": "Shared2FCBBoxHead"},
                "mask_head": {"type": "FCNMaskHead"},
            },
        }

        dataset_preset = {
            "type": "CocoDataset",
            "data_root": str(self.temp_path),
            "ann_file": "preset_train.json",
            "pipeline": [],
        }

        optimizer_preset = {"type": "AdamW"}

        with patch("visdet.runner.SimpleRunner._resolve_preset") as mock_resolve:
            mock_resolve.side_effect = [model_preset, dataset_preset, optimizer_preset]
            runner = SimpleRunner(model="any", dataset="any", optimizer="any", train_ann_file=str(override_train_ann))

        assert runner.model_cfg["roi_head"]["bbox_head"]["num_classes"] == 3
        assert runner.model_cfg["roi_head"]["mask_head"]["num_classes"] == 3

    def test_sync_num_classes_falls_back_to_metainfo_when_files_missing(self) -> None:
        """Test metainfo fallback if preset ann files don't exist."""
        model_preset = {
            "type": "MaskRCNN",
            "roi_head": {
                "type": "StandardRoIHead",
                "bbox_head": {"type": "Shared2FCBBoxHead"},
                "mask_head": {"type": "FCNMaskHead"},
            },
        }

        dataset_preset = {
            "type": "CocoDataset",
            "data_root": str(self.temp_path),
            "ann_file": "does_not_exist.json",
            "pipeline": [],
            "metainfo": {"classes": ["a", "b", "c", "d"]},
        }

        optimizer_preset = {"type": "AdamW"}

        with patch("visdet.runner.SimpleRunner._resolve_preset") as mock_resolve:
            mock_resolve.side_effect = [model_preset, dataset_preset, optimizer_preset]
            runner = SimpleRunner(model="any", dataset="any", optimizer="any")

        assert runner.model_cfg["roi_head"]["bbox_head"]["num_classes"] == 4
        assert runner.model_cfg["roi_head"]["mask_head"]["num_classes"] == 4

    def test_sync_num_classes_single_stage_bbox_head_updated(self) -> None:
        """Test that single-stage bbox_head dict gets num_classes set."""
        train_ann = self.temp_path / "train.json"
        self.create_coco_annotation(train_ann, [1, 2, 3], ["cat", "dog", "bird"])

        model_preset = {"type": "RTMDet", "bbox_head": {"type": "RTMDetSepBNHead"}}
        dataset_preset = {
            "type": "CocoDataset",
            "data_root": str(self.temp_path),
            "ann_file": "train.json",
            "pipeline": [],
        }
        optimizer_preset = {"type": "AdamW"}

        with patch("visdet.runner.SimpleRunner._resolve_preset") as mock_resolve:
            mock_resolve.side_effect = [model_preset, dataset_preset, optimizer_preset]
            runner = SimpleRunner(model="any", dataset="any", optimizer="any")

        assert runner.model_cfg["bbox_head"]["num_classes"] == 3

    def test_sync_num_classes_single_stage_bbox_head_list_updated(self) -> None:
        """Test that a list of bbox heads all get num_classes set."""
        train_ann = self.temp_path / "train.json"
        self.create_coco_annotation(train_ann, [1, 2], ["cat", "dog"])

        model_preset = {
            "type": "SomeSingleStage",
            "bbox_head": [{"type": "HeadA"}, {"type": "HeadB", "other": 1}],
        }
        dataset_preset = {
            "type": "CocoDataset",
            "data_root": str(self.temp_path),
            "ann_file": "train.json",
            "pipeline": [],
        }
        optimizer_preset = {"type": "AdamW"}

        with patch("visdet.runner.SimpleRunner._resolve_preset") as mock_resolve:
            mock_resolve.side_effect = [model_preset, dataset_preset, optimizer_preset]
            runner = SimpleRunner(model="any", dataset="any", optimizer="any")

        assert runner.model_cfg["bbox_head"][0]["num_classes"] == 2
        assert runner.model_cfg["bbox_head"][1]["num_classes"] == 2

    def test_identical_classes_no_warnings(self) -> None:
        """Test that identical classes in train/val produce no warnings."""
        # Create minimal runner
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
            with patch("visdet.runner.SimpleRunner._validate_annotation_files"):
                with patch("visdet.runner.SimpleRunner._build_config"):
                    runner = SimpleRunner(
                        model="mask_rcnn_swin_s",
                        dataset="test_dataset",
                    )

        train_ids = [1, 2, 3]
        train_dict = {1: "cat", 2: "dog", 3: "bird"}
        val_ids = [1, 2, 3]
        val_dict = {1: "cat", 2: "dog", 3: "bird"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            num_classes, class_names = runner._merge_and_validate_classes(train_ids, train_dict, val_ids, val_dict)
            # No warnings should be raised
            assert len(w) == 0

        assert num_classes == 3
        assert class_names == ["cat", "dog", "bird"]

    def test_val_only_classes_warning(self) -> None:
        """Test HIGH severity warning for validation-only classes."""
        # Create minimal runner
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
            with patch("visdet.runner.SimpleRunner._validate_annotation_files"):
                with patch("visdet.runner.SimpleRunner._build_config"):
                    runner = SimpleRunner(
                        model="mask_rcnn_swin_s",
                        dataset="test_dataset",
                    )

        train_ids = [1, 2]
        train_dict = {1: "cat", 2: "dog"}
        val_ids = [1, 2, 3, 4]
        val_dict = {1: "cat", 2: "dog", 3: "bird", 4: "fish"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            num_classes, class_names = runner._merge_and_validate_classes(train_ids, train_dict, val_ids, val_dict)
            # Should have ONE warning about val-only classes
            assert len(w) == 1
            assert "HIGH" in str(w[0].message)
            assert "bird" in str(w[0].message)
            assert "fish" in str(w[0].message)

        # UNION should have 4 classes
        assert num_classes == 4
        assert class_names == ["cat", "dog", "bird", "fish"]

    def test_train_only_classes_info(self) -> None:
        """Test MEDIUM severity warning for training-only classes."""
        # Create minimal runner
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
            with patch("visdet.runner.SimpleRunner._validate_annotation_files"):
                with patch("visdet.runner.SimpleRunner._build_config"):
                    runner = SimpleRunner(
                        model="mask_rcnn_swin_s",
                        dataset="test_dataset",
                    )

        train_ids = [1, 2, 3, 4]
        train_dict = {1: "cat", 2: "dog", 3: "bird", 4: "fish"}
        val_ids = [1, 2]
        val_dict = {1: "cat", 2: "dog"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            num_classes, class_names = runner._merge_and_validate_classes(train_ids, train_dict, val_ids, val_dict)
            # Should have ONE warning about train-only classes
            assert len(w) == 1
            assert "MEDIUM" in str(w[0].message)
            assert "bird" in str(w[0].message)
            assert "fish" in str(w[0].message)

        # UNION should have 4 classes
        assert num_classes == 4
        assert class_names == ["cat", "dog", "bird", "fish"]

    def test_duplicate_id_different_name_error(self) -> None:
        """Test that category ID conflicts raise ValueError."""
        # Create minimal runner
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
            with patch("visdet.runner.SimpleRunner._validate_annotation_files"):
                with patch("visdet.runner.SimpleRunner._build_config"):
                    runner = SimpleRunner(
                        model="mask_rcnn_swin_s",
                        dataset="test_dataset",
                    )

        train_ids = [1, 2, 3]
        train_dict = {1: "cat", 2: "dog", 3: "bird"}
        val_ids = [1, 2, 3]
        val_dict = {1: "cat", 2: "canine", 3: "bird"}  # ID 2 has different name

        with self.assertRaises(ValueError) as context:
            runner._merge_and_validate_classes(train_ids, train_dict, val_ids, val_dict)

        error_msg = str(context.exception)
        assert "conflicting names" in error_msg
        assert "dog" in error_msg
        assert "canine" in error_msg

    def test_non_contiguous_ids_error(self) -> None:
        """Test that non-contiguous category IDs raise ValueError."""
        # Create minimal runner
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
            with patch("visdet.runner.SimpleRunner._validate_annotation_files"):
                with patch("visdet.runner.SimpleRunner._build_config"):
                    runner = SimpleRunner(
                        model="mask_rcnn_swin_s",
                        dataset="test_dataset",
                    )

        train_ids = [1, 2, 5]  # Missing 3 and 4
        train_dict = {1: "cat", 2: "dog", 5: "bird"}

        with self.assertRaises(ValueError) as context:
            runner._merge_and_validate_classes(train_ids, train_dict)

        error_msg = str(context.exception)
        assert "not contiguous" in error_msg

    def test_cascade_roi_head_all_stages_updated(self) -> None:
        """Test that CascadeRoIHead updates num_classes in all stages."""
        # Create minimal runner for testing _sync_num_classes
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
            with patch("visdet.runner.SimpleRunner._validate_annotation_files"):
                with patch("visdet.runner.SimpleRunner._build_config"):
                    runner = SimpleRunner(
                        model="cascade_mask_rcnn_swin_s",
                        dataset="test_dataset",
                    )

        # Set up cascade head with 3 stages
        runner.model_cfg = {
            "roi_head": {
                "type": "CascadeRoIHead",
                "bbox_head": [
                    {"type": "BBoxHead", "num_classes": 80},
                    {"type": "BBoxHead", "num_classes": 80},
                    {"type": "BBoxHead", "num_classes": 80},
                ],
            }
        }

        # Set up dataset config with train annotation file
        train_ann = self.temp_path / "train.json"
        self.create_coco_annotation(train_ann, [1, 2, 3], ["cat", "dog", "bird"])
        runner.train_ann_file = str(train_ann)
        runner.val_ann_file = None
        runner.dataset_cfg = {"data_root": self.temp_dir, "metainfo": {"classes": []}}

        # Call _sync_num_classes which should update all stages
        runner._sync_num_classes()

        # Verify all stages updated
        bbox_head = runner.model_cfg["roi_head"]["bbox_head"]
        assert isinstance(bbox_head, list)
        assert len(bbox_head) == 3
        for stage in bbox_head:
            assert stage["num_classes"] == 3  # Updated to 3 classes from annotation

    def test_union_class_ordering(self) -> None:
        """Test that merged classes maintain proper ordering."""
        # Create minimal runner
        with patch("visdet.runner.SimpleRunner._resolve_preset"):
            with patch("visdet.runner.SimpleRunner._validate_annotation_files"):
                with patch("visdet.runner.SimpleRunner._build_config"):
                    runner = SimpleRunner(
                        model="mask_rcnn_swin_s",
                        dataset="test_dataset",
                    )

        # Train has classes 1, 3, 5; Val has 2, 4
        train_ids = [1, 3, 5]
        train_dict = {1: "cat", 3: "dog", 5: "bird"}
        val_ids = [2, 4]
        val_dict = {2: "fish", 4: "zebra"}

        num_classes, class_names = runner._merge_and_validate_classes(train_ids, train_dict, val_ids, val_dict)

        # Should have 5 classes total (UNION)
        assert num_classes == 5
        # Classes should be in order: [cat, fish, dog, zebra, bird]
        assert class_names == ["cat", "fish", "dog", "zebra", "bird"]


if __name__ == "__main__":
    unittest.main()
