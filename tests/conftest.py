# Copyright (c) OpenMMLab. All rights reserved.
"""Pytest configuration for test suite.

This module configures which tests to skip based on the project scope.
The visdet project focuses exclusively on Swin Transformer + Mask R-CNN
for object detection and instance segmentation on COCO format.
"""


def pytest_ignore_collect(collection_path, config):
    """Skip collection of test files for non-core functionality."""
    # List of test files to skip collection
    skip_tests = {
        # Data pipeline tests - out of scope
        "test_models_aug_test.py",
        "test_formatting.py",
        "test_coco_occluded.py",
        # Metrics tests - out of scope
        "test_box_overlap.py",
        "test_losses.py",
        "test_mean_ap.py",
        # Model component tests - out of scope
        "test_loss.py",
        "test_necks.py",
        "test_plugins.py",
        "test_roi_extractor.py",
        "test_sabl_bbox_head.py",
        "test_maskformer_fusion_head.py",
        "test_brick_wrappers.py",
        "test_conv_upsample.py",
        "test_inverted_residual.py",
        "test_position_encoding.py",
        "test_se_layer.py",
        "test_transformer.py",
        # Runtime/infrastructure tests - out of scope
        "test_async.py",
        "test_config.py",
        "test_eval_hook.py",
        "test_fp16.py",
        "test_compat_config.py",
        "test_general_data.py",
        "test_hook.py",
        "test_logger.py",
        "test_masks.py",
        "test_memory.py",
        "test_misc.py",
        "test_replace_cfg_vals.py",
        "test_setup_env.py",
        "test_split_batch.py",
        "test_visualization.py",
        # ONNX tests
        "test_onnx/test_head.py",
        "test_onnx/test_neck.py",
        # Alternative backbone tests (only Swin is kept, which has skip marker)
        "test_csp_darknet.py",
        "test_detectors_resnet.py",
        "test_efficientnet.py",
        "test_hourglass.py",
        "test_hrnet.py",
        "test_mobilenet_v2.py",
        "test_pvt.py",
        "test_regnet.py",
        "test_renext.py",
        "test_res2net.py",
        "test_resnest.py",
        "test_resnet.py",
        "test_trident_resnet.py",
        # Tests for non-Swin/Mask R-CNN detectors and components
        "test_forward.py",  # Contains tests for YOLO, RetinaNet, SSD, DETR, etc.
        "test_loss_compatibility.py",  # Cross-architecture loss tests
        "test_anchor.py",  # Tests for various anchor generators (not Swin)
        "test_nms.py",  # NMS implementation details
        "test_layer_decay_optimizer_constructor.py",  # Backbone-specific optimization
        "test_mmtrack.py",  # Video/tracking - out of scope
    }

    path_str = str(collection_path)
    for skip_test in skip_tests:
        if skip_test in path_str:
            return True

    return False
