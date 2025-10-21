# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from viscv.ops import roi_align


class TestRoIAlign:

    def test_roi_align_cpu(self):
        """Test RoIAlign on CPU."""
        # Create a simple 4x4 feature map
        features = torch.tensor([[[[1., 2., 3., 4.],
                                   [5., 6., 7., 8.],
                                   [9., 10., 11., 12.],
                                   [13., 14., 15., 16.]]]], dtype=torch.float32)
        
        # Single RoI covering the top-left quadrant
        rois = torch.tensor([[0., 0., 0., 2., 2.]], dtype=torch.float32)
        
        output_size = (2, 2)
        spatial_scale = 1.0
        sampling_ratio = 2
        
        output = roi_align(
            features, rois, output_size, 
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio
        )
        
        # Check output shape
        assert output.shape == (1, 1, 2, 2)
        
        # Values should be interpolated from top-left quadrant
        assert output.min() >= 1.0
        assert output.max() <= 6.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
    def test_roi_align_cuda(self):
        """Test RoIAlign on CUDA."""
        features = torch.tensor([[[[1., 2., 3., 4.],
                                   [5., 6., 7., 8.],
                                   [9., 10., 11., 12.],
                                   [13., 14., 15., 16.]]]], dtype=torch.float32).cuda()
        
        rois = torch.tensor([[0., 0., 0., 2., 2.]], dtype=torch.float32).cuda()
        
        output_size = (2, 2)
        spatial_scale = 1.0
        sampling_ratio = 2
        
        output = roi_align(
            features, rois, output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio
        )
        
        # Check output is on CUDA
        assert output.is_cuda
        
        # Check output shape
        assert output.shape == (1, 1, 2, 2)

    def test_roi_align_multiple_rois(self):
        """Test RoIAlign with multiple RoIs."""
        features = torch.rand(2, 3, 8, 8)  # 2 images, 3 channels, 8x8
        
        # 3 RoIs: first two for image 0, last one for image 1
        rois = torch.tensor([
            [0., 0., 0., 4., 4.],  # Top-left of image 0
            [0., 4., 4., 8., 8.],  # Bottom-right of image 0
            [1., 2., 2., 6., 6.]   # Center of image 1
        ])
        
        output_size = (3, 3)
        output = roi_align(features, rois, output_size)
        
        # Check output shape: (num_rois, channels, h, w)
        assert output.shape == (3, 3, 3, 3)

    def test_roi_align_with_different_scales(self):
        """Test RoIAlign with different spatial scales."""
        features = torch.rand(1, 2, 16, 16)
        
        # RoI in original image coordinates
        rois = torch.tensor([[0., 8., 8., 24., 24.]])
        
        # spatial_scale = 0.5 means features are half the size of original
        output = roi_align(features, rois, (4, 4), spatial_scale=0.5)
        
        assert output.shape == (1, 2, 4, 4)

    def test_roi_align_empty_rois(self):
        """Test RoIAlign with empty RoIs."""
        features = torch.rand(1, 3, 8, 8)
        rois = torch.empty((0, 5))  # Empty tensor
        
        output = roi_align(features, rois, (4, 4))
        
        # Should return empty output
        assert output.shape == (0, 3, 4, 4)