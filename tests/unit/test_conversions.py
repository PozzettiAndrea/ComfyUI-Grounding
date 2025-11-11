"""
Unit tests for conversion utilities in ComfyUI-Grounding
"""
import pytest
import torch


@pytest.mark.unit
def test_boxes_to_masks_basic(mock_comfy_environment):
    """Test basic boxes_to_masks conversion"""
    import numpy as np
    from nodes.utils.conversions import boxes_to_masks

    # Create test boxes as numpy arrays: [[x1, y1, x2, y2]]
    boxes = [[np.array([10, 10, 50, 50]), np.array([60, 60, 100, 100])]]
    image_shape = (1, 128, 128, 3)

    masks = boxes_to_masks(boxes, image_shape)

    # Check output shape: (num_boxes, height, width)
    assert masks.shape[0] == 2, "Should have 2 masks"
    assert masks.shape[1] == 128, "Mask height should match image height"
    assert masks.shape[2] == 128, "Mask width should match image width"

    # Check dtype
    assert masks.dtype == torch.float32 or masks.dtype == torch.bool
