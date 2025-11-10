"""
Unit tests for input validation in ComfyUI-Grounding nodes
"""
import pytest


@pytest.mark.unit
def test_grounding_detector_input_types(mock_comfy_environment):
    """Test GroundingDetector INPUT_TYPES"""
    from nodes import GroundingDetector

    input_types = GroundingDetector.INPUT_TYPES()

    # Check required inputs
    required = input_types["required"]
    assert "model" in required
    assert "image" in required
    assert "prompt" in required
    assert "confidence_threshold" in required

    # Check optional inputs
    optional = input_types["optional"]
    assert "single_box_mode" in optional
    assert "bbox_output_format" in optional
    # Note: output_masks is now always enabled (not a UI parameter)
