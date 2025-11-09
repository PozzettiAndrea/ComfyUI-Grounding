"""
Unit tests for node registration in ComfyUI-Grounding
"""
import pytest


@pytest.mark.unit
def test_node_class_mappings_exists(mock_comfy_environment):
    """Test that NODE_CLASS_MAPPINGS is defined and contains all expected nodes"""
    from nodes import NODE_CLASS_MAPPINGS

    expected_nodes = [
        "GroundingModelLoader",
        "GroundingDetector",
        "BboxVisualizer",
        "DownloadAndLoadSAM2Model",
        "Sam2Segmentation",
        "GroundingMaskModelLoader",
        "GroundingMaskDetector"
    ]

    assert NODE_CLASS_MAPPINGS is not None, "NODE_CLASS_MAPPINGS is not defined"

    for node_name in expected_nodes:
        assert node_name in NODE_CLASS_MAPPINGS, f"{node_name} not found in NODE_CLASS_MAPPINGS"
        assert NODE_CLASS_MAPPINGS[node_name] is not None, f"{node_name} maps to None"
