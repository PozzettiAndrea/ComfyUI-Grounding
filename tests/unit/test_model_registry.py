"""
Unit tests for model registry in ComfyUI-Grounding
"""
import pytest


@pytest.mark.unit
def test_model_registry_count(mock_comfy_environment):
    """Test that we have exactly 19 models registered"""
    from nodes.config.bbox_models import MODEL_REGISTRY

    assert len(MODEL_REGISTRY) == 19, f"Expected 19 models, found {len(MODEL_REGISTRY)}"
