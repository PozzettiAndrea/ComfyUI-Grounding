"""
Unit tests for model caching in ComfyUI-Grounding
"""
import pytest


@pytest.mark.unit
def test_model_cache_can_store_and_retrieve(mock_comfy_environment, reset_model_cache):
    """Test that MODEL_CACHE can store and retrieve models"""
    from nodes.utils.cache import MODEL_CACHE

    # Test storing a value
    test_key = "test_model_key"
    test_value = {"model": "test", "type": "grounding_dino"}

    MODEL_CACHE[test_key] = test_value

    # Test retrieving the value
    assert test_key in MODEL_CACHE
    assert MODEL_CACHE[test_key] == test_value
