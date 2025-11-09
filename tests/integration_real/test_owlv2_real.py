"""
Real model integration tests for OWLv2
These tests download and use actual models - marked as slow
"""
import pytest
import torch


@pytest.mark.real_model
def test_owlv2_base_load(mock_comfy_environment):
    """Test loading real OWLv2 Base Patch16 model"""
    from nodes import GroundingModelLoader

    loader = GroundingModelLoader()

    # Load smallest OWLv2 model
    model_dict = loader.load_model(
        model="OWLv2: Base Patch16",
        florence2_attn="eager"
    )[0]

    assert model_dict is not None
    assert "type" in model_dict
    assert model_dict["type"] == "owlv2"
    assert "model" in model_dict
    assert "processor" in model_dict


@pytest.mark.real_model
def test_owlv2_base_detection(mock_comfy_environment, small_image):
    """Test real detection with OWLv2 Base model"""
    from nodes import GroundingModelLoader, GroundingDetector

    # Load model
    loader = GroundingModelLoader()
    model_dict = loader.load_model(
        model="OWLv2: Base Patch16",
        florence2_attn="eager"
    )[0]

    # Run detection
    detector = GroundingDetector()
    bboxes, annotated_img, labels, masks = detector.detect(
        model=model_dict,
        image=small_image,
        prompt="person",
        confidence_threshold=0.3,
        bbox_output_format="dict_with_data",
        output_masks=False
    )

    # Verify outputs structure
    assert bboxes is not None
    assert annotated_img is not None
    assert annotated_img.shape == small_image.shape
    assert isinstance(labels, str)


@pytest.mark.real_model
def test_owlv2_model_caching(mock_comfy_environment, reset_model_cache):
    """Test that OWLv2 models are properly cached"""
    from nodes import GroundingModelLoader
    from nodes.utils.cache import MODEL_CACHE

    loader = GroundingModelLoader()

    # First load
    model1 = loader.load_model(
        model="OWLv2: Base Patch16",
        florence2_attn="eager"
    )[0]

    cache_size_after_first = len(MODEL_CACHE)

    # Second load (should use cache)
    model2 = loader.load_model(
        model="OWLv2: Base Patch16",
        florence2_attn="eager"
    )[0]

    cache_size_after_second = len(MODEL_CACHE)

    # Cache should have model
    assert cache_size_after_first > 0
    # Cache shouldn't grow on second load
    assert cache_size_after_second == cache_size_after_first
