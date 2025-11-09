"""
Real model integration tests for YOLO-World
These tests download and use actual models - marked as slow
"""
import pytest
import torch


@pytest.mark.real_model
def test_yolo_world_v2_s_load(mock_comfy_environment):
    """Test loading real YOLO-World v2 S (small) model"""
    from nodes import GroundingModelLoader

    loader = GroundingModelLoader()

    # Load smallest YOLO-World model
    model_dict = loader.load_model(
        model="YOLO-World: v8s (Small)",
        florence2_attn="eager"
    )[0]

    assert model_dict is not None
    assert "type" in model_dict
    assert model_dict["type"] == "yolo_world"
    assert "model" in model_dict


@pytest.mark.real_model
def test_yolo_world_v2_s_detection(mock_comfy_environment, small_image):
    """Test real detection with YOLO-World v2 S model"""
    from nodes import GroundingModelLoader, GroundingDetector

    # Load model
    loader = GroundingModelLoader()
    model_dict = loader.load_model(
        model="YOLO-World: v8s (Small)",
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
def test_yolo_world_model_caching(mock_comfy_environment, reset_model_cache):
    """Test that YOLO-World models are properly cached"""
    from nodes import GroundingModelLoader
    from nodes.utils.cache import MODEL_CACHE

    loader = GroundingModelLoader()

    # First load
    model1 = loader.load_model(
        model="YOLO-World: v8s (Small)",
        florence2_attn="eager"
    )[0]

    cache_size_after_first = len(MODEL_CACHE)

    # Second load (should use cache)
    model2 = loader.load_model(
        model="YOLO-World: v8s (Small)",
        florence2_attn="eager"
    )[0]

    cache_size_after_second = len(MODEL_CACHE)

    # Cache should have model
    assert cache_size_after_first > 0
    # Cache shouldn't grow on second load
    assert cache_size_after_second == cache_size_after_first
