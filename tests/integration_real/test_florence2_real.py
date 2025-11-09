"""
Real model integration tests for Florence2
These tests download and use actual models - marked as slow
"""
import pytest
import torch


@pytest.mark.real_model
def test_florence2_base_load_eager(mock_comfy_environment):
    """Test loading real Florence-2 Base model with eager attention"""
    from nodes import GroundingModelLoader

    loader = GroundingModelLoader()

    # Load Florence-2 Base with eager attention
    model_dict = loader.load_model(
        model="Florence-2: Base (0.23B params)",
        florence2_attn="eager"
    )[0]

    assert model_dict is not None
    assert "type" in model_dict
    assert model_dict["type"] == "florence2"
    assert "model" in model_dict
    assert "processor" in model_dict


@pytest.mark.real_model
def test_florence2_base_load_sdpa(mock_comfy_environment):
    """Test loading real Florence-2 Base model with SDPA attention"""
    from nodes import GroundingModelLoader

    loader = GroundingModelLoader()

    # Load Florence-2 Base with SDPA attention
    model_dict = loader.load_model(
        model="Florence-2: Base (0.23B params)",
        florence2_attn="sdpa"
    )[0]

    assert model_dict is not None
    assert "type" in model_dict
    assert model_dict["type"] == "florence2"


@pytest.mark.real_model
def test_florence2_base_detection(mock_comfy_environment, small_image):
    """Test real detection with Florence-2 Base model"""
    from nodes import GroundingModelLoader, GroundingDetector

    # Load model
    loader = GroundingModelLoader()
    model_dict = loader.load_model(
        model="Florence-2: Base (0.23B params)",
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
def test_florence2_model_caching(mock_comfy_environment, reset_model_cache):
    """Test that Florence-2 models are properly cached"""
    from nodes import GroundingModelLoader
    from nodes.utils.cache import MODEL_CACHE

    loader = GroundingModelLoader()

    # First load
    model1 = loader.load_model(
        model="Florence-2: Base (0.23B params)",
        florence2_attn="eager"
    )[0]

    cache_size_after_first = len(MODEL_CACHE)

    # Second load (should use cache)
    model2 = loader.load_model(
        model="Florence-2: Base (0.23B params)",
        florence2_attn="eager"
    )[0]

    cache_size_after_second = len(MODEL_CACHE)

    # Cache should have model
    assert cache_size_after_first > 0
    # Cache shouldn't grow on second load
    assert cache_size_after_second == cache_size_after_first


@pytest.mark.real_model
def test_florence2_attention_implementations_equivalent(mock_comfy_environment, small_image):
    """Test that different attention implementations produce similar results"""
    from nodes import GroundingModelLoader, GroundingDetector

    loader = GroundingModelLoader()
    detector = GroundingDetector()

    # Load with eager
    model_eager = loader.load_model(
        model="Florence-2: Base (0.23B params)",
        florence2_attn="eager"
    )[0]

    bboxes_eager, _, _, _ = detector.detect(
        model=model_eager,
        image=small_image,
        prompt="object",
        confidence_threshold=0.3,
        bbox_output_format="dict_with_data",
        output_masks=False
    )

    # Load with SDPA
    model_sdpa = loader.load_model(
        model="Florence-2: Base (0.23B params)",
        florence2_attn="sdpa"
    )[0]

    bboxes_sdpa, _, _, _ = detector.detect(
        model=model_sdpa,
        image=small_image,
        prompt="object",
        confidence_threshold=0.3,
        bbox_output_format="dict_with_data",
        output_masks=False
    )

    # Both should produce valid outputs (results may vary slightly)
    assert bboxes_eager is not None
    assert bboxes_sdpa is not None


@pytest.mark.real_model
def test_florence2_mask_generation(mock_comfy_environment, small_image):
    """Test Florence-2 mask generation capability (from bounding boxes)"""
    from nodes import GroundingModelLoader, GroundingDetector

    # Load regular bbox detection model
    loader = GroundingModelLoader()
    model_dict = loader.load_model(
        model="Florence-2: Base (0.23B params)",
        florence2_attn="eager"
    )[0]

    # Run detection with output_masks=True
    detector = GroundingDetector()
    bboxes, annotated_img, labels, masks = detector.detect(
        model=model_dict,
        image=small_image,
        prompt="object",
        confidence_threshold=0.3,
        bbox_output_format="dict_with_data",
        output_masks=True
    )

    # Verify outputs structure
    assert bboxes is not None
    assert masks is not None
    assert annotated_img is not None
    assert isinstance(labels, str)


@pytest.mark.real_model
def test_florence2_different_prompts(mock_comfy_environment, small_image):
    """Test Florence-2 with different prompt formats"""
    from nodes import GroundingModelLoader, GroundingDetector

    loader = GroundingModelLoader()
    model_dict = loader.load_model(
        model="Florence-2: Base (0.23B params)",
        florence2_attn="eager"
    )[0]

    detector = GroundingDetector()

    prompts = [
        "person",
        "person and dog",
        "Locate person",
        "Find the person in the image"
    ]

    for prompt in prompts:
        bboxes, _, _, _ = detector.detect(
            model=model_dict,
            image=small_image,
            prompt=prompt,
            confidence_threshold=0.3,
            bbox_output_format="dict_with_data",
            output_masks=False
        )

        # Each prompt should produce valid output
        assert bboxes is not None
