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

    # Load Florence-2 Base
    model_dict = loader.load_model(
        model="Florence-2: Base (0.23B params)"
    )[0]

    assert model_dict is not None
    assert "type" in model_dict
    assert model_dict["type"] == "florence2"
    assert "model" in model_dict
    assert "processor" in model_dict


@pytest.mark.real_model
def test_florence2_base_detection(mock_comfy_environment, small_image):
    """Test real detection with Florence-2 Base model"""
    from nodes import GroundingModelLoader, GroundingDetector
    from pathlib import Path
    from PIL import Image
    import numpy as np

    # Load model
    loader = GroundingModelLoader()
    model_dict = loader.load_model(
        model="Florence-2: Base (0.23B params)"
    )[0]

    # Run detection
    detector = GroundingDetector()
    bboxes, annotated_img, labels, masks = detector.detect(
        model=model_dict,
        image=small_image,
        prompt="plant. watering can.",
        confidence_threshold=0.3,
        bbox_output_format="dict_with_data"
    )

    # Print detection results
    print(f"\n=== Florence-2 Eager Detection Results ===")
    print(f"Detected labels: {labels}")
    print(f"Bboxes: {bboxes}")

    # Save annotated image
    output_dir = Path(__file__).parent.parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    # Convert tensor to PIL Image and save
    img_np = (annotated_img[0].cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(output_dir / "florence2_eager_detection.png")
    print(f"Saved annotated image to {output_dir / 'florence2_eager_detection.png'}")

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
        model="Florence-2: Base (0.23B params)"
    )[0]

    cache_size_after_first = len(MODEL_CACHE)

    # Second load (should use cache)
    model2 = loader.load_model(
        model="Florence-2: Base (0.23B params)"
    )[0]

    cache_size_after_second = len(MODEL_CACHE)

    # Cache should have model
    assert cache_size_after_first > 0
    # Cache shouldn't grow on second load
    assert cache_size_after_second == cache_size_after_first


@pytest.mark.real_model
def test_florence2_mask_generation(mock_comfy_environment, small_image):
    """Test Florence-2 mask generation capability (from bounding boxes)"""
    from nodes import GroundingModelLoader, GroundingDetector

    # Load regular bbox detection model
    loader = GroundingModelLoader()
    model_dict = loader.load_model(
        model="Florence-2: Base (0.23B params)"
    )[0]

    # Run detection with masks
    detector = GroundingDetector()
    bboxes, annotated_img, labels, masks = detector.detect(
        model=model_dict,
        image=small_image,
        prompt="plant. watering can.",
        confidence_threshold=0.3,
        bbox_output_format="dict_with_data"
    )

    print(f"\n=== Florence-2 Mask Generation ===")
    print(f"Labels: {labels}, Masks: {masks is not None}")

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
        model="Florence-2: Base (0.23B params)"
    )[0]

    detector = GroundingDetector()

    prompts = [
        "plant. watering can.",
        "plant",
        "watering can",
        "Locate plant and watering can"
    ]

    for prompt in prompts:
        bboxes, _, labels, _ = detector.detect(
            model=model_dict,
            image=small_image,
            prompt=prompt,
            confidence_threshold=0.3,
            bbox_output_format="dict_with_data"
        )

        print(f"\n=== Florence-2 Prompt Test: '{prompt}' ===")
        print(f"Labels: {labels}")

        # Each prompt should produce valid output
        assert bboxes is not None
