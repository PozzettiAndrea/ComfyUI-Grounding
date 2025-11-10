"""
Real model integration tests for GroundingDINO
These tests download and use actual models - marked as slow
"""
import pytest
import torch


@pytest.mark.real_model
def test_grounding_dino_swint_load(mock_comfy_environment):
    """Test loading real GroundingDINO SwinT model"""
    from nodes import GroundingModelLoader

    loader = GroundingModelLoader()

    # Load smallest GroundingDINO model
    model_dict = loader.load_model(
        model="GroundingDINO: SwinT OGC (694MB)"
    )[0]

    assert model_dict is not None
    assert "type" in model_dict
    assert model_dict["type"] == "grounding_dino"
    assert "model" in model_dict
    assert "processor" in model_dict


@pytest.mark.real_model
def test_grounding_dino_swint_detection(mock_comfy_environment, small_image):
    """Test real detection with GroundingDINO SwinT model"""
    from nodes import GroundingModelLoader, GroundingDetector
    from pathlib import Path
    from PIL import Image
    import numpy as np

    # Load model
    loader = GroundingModelLoader()
    model_dict = loader.load_model(
        model="GroundingDINO: SwinT OGC (694MB)"
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
    print(f"\n=== GroundingDINO Detection Results ===")
    print(f"Detected labels: {labels}")
    print(f"Bboxes: {bboxes}")

    # Save annotated image
    output_dir = Path(__file__).parent.parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    # Convert tensor to PIL Image and save
    img_np = (annotated_img[0].cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(output_dir / "grounding_dino_detection.png")
    print(f"Saved annotated image to {output_dir / 'grounding_dino_detection.png'}")

    # Verify outputs structure
    assert bboxes is not None
    assert annotated_img is not None
    assert annotated_img.shape == small_image.shape
    assert isinstance(labels, str)


@pytest.mark.real_model
def test_grounding_dino_model_caching(mock_comfy_environment, reset_model_cache):
    """Test that GroundingDINO models are properly cached"""
    from nodes import GroundingModelLoader
    from nodes.utils.cache import MODEL_CACHE

    loader = GroundingModelLoader()

    # First load
    model1 = loader.load_model(
        model="GroundingDINO: SwinT OGC (694MB)"
    )[0]

    cache_size_after_first = len(MODEL_CACHE)

    # Second load (should use cache)
    model2 = loader.load_model(
        model="GroundingDINO: SwinT OGC (694MB)"
    )[0]

    cache_size_after_second = len(MODEL_CACHE)

    # Cache should have model
    assert cache_size_after_first > 0
    # Cache shouldn't grow on second load
    assert cache_size_after_second == cache_size_after_first
