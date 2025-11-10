"""
Real model integration tests for SA2VA
These tests download and use actual models - marked as slow
"""
import pytest
import torch


@pytest.mark.real_model
def test_sa2va_1b_load(mock_comfy_environment):
    """Test loading real SA2VA 1B model"""
    from nodes import GroundingMaskModelLoader

    loader = GroundingMaskModelLoader()

    # Load smallest SA2VA model
    model_dict = loader.load_model(
        model="SA2VA: 1B",
        sa2va_dtype="auto"
    )[0]

    assert model_dict is not None
    assert "type" in model_dict
    assert model_dict["type"] == "sa2va"
    assert "model" in model_dict
    assert "tokenizer" in model_dict
    assert "framework" in model_dict
    assert model_dict["framework"] == "transformers"

    print(f"\n=== SA2VA Model Loaded Successfully ===")
    print(f"Model type: {model_dict['type']}")
    print(f"Framework: {model_dict['framework']}")


@pytest.mark.real_model
def test_sa2va_1b_segmentation(mock_comfy_environment, small_image):
    """Test real segmentation with SA2VA 1B model"""
    from nodes import GroundingMaskModelLoader, GroundingMaskDetector
    from pathlib import Path
    from PIL import Image
    import numpy as np

    # Load model
    loader = GroundingMaskModelLoader()
    model_dict = loader.load_model(
        model="SA2VA: 1B",
        sa2va_dtype="auto"
    )[0]

    # Run segmentation with the specified prompt
    detector = GroundingMaskDetector()
    masks, overlaid_mask, text = detector.detect(
        model=model_dict,
        image=small_image,
        prompt="Segment the main object in this image, what is it?",
        confidence_threshold=0.3,
        sa2va_max_tokens=2048,
        sa2va_num_beams=1,
        seed=42
    )

    # Print detection results
    print(f"\n=== SA2VA Segmentation Results ===")
    print(f"Generated text: {text}")
    print(f"Masks shape: {masks.shape}")
    print(f"Number of masks: {masks.shape[0]}")

    # Save overlaid mask image
    output_dir = Path(__file__).parent.parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    # Convert tensor to PIL Image and save
    img_np = (overlaid_mask[0].cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(output_dir / "sa2va_segmentation.png")
    print(f"Saved overlaid mask to {output_dir / 'sa2va_segmentation.png'}")

    # Verify outputs structure
    assert masks is not None
    assert masks.shape[0] > 0  # At least one mask
    assert overlaid_mask is not None
    assert overlaid_mask.shape == small_image.shape
    assert isinstance(text, str)
    assert len(text) > 0  # SA2VA should generate text

    print(f"Test passed: SA2VA generated {masks.shape[0]} mask(s) and text output")
