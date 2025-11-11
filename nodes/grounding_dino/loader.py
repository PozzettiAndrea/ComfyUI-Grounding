"""
GroundingDINO model loading
"""
import os
import folder_paths
import comfy.model_management as mm


def load_grounding_dino(model_name, config):
    """Load GroundingDINO model

    Args:
        model_name: Model display name
        config: Model configuration dict from registry

    Returns:
        Dict containing model, processor, type, and framework
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    hf_id = config["hf_id"]
    device = mm.get_torch_device()
    cache_dir = os.path.join(folder_paths.models_dir, "grounding")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"📦 Loading GroundingDINO Model: {model_name}")
    print(f"⚙️  Using attention implementation: eager (only supported option)")

    processor = AutoProcessor.from_pretrained(hf_id, cache_dir=cache_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        attn_implementation="eager"
    )

    model.to(device)
    model.eval()

    print(f"✅ Successfully loaded {model_name}")

    return {
        "model": model,
        "processor": processor,
        "type": "grounding_dino",
        "framework": "transformers"
    }
