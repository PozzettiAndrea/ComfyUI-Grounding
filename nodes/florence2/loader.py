"""
Florence-2 bbox detection model loading
"""
import os
import folder_paths
import comfy.model_management as mm


def load_florence2(model_name, config, florence2_attn="eager"):
    """Load Florence-2 for bbox detection

    Args:
        model_name: Model display name
        config: Model configuration dict from registry
        florence2_attn: Attention implementation

    Returns:
        Dict containing model, processor, type, and framework
    """
    from transformers import AutoProcessor, AutoModelForCausalLM

    hf_id = config["hf_id"]
    device = mm.get_torch_device()
    cache_dir = os.path.join(folder_paths.models_dir, "llm")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"📦 Loading Florence-2 Model: {model_name}")
    print(f"Using Florence-2 attention implementation: {florence2_attn}")

    processor = AutoProcessor.from_pretrained(hf_id, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        attn_implementation=florence2_attn
    )

    model.to(device)
    model.eval()

    print(f"✅ Successfully loaded {model_name}")

    return {
        "model": model,
        "processor": processor,
        "type": "florence2",
        "framework": "transformers"
    }
