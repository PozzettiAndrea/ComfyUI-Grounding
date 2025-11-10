"""
Florence-2 Segmentation model loading
"""
import os
import folder_paths
import comfy.model_management as mm


# Store original get_imports before any patching
from transformers.dynamic_module_utils import get_imports as _original_get_imports


def fixed_get_imports(filename):
    """Patch to remove flash_attn import requirement from Florence-2"""
    if not str(filename).endswith("modeling_florence2.py"):
        return _original_get_imports(filename)

    imports = _original_get_imports(filename)
    try:
        imports.remove("flash_attn")
    except ValueError:
        pass  # flash_attn not in imports

    return imports


def load_florence2_seg(model_name, config):
    """Load Florence-2 for segmentation

    Args:
        model_name: Model display name
        config: Model configuration dict from registry

    Returns:
        Dict containing model, processor, type, and framework
    """
    from transformers import AutoProcessor, AutoModelForCausalLM
    from unittest.mock import patch

    hf_id = config["hf_id"]
    device = mm.get_torch_device()

    # Use ComfyUI standard model directories
    cache_dir = os.path.join(folder_paths.models_dir, "llm")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"📦 Loading Florence-2 Segmentation Model: {model_name}")

    # Processor needs trust_remote_code for custom attributes like image_token
    processor = AutoProcessor.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Load model with patched get_imports (removes flash_attn requirement)
    # Note: Florence-2 uses eager attention only due to _supports_sdpa bug in transformers 4.50+
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
            attn_implementation="eager"
        )

    model.to(device)
    model.eval()

    print(f"✅ Successfully loaded {model_name}")

    return {
        "model": model,
        "processor": processor,
        "type": "florence2_seg",
        "framework": "transformers"
    }
