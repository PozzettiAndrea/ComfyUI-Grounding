"""
SA2VA model loading
"""
import torch
import os
import folder_paths
import comfy.model_management as mm


def load_sa2va(model_name, config, sa2va_dtype="auto"):
    """Load SA2VA model for vision-language segmentation

    Args:
        model_name: Model display name
        config: Model configuration dict from registry
        sa2va_dtype: Model precision (auto, fp16, bf16, fp32)

    Returns:
        Dict containing model, tokenizer, type, and framework
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = config["hf_id"]
    device = mm.get_torch_device()

    # Use ComfyUI standard model directories
    cache_dir = os.path.join(folder_paths.models_dir, "sa2va")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"📦 Loading SA2VA Model: {model_name}")
    print(f"🎨 Using SA2VA dtype: {sa2va_dtype}")

    # Map dtype string to torch dtype
    dtype_map = {
        "auto": "auto",
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    torch_dtype = dtype_map.get(sa2va_dtype, "auto")

    # Force float32 on CPU to avoid dtype mismatches
    # SA2VA with "auto" uses bfloat16 on GPU but this causes issues on CPU
    if torch_dtype == "auto" and not torch.cuda.is_available():
        torch_dtype = torch.float32
        print(f"⚙️  Forcing float32 dtype for CPU compatibility")

    print(f"📥 Loading model from HuggingFace ({hf_id})...")
    print(f"📂 Cache directory: {cache_dir}")
    print(f"⚠️  IMPORTANT: trust_remote_code=True is required for SA2VA")

    # Try loading with flash_attn first for better performance
    try:
        print(f"⚡ Attempting to load with flash_attn for faster inference...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=device,
            cache_dir=cache_dir,
            trust_remote_code=True,  # Required for SA2VA custom code
            attn_implementation="flash_attention_2"  # Use flash_attn if available
        )
        print(f"✅ Flash attention enabled for SA2VA")
    except (ImportError, Exception) as e:
        # Fallback to eager attention (no flash_attn required)
        print(f"⚠️  Flash attention not available ({str(e)[:50]}...)")
        print(f"⚡ Loading SA2VA with eager attention (slower but functional)")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=device,
            cache_dir=cache_dir,
            trust_remote_code=True,  # Required for SA2VA custom code
            attn_implementation="eager"  # Use standard PyTorch attention
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        trust_remote_code=True  # Required for SA2VA custom code
    )

    # Ensure all model parameters are float32 on CPU to avoid dtype mismatches
    if not torch.cuda.is_available() and torch_dtype == torch.float32:
        model = model.float()
        print(f"✅ Converted all model parameters to float32 for CPU")

    model.eval()

    print(f"✅ Successfully loaded {model_name}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "type": "sa2va",
        "framework": "transformers"
    }
