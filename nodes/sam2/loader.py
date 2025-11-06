"""
SAM2 model loading logic
"""
import os
import torch
import folder_paths
from ..utils.cache import MODEL_CACHE


def load_sam2(model, segmentor, device, precision, script_directory):
    """Load SAM2 model

    Args:
        model: Model filename (e.g., 'sam2_hiera_large.safetensors')
        segmentor: Type of segmentation ('single_image', 'video', 'automaskgenerator')
        device: Device name ('cuda', 'cpu', 'mps')
        precision: Precision ('fp16', 'bf16', 'fp32')
        script_directory: Path to the custom_nodes/ComfyUI-Grounding directory

    Returns:
        Dict containing model, dtype, device, segmentor, version
    """
    from .load_model import load_model

    if precision != 'fp32' and device == 'cpu':
        raise ValueError("fp16 and bf16 are not supported on cpu")

    if device == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    device = {"cuda": torch.device("cuda"), "cpu": torch.device("cpu"), "mps": torch.device("mps")}[device]

    # Create cache key for MODEL_CACHE
    cache_key = f"sam2_{model}_{segmentor}_{precision}_{device}"

    # Check cache
    if cache_key in MODEL_CACHE:
        print(f"✅ Loading SAM2 model from cache")
        return MODEL_CACHE[cache_key]

    download_path = os.path.join(folder_paths.models_dir, "sam2")
    if precision != 'fp32' and "2.1" in model:
        base_name, extension = model.rsplit('.', 1)
        model = f"{base_name}-fp16.{extension}"
    model_path = os.path.join(download_path, model)
    print("model_path: ", model_path)

    if not os.path.exists(model_path):
        print(f"Downloading SAM2 model to: {model_path}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="Kijai/sam2-safetensors",
                        allow_patterns=[f"*{model}*"],
                        local_dir=download_path,
                        local_dir_use_symlinks=False)

    model_mapping = {
        "2.0": {
            "base": "sam2_hiera_b+.yaml",
            "large": "sam2_hiera_l.yaml",
            "small": "sam2_hiera_s.yaml",
            "tiny": "sam2_hiera_t.yaml"
        },
        "2.1": {
            "base": "sam2.1_hiera_b+.yaml",
            "large": "sam2.1_hiera_l.yaml",
            "small": "sam2.1_hiera_s.yaml",
            "tiny": "sam2.1_hiera_t.yaml"
        }
    }
    version = "2.1" if "2.1" in model else "2.0"

    model_cfg_path = next(
        (os.path.join(script_directory, "nodes", "sam2", "configs", cfg)
        for key, cfg in model_mapping[version].items() if key in model),
        None
    )
    print(f"Using model config: {model_cfg_path}")

    model = load_model(model_path, model_cfg_path, segmentor, dtype, device)

    sam2_model = {
        'model': model,
        'dtype': dtype,
        'device': device,
        'segmentor': segmentor,
        'version': version
    }

    # Cache the loaded model
    MODEL_CACHE[cache_key] = sam2_model
    print(f"💾 Cached SAM2 model in memory")

    return sam2_model
