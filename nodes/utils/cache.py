"""
Model cache management for ComfyUI-Grounding
"""
import comfy.model_management as mm

# Global model cache for keeping models in memory
MODEL_CACHE = {}


def clear_model_from_cache(cache_key):
    """Remove a model from the cache by its key"""
    if cache_key in MODEL_CACHE:
        del MODEL_CACHE[cache_key]
        return True
    return False


def offload_model(model_dict, offload_device=None):
    """
    Offload model to CPU/offload device to free VRAM.

    Args:
        model_dict: Dictionary containing 'model' key with the PyTorch model
        offload_device: Device to move model to. If None, uses ComfyUI's offload device.
    """
    if offload_device is None:
        offload_device = mm.unet_offload_device()

    model = model_dict.get("model")
    if model is not None:
        try:
            model.to(offload_device)
        except Exception:
            # Some models have nested structure
            if hasattr(model, "model"):
                model.model.to(offload_device)

    # Also handle processor if it has device state
    processor = model_dict.get("processor")
    if processor is not None and hasattr(processor, "to"):
        try:
            processor.to(offload_device)
        except Exception:
            pass

    mm.soft_empty_cache()
