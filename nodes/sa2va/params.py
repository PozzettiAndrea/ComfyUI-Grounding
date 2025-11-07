"""
SA2VA model parameters
"""

# SA2VA-specific loader parameters
LOADER_PARAMS = {
    "optional": {
        "sa2va_dtype": (["auto", "fp16", "bf16", "fp32"], {
            "default": "auto",
            "tooltip": "🎨 SA2VA ONLY! Model precision: auto=automatic selection, fp16=half precision, bf16=bfloat16, fp32=full precision"
        }),
    }
}

# SA2VA-specific detector parameters
DETECTOR_PARAMS = {
    "optional": {
        "sa2va_max_tokens": ("INT", {
            "default": 2048,
            "min": 512,
            "max": 4096,
            "step": 1,
            "tooltip": "🎨 SA2VA ONLY! Max tokens for generation. Typical: 1024 (fast), 2048 (balanced/default), 4096 (complex)"
        }),
        "sa2va_num_beams": ("INT", {
            "default": 1,
            "min": 1,
            "max": 10,
            "step": 1,
            "tooltip": "🎨 SA2VA ONLY! Beam search width. 1 (greedy/fastest/default), 3-5 (better quality, slower)"
        }),
    }
}

# Return types for SA2VA detector
RETURN_TYPES = ("MASK", "IMAGE", "STRING")
RETURN_NAMES = ("masks", "overlaid_mask", "text")
