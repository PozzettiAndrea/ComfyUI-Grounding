"""
Florence-2 Segmentation model parameters
"""

# Florence-2 Seg specific loader parameters
LOADER_PARAMS = {
    "optional": {
        "florence2_attn": (["eager", "sdpa", "flash_attention_2"], {
            "default": "eager",
            "tooltip": "🌸 Florence-2 ONLY! Attention implementation: eager=most compatible, sdpa=PyTorch 2.0+, flash_attention_2=A100/H100"
        }),
    }
}

# Florence-2 Seg specific detector parameters
DETECTOR_PARAMS = {
    "optional": {
        "florence2_max_tokens": ("INT", {
            "default": 1024,
            "min": 1,
            "max": 4096,
            "step": 1,
            "tooltip": "🌸 Florence-2 ONLY! Max tokens for generation. Typical: 512 (fast), 1024 (balanced), 2048+ (complex scenes)"
        }),
        "florence2_num_beams": ("INT", {
            "default": 3,
            "min": 1,
            "max": 10,
            "step": 1,
            "tooltip": "🌸 Florence-2 ONLY! Beam search width. 1 (greedy/fastest), 3 (balanced/default), 5+ (better quality, slower)"
        }),
    }
}

# Return types for mask detector
RETURN_TYPES = ("MASK", "BBOX", "STRING", "IMAGE", "STRING")
RETURN_NAMES = ("masks", "bboxes", "labels", "annotated_image", "text")
