"""
Florence-2 bbox detection model parameters
"""

# Florence-2 bbox specific loader parameters
# Note: SDPA is disabled due to _supports_sdpa bug in transformers 4.50+
LOADER_PARAMS = {
    "optional": {}
}

# Florence-2 bbox specific detector parameters
DETECTOR_PARAMS = {
    "optional": {
        "florence2_max_tokens": ("INT", {
            "default": 1024,
            "min": 1,
            "max": 4096,
            "step": 1,
            "tooltip": "🌸 Florence-2 ONLY! Max tokens for generation"
        }),
        "florence2_num_beams": ("INT", {
            "default": 3,
            "min": 1,
            "max": 10,
            "step": 1,
            "tooltip": "🌸 Florence-2 ONLY! Beam search width"
        }),
    }
}

# Return types for bbox detector
RETURN_TYPES = ("IMAGE", "BBOX", "STRING", "FLOAT", "MASK")
RETURN_NAMES = ("annotated_image", "bboxes", "labels", "scores", "masks")
