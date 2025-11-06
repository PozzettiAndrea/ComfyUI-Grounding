"""
GroundingDINO model parameters
"""

# GroundingDINO specific loader parameters
# Note: GroundingDINO only supports "eager" attention, so no parameter needed
LOADER_PARAMS = {
    "optional": {}
}

# GroundingDINO specific detector parameters
DETECTOR_PARAMS = {
    "optional": {
        "text_threshold": ("FLOAT", {
            "default": 0.25,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "tooltip": "GroundingDINO ONLY! Text confidence threshold"
        }),
    }
}

# Return types for bbox detector
RETURN_TYPES = ("IMAGE", "BBOX", "STRING", "FLOAT", "MASK")
RETURN_NAMES = ("annotated_image", "bboxes", "labels", "scores", "masks")
