"""
BboxVisualizer parameters
"""

# BboxVisualizer parameters
VISUALIZER_PARAMS = {
    "required": {
        "image": ("IMAGE",),
        "bboxes": ("BBOX",),
    },
    "optional": {
        "line_width": ("INT", {"default": 3, "min": 1, "max": 10}),
    }
}

# Return types
RETURN_TYPES = ("IMAGE",)
RETURN_NAMES = ("annotated_image",)
