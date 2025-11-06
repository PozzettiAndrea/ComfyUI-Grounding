"""
OWLv2 model parameters
"""

# OWLv2 specific loader parameters
# Note: OWLv2 only supports "eager" attention, so no parameter needed
LOADER_PARAMS = {
    "optional": {}
}

# OWLv2 specific detector parameters (none - uses common params)
DETECTOR_PARAMS = {
    "optional": {}
}

# Return types for bbox detector
RETURN_TYPES = ("IMAGE", "BBOX", "STRING", "FLOAT", "MASK")
RETURN_NAMES = ("annotated_image", "bboxes", "labels", "scores", "masks")
