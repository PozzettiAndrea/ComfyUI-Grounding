"""
YOLO-World model parameters
"""

# YOLO-World specific loader parameters (none currently)
LOADER_PARAMS = {
    "optional": {}
}

# YOLO-World specific detector parameters
DETECTOR_PARAMS = {
    "optional": {
        "yolo_iou": ("FLOAT", {
            "default": 0.45,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "tooltip": "🌍 YOLO-World ONLY! IoU threshold for NMS. Typical: 0.3-0.4 (keep more overlapping boxes), 0.45 (balanced/default), 0.5-0.7 (aggressive filtering)"
        }),
        "yolo_agnostic_nms": ("BOOLEAN", {
            "default": False,
            "tooltip": "🌍 YOLO-World ONLY! Class-agnostic NMS. Enable when detecting overlapping objects of different classes (e.g., person holding bottle)"
        }),
        "yolo_max_det": ("INT", {
            "default": 300,
            "min": 1,
            "max": 1000,
            "step": 10,
            "tooltip": "🌍 YOLO-World ONLY! Max detections per image. Typical: 100 (sparse), 300 (balanced/default), 500-1000 (dense/crowded scenes)"
        }),
    }
}

# Return types for bbox detector
RETURN_TYPES = ("IMAGE", "BBOX", "STRING", "FLOAT", "MASK")
RETURN_NAMES = ("annotated_image", "bboxes", "labels", "scores", "masks")
