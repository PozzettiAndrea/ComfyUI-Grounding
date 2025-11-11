"""
SAM2 model parameters
"""

# SAM2 loader parameters
LOADER_PARAMS = {
    "required": {
        "model": ([
            'sam2_hiera_base_plus.safetensors',
            'sam2_hiera_large.safetensors',
            'sam2_hiera_small.safetensors',
            'sam2_hiera_tiny.safetensors',
            'sam2.1_hiera_base_plus.safetensors',
            'sam2.1_hiera_large.safetensors',
            'sam2.1_hiera_small.safetensors',
            'sam2.1_hiera_tiny.safetensors',
        ],),
        "segmentor": (['single_image', 'video', 'automaskgenerator'],),
        "device": (['cuda', 'cpu', 'mps'],),
        "precision": (['fp16', 'bf16', 'fp32'], {"default": 'fp16'}),
    }
}

# SAM2 segmentation parameters
SEGMENTATION_PARAMS = {
    "required": {
        "sam2_model": ("SAM2MODEL",),
        "image": ("IMAGE",),
        "keep_model_loaded": ("BOOLEAN", {"default": False}),
    },
    "optional": {
        "coordinates_positive": ("STRING", {"forceInput": True}),
        "coordinates_negative": ("STRING", {"forceInput": True}),
        "bboxes": ("BBOX",),
        "individual_objects": ("BOOLEAN", {"default": False}),
        "mask": ("MASK",),
        "mask_threshold": ("FLOAT", {
            "default": 0.0,
            "min": -10.0,
            "max": 10.0,
            "step": 0.1,
            "tooltip": "Threshold for converting mask logits to binary. Positive=stricter, negative=looser"
        }),
        "max_hole_area": ("FLOAT", {
            "default": 0.0,
            "min": 0.0,
            "max": 100.0,
            "step": 1.0,
            "tooltip": "Fill holes smaller than this area (0=disabled)"
        }),
        "max_sprinkle_area": ("FLOAT", {
            "default": 0.0,
            "min": 0.0,
            "max": 100.0,
            "step": 1.0,
            "tooltip": "Remove isolated regions smaller than this area (0=disabled)"
        }),
    },
}

# Return types
LOADER_RETURN_TYPES = ("SAM2MODEL",)
LOADER_RETURN_NAMES = ("sam2_model",)

SEGMENTATION_RETURN_TYPES = ("MASK",)
SEGMENTATION_RETURN_NAMES = ("mask",)
