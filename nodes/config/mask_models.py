"""
Native segmentation model registry
"""

# Mask model registry for native segmentation models
MASK_MODEL_REGISTRY = {
    # Florence-2 Segmentation models
    "Florence-2: Base (Segmentation)": {
        "type": "florence2_seg",
        "hf_id": "microsoft/Florence-2-base",
        "framework": "transformers",
    },
    "Florence-2: Large (Segmentation)": {
        "type": "florence2_seg",
        "hf_id": "microsoft/Florence-2-large",
        "framework": "transformers",
    },
    # SA2VA Vision-Language Segmentation models
    "SA2VA: 1B": {
        "type": "sa2va",
        "hf_id": "ByteDance/Sa2VA-1B",
        "framework": "transformers",
    },
    "SA2VA: 4B": {
        "type": "sa2va",
        "hf_id": "ByteDance/Sa2VA-4B",
        "framework": "transformers",
    },
    "SA2VA: 8B": {
        "type": "sa2va",
        "hf_id": "ByteDance/Sa2VA-8B",
        "framework": "transformers",
    },
    "SA2VA: 26B": {
        "type": "sa2va",
        "hf_id": "ByteDance/Sa2VA-26B",
        "framework": "transformers",
    },
}
