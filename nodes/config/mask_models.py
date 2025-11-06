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
    # LISA Reasoning Segmentation models
    "LISA: 7B-v1": {
        "type": "lisa",
        "hf_id": "xinlai/LISA-7B-v1",
        "framework": "custom",
    },
    "LISA: 13B-llama2-v1": {
        "type": "lisa",
        "hf_id": "xinlai/LISA-13B-llama2-v1",
        "framework": "custom",
    },
    # PSALM Multi-Task Segmentation
    "PSALM: 2B": {
        "type": "psalm",
        "hf_id": "EnmingZhang/PSALM",
        "framework": "transformers",
    },
}
