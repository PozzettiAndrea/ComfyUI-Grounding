"""
Model configuration registries for ComfyUI-Grounding
"""
from .bbox_models import MODEL_REGISTRY
from .mask_models import MASK_MODEL_REGISTRY

__all__ = [
    'MODEL_REGISTRY',
    'MASK_MODEL_REGISTRY',
]
