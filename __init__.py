"""
ComfyUI-Grounding: Object detection nodes for ComfyUI
Supports GroundingDINO, MM-GroundingDINO, and YOLO-World models
"""

from .nodes import (
    GroundingDINOModelLoader,
    GroundingDINODetector,
    YOLOWorldModelLoader,
    YOLOWorldDetector,
    UnifiedDetector,
    BboxVisualizer,
    BboxToMask
)

NODE_CLASS_MAPPINGS = {
    "GroundingDINOModelLoader": GroundingDINOModelLoader,
    "GroundingDINODetector": GroundingDINODetector,
    "YOLOWorldModelLoader": YOLOWorldModelLoader,
    "YOLOWorldDetector": YOLOWorldDetector,
    "UnifiedDetector": UnifiedDetector,
    "BboxVisualizer": BboxVisualizer,
    "BboxToMask": BboxToMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundingDINOModelLoader": "Grounding DINO Model Loader",
    "GroundingDINODetector": "Grounding DINO Detector",
    "YOLOWorldModelLoader": "YOLO-World Model Loader",
    "YOLOWorldDetector": "YOLO-World Detector",
    "UnifiedDetector": "Unified Detector (Auto)",
    "BboxVisualizer": "Bounding Box Visualizer",
    "BboxToMask": "Bounding Box to Mask",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
