"""
ComfyUI-Grounding: Simplified object detection nodes for ComfyUI
Supports GroundingDINO, MM-GroundingDINO, OWLv2, Florence-2, and YOLO-World
Now includes SAM2 segmentation nodes
"""

from .nodes import (
    GroundingModelLoader,
    GroundingDetector,
    BboxVisualizer,
    DownloadAndLoadSAM2Model,
    Sam2Segmentation,
)

NODE_CLASS_MAPPINGS = {
    "GroundingModelLoader": GroundingModelLoader,
    "GroundingDetector": GroundingDetector,
    "BboxVisualizer": BboxVisualizer,
    "DownloadAndLoadSAM2Model": DownloadAndLoadSAM2Model,
    "Sam2Segmentation": Sam2Segmentation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundingModelLoader": "Grounding Model Loader",
    "GroundingDetector": "Grounding Detector",
    "BboxVisualizer": "Bounding Box Visualizer",
    "DownloadAndLoadSAM2Model": "(Down)Load SAM2Model",
    "Sam2Segmentation": "Sam2 Segmentation",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
