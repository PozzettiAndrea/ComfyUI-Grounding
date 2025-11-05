"""
ComfyUI-Grounding: Simplified object detection nodes for ComfyUI
Supports GroundingDINO, MM-GroundingDINO, OWLv2, Florence-2, and YOLO-World
Now includes SAM2 segmentation nodes
"""

from .grounding_init import init

# Initialize web extensions for compatibility with older ComfyUI versions
init()

from .nodes import (
    GroundingModelLoader,
    GroundingDetector,
    BboxVisualizer,
    DownloadAndLoadSAM2Model,
    Sam2Segmentation,
    GroundMaskModelLoader,
    GroundMaskDetector,
)

NODE_CLASS_MAPPINGS = {
    "GroundingModelLoader": GroundingModelLoader,
    "GroundingDetector": GroundingDetector,
    "BboxVisualizer": BboxVisualizer,
    "DownloadAndLoadSAM2Model": DownloadAndLoadSAM2Model,
    "Sam2Segmentation": Sam2Segmentation,
    "GroundMaskModelLoader": GroundMaskModelLoader,
    "GroundMaskDetector": GroundMaskDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundingModelLoader": "Grounding Model Loader",
    "GroundingDetector": "Grounding Detector",
    "BboxVisualizer": "Bounding Box Visualizer",
    "DownloadAndLoadSAM2Model": "(Down)Load SAM2Model",
    "Sam2Segmentation": "Sam2 Segmentation",
    "GroundMaskModelLoader": "Ground Mask Model Loader",
    "GroundMaskDetector": "Ground Mask Detector",
}

# Export web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
