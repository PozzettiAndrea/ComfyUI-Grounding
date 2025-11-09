"""
ComfyUI-Grounding: Simplified object detection nodes for ComfyUI
Supports GroundingDINO, MM-GroundingDINO, OWLv2, Florence-2, and YOLO-World
Now includes SAM2 segmentation nodes
"""

# Only run initialization and imports when loaded by ComfyUI, not during pytest
# This prevents relative import errors when pytest collects test modules
import sys
if 'pytest' not in sys.modules:
    from .grounding_init import init
    # Initialize web extensions for compatibility with older ComfyUI versions
    init()

    from .nodes import (
        GroundingModelLoader,
        GroundingDetector,
        BboxVisualizer,
        DownloadAndLoadSAM2Model,
        Sam2Segmentation,
        GroundingMaskModelLoader,
        GroundingMaskDetector,
    )
else:
    # During testing, set dummy values
    GroundingModelLoader = None
    GroundingDetector = None
    BboxVisualizer = None
    DownloadAndLoadSAM2Model = None
    Sam2Segmentation = None
    GroundingMaskModelLoader = None
    GroundingMaskDetector = None

NODE_CLASS_MAPPINGS = {
    "GroundingModelLoader": GroundingModelLoader,
    "GroundingDetector": GroundingDetector,
    "BboxVisualizer": BboxVisualizer,
    "DownloadAndLoadSAM2Model": DownloadAndLoadSAM2Model,
    "Sam2Segmentation": Sam2Segmentation,
    "GroundingMaskModelLoader": GroundingMaskModelLoader,
    "GroundingMaskDetector": GroundingMaskDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundingModelLoader": "Grounding Model Loader",
    "GroundingDetector": "Grounding Detector",
    "BboxVisualizer": "Bounding Box Visualizer",
    "DownloadAndLoadSAM2Model": "(Down)Load SAM2Model",
    "Sam2Segmentation": "Sam2 Segmentation",
    "GroundingMaskModelLoader": "Grounding Mask Model Loader",
    "GroundingMaskDetector": "Grounding Mask Detector",
}

# Export web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
