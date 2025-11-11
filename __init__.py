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
        DownLoadSAM2Model,
        Sam2Segment,
        GroundingMaskModelLoader,
        GroundingMaskDetector,
        BatchCropAndPadFromMask,
    )
else:
    # During testing, set dummy values
    GroundingModelLoader = None
    GroundingDetector = None
    BboxVisualizer = None
    DownLoadSAM2Model = None
    Sam2Segment = None
    GroundingMaskModelLoader = None
    GroundingMaskDetector = None
    BatchCropAndPadFromMask = None

NODE_CLASS_MAPPINGS = {
    "GroundingModelLoader": GroundingModelLoader,
    "GroundingDetector": GroundingDetector,
    "BboxVisualizer": BboxVisualizer,
    "DownLoadSAM2Model": DownLoadSAM2Model,
    "Sam2Segment": Sam2Segment,
    "GroundingMaskModelLoader": GroundingMaskModelLoader,
    "GroundingMaskDetector": GroundingMaskDetector,
    "BatchCropAndPadFromMask": BatchCropAndPadFromMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundingModelLoader": "Grounding Model (down)Loader",
    "GroundingDetector": "Grounding Detector",
    "BboxVisualizer": "Bounding Box Visualizer",
    "DownLoadSAM2Model": "SAM2Model (down)Loader",
    "Sam2Segment": "Sam2 Segment",
    "GroundingMaskModelLoader": "Grounding Mask Model (down)Loader",
    "GroundingMaskDetector": "Grounding Mask Detector",
    "BatchCropAndPadFromMask": "Batch Crop and Pad From Mask",
}

# Export web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
