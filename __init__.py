"""
ComfyUI-Grounding: Simplified object detection nodes for ComfyUI
Supports GroundingDINO, MM-GroundingDINO, OWLv2, Florence-2, and YOLO-World
Now includes SAM2 segmentation nodes
"""

# Only run initialization and imports when loaded by ComfyUI, not during pytest
# This prevents relative import errors when pytest collects test modules
import sys
import traceback

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

if 'pytest' not in sys.modules:
    print("[ComfyUI-Grounding] Initializing custom node...")

    # Step 1: Run initialization script
    try:
        from .grounding_init import init
        # Initialize web extensions for compatibility with older ComfyUI versions
        init()
        print("[ComfyUI-Grounding] [OK] Initialization script completed")
    except Exception as e:
        error_msg = f"Initialization script failed: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[ComfyUI-Grounding] [WARNING] {error_msg}")
        print(f"[ComfyUI-Grounding] Traceback:\n{traceback.format_exc()}")

    # Step 2: Import node classes
    try:
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
        print("[ComfyUI-Grounding] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[ComfyUI-Grounding] [WARNING] {error_msg}")
        print(f"[ComfyUI-Grounding] Traceback:\n{traceback.format_exc()}")

        # Set all to None if import failed
        GroundingModelLoader = None
        GroundingDetector = None
        BboxVisualizer = None
        DownLoadSAM2Model = None
        Sam2Segment = None
        GroundingMaskModelLoader = None
        GroundingMaskDetector = None
        BatchCropAndPadFromMask = None

    # Report final status
    if INIT_SUCCESS:
        print("[ComfyUI-Grounding] [OK] Loaded successfully!")
    else:
        print(f"[ComfyUI-Grounding] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[ComfyUI-Grounding] Please check the errors above and your installation.")

else:
    # During testing, set dummy values
    print("[ComfyUI-Grounding] Running in pytest mode - skipping initialization")
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
