"""
ComfyUI-Grounding: Unified object detection nodes for ComfyUI
Reorganized architecture with thin wrapper nodes delegating to model implementations
"""

import torch
import numpy as np
import random
import os

# Import configurations
from .config import MODEL_REGISTRY, MASK_MODEL_REGISTRY
from .utils.cache import MODEL_CACHE
from .utils import draw_boxes, boxes_to_masks

# Import model loaders and detectors
from . import grounding_dino, owlv2, florence2, yolo_world, florence2_seg, sa2va
from . import sam2, visualizer

# Get the ComfyUI-Grounding root directory (parent of nodes/)
script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Bbox Detection Nodes
# ============================================================================

class GroundingModelLoader:
    """
    Unified model loader for all grounding models
    Supports: GroundingDINO, MM-GroundingDINO, OWLv2, Florence-2, YOLO-World
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODEL_REGISTRY.keys()), {
                    "default": "Florence-2: Base (0.23B params)",
                }),
            },
            "optional": {
                # GroundingDINO parameters
                **grounding_dino.params.LOADER_PARAMS.get("optional", {}),
                # Florence-2 parameters
                **florence2.params.LOADER_PARAMS.get("optional", {}),
                # YOLO-World parameters (none currently in loader)
                **yolo_world.params.LOADER_PARAMS.get("optional", {}),
            }
        }

    RETURN_TYPES = ("GROUNDING_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "grounding"

    def load_model(self, model, florence2_attn="eager"):
        """Load bbox detection model"""
        config = MODEL_REGISTRY[model]
        model_type = config["type"]

        # Create cache key including attention parameters
        cache_key = f"{model_type}_{model}_f2_attn_{florence2_attn}"
        if cache_key in MODEL_CACHE:
            print(f"✅ Loading {model} from cache (f2_attn={florence2_attn})")
            return (MODEL_CACHE[cache_key],)

        print(f"📂 Loading {model}...")

        # Route to appropriate loader
        if model_type == "grounding_dino":
            model_dict = grounding_dino.load_grounding_dino(model, config)
        elif model_type == "owlv2":
            model_dict = owlv2.load_owlv2(model, config)
        elif model_type == "florence2":
            model_dict = florence2.load_florence2(model, config, florence2_attn)
        elif model_type == "yolo_world":
            model_dict = yolo_world.load_yolo_world(model, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Cache the loaded model
        MODEL_CACHE[cache_key] = model_dict
        print(f"✅ Successfully loaded {model}")

        return (model_dict,)


class GroundingDetector:
    """
    Unified detector for all grounding models
    Auto-detects model type and uses appropriate detection method
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("GROUNDING_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "person . car . dog .",
                    "multiline": True,
                    "tooltip": "Period-separated (.) = multiple objects: 'banana. orange' finds bananas AND oranges. Comma or no separator = single object: 'banana, orange' finds items labeled 'banana, orange'"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Confidence threshold for detections. Typical: 0.2-0.35 (permissive), 0.35-0.5 (balanced), 0.5+ (strict)"
                }),
            },
            "optional": {
                "single_box_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return only the highest-scoring detection. Use for referring expressions (e.g., 'the red car on the left')"
                }),
                "single_box_per_prompt_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return highest-scoring detection for each prompt/label (e.g., 'banana. orange' returns best banana and best orange). Ignored if single_box_mode is True"
                }),
                "bbox_output_format": (["list_only", "dict_with_data"], {
                    "default": "list_only",
                    "tooltip": "list_only: SAM2-compatible | dict_with_data: includes labels/scores"
                }),
                "output_masks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Convert bounding boxes to binary masks. Enable for segmentation tasks or mask-based operations"
                }),
                # GroundingDINO parameters
                **grounding_dino.params.DETECTOR_PARAMS.get("optional", {}),
                # Florence-2 parameters
                **florence2.params.DETECTOR_PARAMS.get("optional", {}),
                # YOLO-World parameters
                **yolo_world.params.DETECTOR_PARAMS.get("optional", {}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffff,  # 2^32 - 1, max for numpy.random.seed
                    "tooltip": "Fixed seed for reproducible results (affects mask visualization colors and model randomness)"
                }),
            }
        }

    RETURN_TYPES = ("BBOX", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("bboxes", "annotated_image", "labels", "masks")
    FUNCTION = "detect"
    CATEGORY = "grounding"

    def detect(self, model, image, prompt: str, confidence_threshold: float,
               text_threshold: float = 0.25, single_box_mode: bool = False,
               single_box_per_prompt_mode: bool = False,
               bbox_output_format: str = "list_only", output_masks: bool = False,
               florence2_max_tokens: int = 1024, florence2_num_beams: int = 3,
               yolo_iou: float = 0.45, yolo_agnostic_nms: bool = False, yolo_max_det: int = 300,
               seed: int = 0):
        """Universal detection that works with any model"""

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_type = model["type"]

        # Route to appropriate detection method
        if model_type == "grounding_dino":
            return grounding_dino.detect_grounding_dino(
                model, image, prompt, confidence_threshold, text_threshold,
                single_box_mode, single_box_per_prompt_mode, bbox_output_format, output_masks, self._format_output
            )
        elif model_type == "owlv2":
            return owlv2.detect_owlv2(
                model, image, prompt, confidence_threshold,
                single_box_mode, single_box_per_prompt_mode, bbox_output_format, output_masks, self._format_output
            )
        elif model_type == "florence2":
            return florence2.detect_florence2(
                model, image, prompt, confidence_threshold, single_box_mode,
                single_box_per_prompt_mode, bbox_output_format, output_masks, florence2_max_tokens,
                florence2_num_beams, self._format_output
            )
        elif model_type == "yolo_world":
            return yolo_world.detect_yolo_world(
                model, image, prompt, confidence_threshold, single_box_mode,
                single_box_per_prompt_mode, bbox_output_format, output_masks, yolo_iou, yolo_agnostic_nms,
                yolo_max_det, self._format_output
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _format_output(self, all_boxes, all_labels, all_scores, annotated_images,
                       image_shape, bbox_output_format, output_masks):
        """Format detection output and optionally create masks"""

        # Combine annotated images
        annotated_batch = torch.cat(annotated_images, dim=0)

        # Format bboxes based on bbox_output_format
        if bbox_output_format == "list_only":
            batched_bboxes = all_boxes
        else:  # dict_with_data
            batched_bboxes = {
                "boxes": all_boxes,
                "labels": all_labels,
                "scores": all_scores,
            }

        # Format labels string (from first image)
        if len(all_labels) > 0 and len(all_labels[0]) > 0:
            labels_str = ", ".join([f"{label} ({score:.2f})" for label, score in zip(all_labels[0], all_scores[0])])
        else:
            labels_str = ""

        # Create masks if requested
        if output_masks:
            masks = boxes_to_masks(all_boxes, image_shape)
        else:
            # Return empty mask
            batch_size, height, width, _ = image_shape
            masks = torch.zeros((1, height, width))

        return (batched_bboxes, annotated_batch, labels_str, masks)


# ============================================================================
# Mask Generation Nodes
# ============================================================================

class GroundingMaskModelLoader:
    """
    Unified model loader for all mask generation models
    Supports: Florence-2 Seg, SA2VA, LISA, PSALM
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MASK_MODEL_REGISTRY.keys()), {
                    "default": "Florence-2: Base (Segmentation)",
                }),
            },
            "optional": {
                # Florence-2 Seg parameters
                **florence2_seg.params.LOADER_PARAMS.get("optional", {}),
                # SA2VA parameters
                **sa2va.params.LOADER_PARAMS.get("optional", {}),
            }
        }

    RETURN_TYPES = ("MASK_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "grounding"

    def load_model(self, model, florence2_attn="eager", sa2va_dtype="auto"):
        """Load mask generation model"""
        config = MASK_MODEL_REGISTRY[model]
        model_type = config["type"]

        # Create cache key
        cache_key = f"{model_type}_{model}_attn_{florence2_attn}_dtype_{sa2va_dtype}"
        if cache_key in MODEL_CACHE:
            print(f"✅ Loading {model} from cache")
            return (MODEL_CACHE[cache_key],)

        print(f"📂 Loading {model}...")

        # Route to appropriate loader
        if model_type == "florence2_seg":
            model_dict = florence2_seg.load_florence2_seg(model, config, florence2_attn)
        elif model_type == "sa2va":
            model_dict = sa2va.load_sa2va(model, config, sa2va_dtype)
        elif model_type == "lisa":
            raise NotImplementedError("LISA support coming soon")
        elif model_type == "psalm":
            raise NotImplementedError("PSALM support coming soon")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Cache the loaded model
        MODEL_CACHE[cache_key] = model_dict
        print(f"✅ Successfully loaded {model}")

        return (model_dict,)


class GroundingMaskDetector:
    """
    Unified detector for mask generation models
    Auto-detects model type and uses appropriate detection method
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MASK_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "Segment the main object in the image",
                    "multiline": True,
                    "tooltip": "For Florence-2: descriptive phrase. For SA2VA: explicit segmentation instruction (e.g., 'Segment the person')"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Confidence threshold for mask filtering (where applicable)"
                }),
            },
            "optional": {
                # Florence-2 Seg parameters (only used when Florence-2 Seg model is loaded)
                "florence2_max_tokens": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "[Florence-2 only] Maximum tokens for generation"
                }),
                "florence2_num_beams": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "[Florence-2 only] Beam search width"
                }),
                # SA2VA parameters (only used when SA2VA model is loaded)
                "sa2va_max_tokens": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "[SA2VA only] Maximum tokens for generation"
                }),
                "sa2va_num_beams": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "[SA2VA only] Beam search width"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Fixed seed for reproducible results"
                }),
            }
        }

    RETURN_TYPES = sa2va.params.RETURN_TYPES
    RETURN_NAMES = sa2va.params.RETURN_NAMES
    FUNCTION = "detect"
    CATEGORY = "grounding"

    def detect(self, model, image, prompt: str, confidence_threshold: float,
               florence2_max_tokens: int = 1024, florence2_num_beams: int = 3,
               sa2va_max_tokens: int = 2048, sa2va_num_beams: int = 1,
               seed: int = 0):
        """Universal mask detection that works with any model"""

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_type = model["type"]

        # Route to appropriate detection method
        if model_type == "florence2_seg":
            return florence2_seg.detect_florence2_seg(
                model, image, prompt, confidence_threshold,
                florence2_max_tokens, florence2_num_beams,
                self._format_output
            )
        elif model_type == "sa2va":
            return sa2va.detect_sa2va(
                model, image, prompt, confidence_threshold,
                sa2va_max_tokens, sa2va_num_beams, self._format_output
            )
        elif model_type == "lisa":
            raise NotImplementedError("LISA support coming soon")
        elif model_type == "psalm":
            raise NotImplementedError("PSALM support coming soon")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _format_output(self, masks, bboxes, labels_list, annotated_images, text):
        """Format mask detection output

        Args:
            masks: List of mask tensors per batch
            bboxes: List of bboxes per batch (not returned, used internally)
            labels_list: List of labels per batch
            annotated_images: List of overlaid mask images
            text: Generated text output (for SA2VA)

        Returns:
            Tuple of (masks, labels_str, overlaid_mask, text)
        """
        # Stack masks - convert from list of lists of numpy arrays to single tensor
        if len(masks) > 0 and any(len(batch_masks) > 0 for batch_masks in masks):
            # Flatten list of lists and convert numpy arrays to tensors
            all_mask_tensors = []
            for batch_masks in masks:
                for mask in batch_masks:
                    if isinstance(mask, np.ndarray):
                        mask_tensor = torch.from_numpy(mask).float()
                    else:
                        mask_tensor = mask
                    # Ensure 2D mask becomes 3D (1, H, W)
                    if mask_tensor.ndim == 2:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    all_mask_tensors.append(mask_tensor)
            if len(all_mask_tensors) > 0:
                masks_batch = torch.cat(all_mask_tensors, dim=0)
            else:
                # No masks found, return empty
                if len(annotated_images) > 0:
                    _, H, W, _ = annotated_images[0].shape
                else:
                    H, W = 512, 512
                masks_batch = torch.zeros((1, H, W))
        else:
            # Return empty mask
            if len(annotated_images) > 0:
                _, H, W, _ = annotated_images[0].shape
            else:
                H, W = 512, 512
            masks_batch = torch.zeros((1, H, W))

        # Combine overlaid mask images
        if len(annotated_images) > 0:
            overlaid_mask = torch.cat(annotated_images, dim=0)
        else:
            overlaid_mask = torch.zeros((1, 512, 512, 3))

        return (masks_batch, overlaid_mask, text)


# ============================================================================
# SAM2 Nodes
# ============================================================================

class DownloadAndLoadSAM2Model:
    @classmethod
    def INPUT_TYPES(s):
        return sam2.params.LOADER_PARAMS

    RETURN_TYPES = sam2.params.LOADER_RETURN_TYPES
    RETURN_NAMES = sam2.params.LOADER_RETURN_NAMES
    FUNCTION = "loadmodel"
    CATEGORY = "SAM2"

    def loadmodel(self, model, segmentor, device, precision):
        """Load SAM2 model"""
        sam2_model = sam2.load_sam2(model, segmentor, device, precision, script_directory)
        return (sam2_model,)


class Sam2Segmentation:
    def __init__(self):
        self.inference_state = None

    @classmethod
    def INPUT_TYPES(s):
        return sam2.params.SEGMENTATION_PARAMS

    RETURN_TYPES = sam2.params.SEGMENTATION_RETURN_TYPES
    RETURN_NAMES = sam2.params.SEGMENTATION_RETURN_NAMES
    FUNCTION = "segment"
    CATEGORY = "SAM2"

    def segment(self, image, sam2_model, keep_model_loaded, coordinates_positive=None,
                coordinates_negative=None, individual_objects=False, bboxes=None,
                mask=None, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0):
        """Perform SAM2 segmentation"""
        # Create a reference dict to hold inference_state
        inference_state_ref = {'state': self.inference_state}

        result = sam2.segment_sam2(
            image, sam2_model, keep_model_loaded, coordinates_positive,
            coordinates_negative, individual_objects, bboxes, mask,
            mask_threshold, max_hole_area, max_sprinkle_area, inference_state_ref
        )

        # Update instance inference_state
        self.inference_state = inference_state_ref['state']

        return result


# ============================================================================
# Visualization Nodes
# ============================================================================

class BboxVisualizer:
    """
    Visualizes bounding boxes on images
    """

    @classmethod
    def INPUT_TYPES(cls):
        return visualizer.params.VISUALIZER_PARAMS

    RETURN_TYPES = visualizer.params.RETURN_TYPES
    RETURN_NAMES = visualizer.params.RETURN_NAMES
    FUNCTION = "visualize"
    CATEGORY = "grounding"

    def visualize(self, image, bboxes, line_width=3):
        """Draw bounding boxes on image (supports batches)"""
        return visualizer.visualize_bboxes(image, bboxes, line_width)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "GroundingModelLoader": GroundingModelLoader,
    "GroundingDetector": GroundingDetector,
    "GroundingMaskModelLoader": GroundingMaskModelLoader,
    "GroundingMaskDetector": GroundingMaskDetector,
    "DownloadAndLoadSAM2Model": DownloadAndLoadSAM2Model,
    "Sam2Segmentation": Sam2Segmentation,
    "BboxVisualizer": BboxVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundingModelLoader": "Grounding Model Loader",
    "GroundingDetector": "Grounding Detector",
    "GroundingMaskModelLoader": "Grounding Mask Loader",
    "GroundingMaskDetector": "Grounding Mask Detector",
    "DownloadAndLoadSAM2Model": "SAM2 Model Loader",
    "Sam2Segmentation": "SAM2 Segmentation",
    "BboxVisualizer": "Bounding Box Visualizer",
}
