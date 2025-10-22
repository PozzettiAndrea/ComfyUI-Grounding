"""
Main node implementations for ComfyUI-Grounding
"""

import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as mm
import os
from typing import Tuple, List, Optional
import torchvision.transforms as T

# Global model cache for keeping models in memory
MODEL_CACHE = {}

# Model registry for GroundingDINO variants
GROUNDING_DINO_MODELS = {
    # Original GroundingDINO models
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config": "GroundingDINO_SwinT_OGC.cfg.py",
        "checkpoint": "groundingdino_swint_ogc.pth",
        "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        "type": "local",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config": "GroundingDINO_SwinB.cfg.py",
        "checkpoint": "groundingdino_swinb_cogcoor.pth",
        "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
        "type": "local",
    },
    # MM-GroundingDINO Tiny models (Swin-T backbone) - 0.2B parameters
    "MM-GroundingDINO-Tiny (O365+GoldG, 50.4 mAP)": {
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg",
        "type": "transformers",
    },
    "MM-GroundingDINO-Tiny (O365+GoldG+GRIT, 50.5 mAP)": {
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit",
        "type": "transformers",
    },
    "MM-GroundingDINO-Tiny (O365+GoldG+V3Det, 50.6 mAP)": {
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
        "type": "transformers",
    },
    "MM-GroundingDINO-Tiny (O365+GoldG+GRIT+V3Det, 50.4 mAP)": {
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det",
        "type": "transformers",
    },
    # MM-GroundingDINO Base models (Swin-B backbone) - 0.2B parameters
    "MM-GroundingDINO-Base (O365+GoldG+V3Det, 52.5 mAP)": {
        "hf_id": "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
        "type": "transformers",
    },
    "MM-GroundingDINO-Base (All datasets, 59.5 mAP)": {
        "hf_id": "openmmlab-community/mm_grounding_dino_base_all",
        "type": "transformers",
    },
    # MM-GroundingDINO Large models (Swin-L backbone) - 0.3B parameters
    "MM-GroundingDINO-Large (O365v2+OIv6+GoldG, 53.0 mAP)": {
        "hf_id": "openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg",
        "type": "transformers",
    },
    "MM-GroundingDINO-Large (All datasets, 60.3 mAP)": {
        "hf_id": "openmmlab-community/mm_grounding_dino_large_all",
        "type": "transformers",
    },
}

YOLO_WORLD_MODELS = {
    "yolov8s-world": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt",
    },
    "yolov8m-world": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-world.pt",
    },
    "yolov8l-world": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-world.pt",
    },
    "yolov8x-world": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-world.pt",
    },
}


class GroundingDINOModelLoader:
    """
    Loads GroundingDINO models for open-vocabulary object detection
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(GROUNDING_DINO_MODELS.keys()), {"default": "GroundingDINO_SwinT_OGC (694MB)"}),
                "keep_in_memory": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("GROUNDING_DINO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "grounding"

    def load_model(self, model_name: str, keep_in_memory: bool = True):
        """Load GroundingDINO model with caching support"""

        # Create cache key
        cache_key = f"grounding_dino_{model_name}"

        # Check cache if keep_in_memory is True
        if keep_in_memory and cache_key in MODEL_CACHE:
            print(f"Loading {model_name} from cache")
            return (MODEL_CACHE[cache_key],)

        # Clear from cache if keep_in_memory is False
        if not keep_in_memory and cache_key in MODEL_CACHE:
            print(f"Removing {model_name} from cache")
            del MODEL_CACHE[cache_key]

        # Get model config
        model_config = GROUNDING_DINO_MODELS[model_name]
        model_type = model_config.get("type", "local")

        # Load model based on type
        if model_type == "local":
            model = self._load_local(model_name, model_config)
        else:  # transformers
            model = self._load_with_transformers(model_name, model_config)

        # Cache if requested
        if keep_in_memory:
            MODEL_CACHE[cache_key] = model
            print(f"Cached {model_name} in memory")

        return (model,)

    def _load_local(self, model_name: str, model_config: dict):
        """Load GroundingDINO using local implementation"""
        try:
            # Try using the local implementation from comfyui-sam2
            from custom_nodes.comfyui_sam2.node import load_groundingdino_model
            model = load_groundingdino_model(model_name)
            return model
        except ImportError:
            raise RuntimeError(
                f"Local GroundingDINO implementation not found. "
                f"Please install ComfyUI-segment-anything-2 to use {model_name}, "
                f"or use MM-GroundingDINO models instead."
            )

    def _load_with_transformers(self, model_name: str, model_config: dict):
        """Load MM-GroundingDINO using HuggingFace transformers"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            device = mm.get_torch_device()

            # Save models to local Models folder in plugin directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, "Models")
            os.makedirs(models_dir, exist_ok=True)

            # Get HuggingFace model ID
            hf_id = model_config.get("hf_id")

            if not hf_id:
                raise ValueError(f"No HuggingFace model ID found for {model_name}")

            print(f"📥 Loading {model_name} from HuggingFace ({hf_id})...")
            print("Downloading model files (progress bars will appear below):")

            # Load processor and model (automatically shows tqdm progress bars)
            processor = AutoProcessor.from_pretrained(hf_id, cache_dir=models_dir)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_id, cache_dir=models_dir)

            model.to(device)
            model.eval()

            print(f"✅ Successfully loaded {model_name}")

            return {"model": model, "processor": processor, "type": "transformers"}
        except Exception as e:
            raise RuntimeError(f"Failed to load GroundingDINO model: {e}")


class GroundingDINODetector:
    """
    Performs object detection using GroundingDINO with text prompts
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("GROUNDING_DINO_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "person . car . dog .",
                    "multiline": True,
                }),
                "box_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "text_threshold": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "single_box_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return only the highest-scoring detection (REC mode for referring expressions)"
                }),
                "output_format": (["list_only", "dict_with_data"], {
                    "default": "list_only",
                    "tooltip": "list_only: SAM2-compatible list of lists | dict_with_data: dict with boxes/labels/scores"
                }),
            }
        }

    RETURN_TYPES = ("BBOX", "IMAGE", "STRING")
    RETURN_NAMES = ("bboxes", "annotated_image", "labels")
    FUNCTION = "detect"
    CATEGORY = "grounding"

    def detect(self, model, image, prompt: str, box_threshold: float, text_threshold: float, single_box_mode: bool = False, output_format: str = "list_only"):
        """Run detection on image with text prompt"""

        # Process all images in batch
        batch_size = image.shape[0]
        all_boxes = []
        all_labels = []
        all_scores = []
        annotated_images = []

        for i in range(batch_size):
            # Convert ComfyUI image format (B, H, W, C) in [0, 1] to PIL
            image_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            # Check model type and run detection
            if isinstance(model, dict) and model.get("type") == "transformers":
                bboxes, annotated_tensor, _ = self._detect_transformers(model, pil_image, prompt, box_threshold, text_threshold, single_box_mode)
            else:
                bboxes, annotated_tensor, _ = self._detect_local(model, pil_image, prompt, box_threshold, single_box_mode)

            # Accumulate results
            all_boxes.append(bboxes["boxes"])
            all_labels.append(bboxes["labels"])
            all_scores.append(bboxes["scores"])
            annotated_images.append(annotated_tensor)

        # Combine annotated images
        annotated_batch = torch.cat(annotated_images, dim=0)

        # Format batched output based on output_format
        if output_format == "list_only":
            # SAM2-compatible format: list of lists (boxes only)
            batched_bboxes = all_boxes
        else:  # dict_with_data
            # Dict format with all information
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

        return (batched_bboxes, annotated_batch, labels_str)

    def _detect_transformers(self, model_dict, pil_image, prompt, box_threshold, text_threshold, single_box_mode=False):
        """Detection using transformers implementation"""
        model = model_dict["model"]
        processor = model_dict["processor"]
        device = mm.get_torch_device()

        # Prepare inputs
        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process (use 'threshold' instead of 'box_threshold' for GroundingDINO)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]]
        )[0]

        # Extract boxes and labels
        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"]
        scores = results["scores"].cpu().numpy()

        # Override labels if prompt has no separators
        # Check for periods and commas (common separators in GroundingDINO prompts)
        if '.' not in prompt and ',' not in prompt:
            # Use full prompt as label for all detections
            clean_prompt = prompt.strip()
            labels = [clean_prompt] * len(labels)

        # Single box mode: return only highest-scoring detection (REC mode)
        if single_box_mode and len(boxes) > 0:
            top_idx = scores.argmax()
            boxes = boxes[top_idx:top_idx+1]
            labels = [labels[top_idx]]
            scores = scores[top_idx:top_idx+1]

        # Format output
        bboxes = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

        # Create annotated image
        annotated = self._draw_boxes(pil_image, boxes, labels, scores)
        annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

        # Format labels string
        labels_str = ", ".join([f"{label} ({score:.2f})" for label, score in zip(labels, scores)])

        return (bboxes, annotated_tensor, labels_str)

    def _detect_local(self, model, pil_image, prompt, box_threshold, single_box_mode=False):
        """Detection using local GroundingDINO implementation"""
        # Use the implementation from comfyui-sam2
        from custom_nodes.comfyui_sam2.node import groundingdino_predict

        boxes = groundingdino_predict(model, pil_image, prompt, box_threshold)

        # Convert to numpy
        boxes_np = boxes.cpu().numpy()

        # Create labels (simplified - just use prompt)
        labels = [prompt.strip().rstrip('.')] * len(boxes_np)
        scores = np.ones(len(boxes_np)) * box_threshold  # Approximate

        # Single box mode: return only first detection (local impl doesn't provide scores)
        if single_box_mode and len(boxes_np) > 0:
            boxes_np = boxes_np[0:1]
            labels = [labels[0]]
            scores = scores[0:1]

        bboxes = {
            "boxes": boxes_np,
            "labels": labels,
            "scores": scores,
        }

        # Create annotated image
        annotated = self._draw_boxes(pil_image, boxes_np, labels, scores)
        annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

        labels_str = ", ".join(labels)

        return (bboxes, annotated_tensor, labels_str)

    def _draw_boxes(self, image, boxes, labels, scores):
        """Draw bounding boxes on image"""
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(image)

        for box, label, score in zip(boxes, labels, scores):
            # box format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw label
            text = f"{label}: {score:.2f}"
            draw.text((x1, y1 - 10), text, fill="red")

        return image


class YOLOWorldModelLoader:
    """
    Loads YOLO-World models for open-vocabulary object detection
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(YOLO_WORLD_MODELS.keys()), {"default": "yolov8s-world"}),
                "keep_in_memory": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("YOLO_WORLD_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "grounding"

    def load_model(self, model_name: str, keep_in_memory: bool = True):
        """Load YOLO-World model with caching support"""

        # Create cache key
        cache_key = f"yolo_world_{model_name}"

        # Check cache if keep_in_memory is True
        if keep_in_memory and cache_key in MODEL_CACHE:
            print(f"Loading {model_name} from cache")
            return (MODEL_CACHE[cache_key],)

        # Clear from cache if keep_in_memory is False
        if not keep_in_memory and cache_key in MODEL_CACHE:
            print(f"Removing {model_name} from cache")
            del MODEL_CACHE[cache_key]

        try:
            from ultralytics import YOLOWorld

            # Save models to local Models folder in plugin directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, "Models")
            os.makedirs(models_dir, exist_ok=True)

            model_filename = f"{model_name}.pt"
            model_path = os.path.join(models_dir, model_filename)

            # Download if needed
            if not os.path.exists(model_path):
                print(f"📥 Downloading {model_name} to {models_dir}...")
                print("Progress bar:")
                from torch.hub import download_url_to_file
                download_url_to_file(
                    YOLO_WORLD_MODELS[model_name]["url"],
                    model_path,
                    progress=True  # Explicitly enable progress bar
                )
                print(f"✅ Downloaded {model_name} successfully")

            # Load model
            print(f"Loading {model_name} from {model_path}")
            model = YOLOWorld(model_path)

            # Cache if requested
            if keep_in_memory:
                MODEL_CACHE[cache_key] = model
                print(f"Cached {model_name} in memory")

            return (model,)
        except ImportError:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO-World model: {e}")


class YOLOWorldDetector:
    """
    Performs object detection using YOLO-World with text prompts
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("YOLO_WORLD_MODEL",),
                "image": ("IMAGE",),
                "classes": ("STRING", {
                    "default": "person, car, dog",
                    "multiline": True,
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "single_box_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return only the highest-scoring detection (REC mode for referring expressions)"
                }),
                "output_format": (["list_only", "dict_with_data"], {
                    "default": "list_only",
                    "tooltip": "list_only: SAM2-compatible list of lists | dict_with_data: dict with boxes/labels/scores"
                }),
            }
        }

    RETURN_TYPES = ("BBOX", "IMAGE", "STRING")
    RETURN_NAMES = ("bboxes", "annotated_image", "labels")
    FUNCTION = "detect"
    CATEGORY = "grounding"

    def detect(self, model, image, classes: str, confidence_threshold: float, single_box_mode: bool = False, output_format: str = "list_only"):
        """Run detection with YOLO-World"""

        # Process all images in batch
        batch_size = image.shape[0]
        all_boxes = []
        all_labels = []
        all_scores = []
        annotated_images = []

        # Parse classes - check if we should use full text as label
        has_separator = ',' in classes
        if has_separator:
            class_list = [c.strip() for c in classes.split(",")]
        else:
            # No comma separator - treat as single class/referring expression
            class_list = [classes.strip()]

        # Set classes for the model
        model.set_classes(class_list)

        for i in range(batch_size):
            # Convert ComfyUI image format
            image_np = (image[i].cpu().numpy() * 255).astype(np.uint8)

            # Run inference
            results = model(image_np, conf=confidence_threshold)

            # Extract detections
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
            class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])

            # Map class IDs to names
            labels = [class_list[cid] for cid in class_ids]

            # Override labels if no separator (referring expression mode)
            if not has_separator and len(labels) > 0:
                # Use full prompt as label for all detections
                labels = [classes.strip()] * len(labels)

            # Single box mode: return only highest-scoring detection (REC mode)
            if single_box_mode and len(boxes) > 0:
                top_idx = scores.argmax()
                boxes = boxes[top_idx:top_idx+1]
                labels = [labels[top_idx]]
                scores = scores[top_idx:top_idx+1]

            # Get annotated image
            annotated_np = result.plot()
            annotated_tensor = torch.from_numpy(annotated_np.astype(np.float32) / 255.0).unsqueeze(0)

            # Accumulate results
            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)
            annotated_images.append(annotated_tensor)

        # Combine annotated images
        annotated_batch = torch.cat(annotated_images, dim=0)

        # Format batched output based on output_format
        if output_format == "list_only":
            # SAM2-compatible format: list of lists (boxes only)
            batched_bboxes = all_boxes
        else:  # dict_with_data
            # Dict format with all information
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

        return (batched_bboxes, annotated_batch, labels_str)


class BboxVisualizer:
    """
    Visualizes bounding boxes on images
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOX",),
            },
            "optional": {
                "line_width": ("INT", {"default": 3, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("annotated_image",)
    FUNCTION = "visualize"
    CATEGORY = "grounding"

    def visualize(self, image, bboxes, line_width=3):
        """Draw bounding boxes on image"""
        from PIL import ImageDraw

        # Convert to PIL
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        draw = ImageDraw.Draw(pil_image)

        boxes = bboxes["boxes"]
        labels = bboxes.get("labels", [])
        scores = bboxes.get("scores", [])

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

            if i < len(labels):
                label = labels[i]
                score = scores[i] if i < len(scores) else 0.0
                text = f"{label}: {score:.2f}"
                draw.text((x1, y1 - 10), text, fill="red")

        # Convert back to tensor
        annotated_np = np.array(pil_image).astype(np.float32) / 255.0
        annotated_tensor = torch.from_numpy(annotated_np).unsqueeze(0)

        return (annotated_tensor,)


class BboxToMask:
    """
    Converts bounding boxes to masks
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOX",),
            },
            "optional": {
                "combine_masks": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "bbox_to_mask"
    CATEGORY = "grounding"

    def bbox_to_mask(self, image, bboxes, combine_masks=False):
        """Convert bounding boxes to binary masks"""

        # Get image dimensions
        batch_size, height, width, channels = image.shape
        boxes = bboxes["boxes"]

        if combine_masks:
            # Single combined mask
            mask = np.zeros((height, width), dtype=np.float32)
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                mask[y1:y2, x1:x2] = 1.0

            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        else:
            # Individual masks for each box
            masks = []
            for box in boxes:
                mask = np.zeros((height, width), dtype=np.float32)
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                mask[y1:y2, x1:x2] = 1.0
                masks.append(mask)

            if len(masks) == 0:
                # No detections - return empty mask
                mask_tensor = torch.zeros((1, height, width))
            else:
                mask_tensor = torch.from_numpy(np.stack(masks))

        return (mask_tensor,)


class UnifiedDetector:
    """
    Universal detector that works with both GroundingDINO and YOLO-World models
    Automatically detects model type and uses appropriate inference method
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("GROUNDING_DINO_MODEL,YOLO_WORLD_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "person . car . dog .",
                    "multiline": True,
                    "tooltip": "For GroundingDINO: period-separated (e.g., 'cat . dog .'). For YOLO-World: comma-separated (e.g., 'cat, dog')"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "text_threshold": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Only used for GroundingDINO models"
                }),
                "single_box_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return only the highest-scoring detection (REC mode for referring expressions)"
                }),
            }
        }

    RETURN_TYPES = ("BBOX", "IMAGE", "STRING")
    RETURN_NAMES = ("bboxes", "annotated_image", "labels")
    FUNCTION = "detect"
    CATEGORY = "grounding"

    def detect(self, model, image, prompt: str, threshold: float, text_threshold: float = 0.25, single_box_mode: bool = False):
        """Universal detection that auto-detects model type"""

        # Detect model type
        model_type = self._detect_model_type(model)

        if model_type == "yolo_world":
            return self._detect_yolo_world(model, image, prompt, threshold, single_box_mode)
        else:  # grounding_dino or transformers
            return self._detect_grounding_dino(model, image, prompt, threshold, text_threshold, single_box_mode)

    def _detect_model_type(self, model):
        """Detect whether this is a YOLO-World or GroundingDINO model"""
        # Check for YOLO-World model
        try:
            from ultralytics import YOLOWorld
            if isinstance(model, YOLOWorld):
                return "yolo_world"
        except ImportError:
            pass

        # Check for transformers-based model
        if isinstance(model, dict) and model.get("type") == "transformers":
            return "grounding_dino_transformers"

        # Default to GroundingDINO (local)
        return "grounding_dino_local"

    def _detect_grounding_dino(self, model, image, prompt, box_threshold, text_threshold, single_box_mode=False):
        """Detection using GroundingDINO"""
        # Convert ComfyUI image format
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Check model type
        if isinstance(model, dict) and model.get("type") == "transformers":
            return self._detect_transformers(model, pil_image, prompt, box_threshold, text_threshold, single_box_mode)
        else:
            return self._detect_local(model, pil_image, prompt, box_threshold, single_box_mode)

    def _detect_transformers(self, model_dict, pil_image, prompt, box_threshold, text_threshold, single_box_mode=False):
        """Detection using transformers implementation"""
        model = model_dict["model"]
        processor = model_dict["processor"]
        device = mm.get_torch_device()

        # Prepare inputs
        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process (use 'threshold' instead of 'box_threshold' for GroundingDINO)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]]
        )[0]

        # Extract boxes and labels
        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"]
        scores = results["scores"].cpu().numpy()

        # Single box mode: return only highest-scoring detection (REC mode)
        if single_box_mode and len(boxes) > 0:
            top_idx = scores.argmax()
            boxes = boxes[top_idx:top_idx+1]
            labels = [labels[top_idx]]
            scores = scores[top_idx:top_idx+1]

        # Format output
        bboxes = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

        # Create annotated image
        annotated = self._draw_boxes(pil_image, boxes, labels, scores)
        annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

        # Format labels string
        labels_str = ", ".join([f"{label} ({score:.2f})" for label, score in zip(labels, scores)])

        return (bboxes, annotated_tensor, labels_str)

    def _detect_local(self, model, pil_image, prompt, box_threshold, single_box_mode=False):
        """Detection using local GroundingDINO implementation"""
        from custom_nodes.comfyui_sam2.node import groundingdino_predict

        boxes = groundingdino_predict(model, pil_image, prompt, box_threshold)

        # Convert to numpy
        boxes_np = boxes.cpu().numpy()

        # Create labels
        labels = [prompt.strip().rstrip('.')] * len(boxes_np)
        scores = np.ones(len(boxes_np)) * box_threshold

        # Single box mode: return only first detection (local impl doesn't provide scores)
        if single_box_mode and len(boxes_np) > 0:
            boxes_np = boxes_np[0:1]
            labels = [labels[0]]
            scores = scores[0:1]

        bboxes = {
            "boxes": boxes_np,
            "labels": labels,
            "scores": scores,
        }

        # Create annotated image
        annotated = self._draw_boxes(pil_image, boxes_np, labels, scores)
        annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

        labels_str = ", ".join(labels)

        return (bboxes, annotated_tensor, labels_str)

    def _detect_yolo_world(self, model, image, classes, confidence_threshold, single_box_mode=False):
        """Detection using YOLO-World"""
        # Convert ComfyUI image format
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)

        # Parse classes (support both comma and period separation)
        if ',' in classes:
            class_list = [c.strip() for c in classes.split(",")]
        else:
            class_list = [c.strip() for c in classes.split(".") if c.strip()]

        # Set classes for the model
        model.set_classes(class_list)

        # Run inference
        results = model(image_np, conf=confidence_threshold)

        # Extract detections
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])

        # Map class IDs to names
        labels = [class_list[cid] for cid in class_ids]

        # Single box mode: return only highest-scoring detection (REC mode)
        if single_box_mode and len(boxes) > 0:
            top_idx = scores.argmax()
            boxes = boxes[top_idx:top_idx+1]
            labels = [labels[top_idx]]
            scores = scores[top_idx:top_idx+1]

        # Format output
        bboxes = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

        # Get annotated image
        annotated_np = result.plot()
        annotated_tensor = torch.from_numpy(annotated_np.astype(np.float32) / 255.0).unsqueeze(0)

        # Format labels string
        labels_str = ", ".join([f"{label} ({score:.2f})" for label, score in zip(labels, scores)])

        return (bboxes, annotated_tensor, labels_str)

    def _draw_boxes(self, image, boxes, labels, scores):
        """Draw bounding boxes on image"""
        from PIL import ImageDraw

        # Make a copy to avoid modifying original
        image = image.copy()
        draw = ImageDraw.Draw(image)

        for box, label, score in zip(boxes, labels, scores):
            # box format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw label
            text = f"{label}: {score:.2f}"
            draw.text((x1, y1 - 10), text, fill="red")

        return image
