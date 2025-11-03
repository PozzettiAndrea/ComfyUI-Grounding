"""
ComfyUI-Grounding: Unified object detection nodes for ComfyUI
Simplified architecture with just 3 nodes for all grounding models
"""

import torch
from torch.functional import F
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar, common_upscale
import os
import json
import random
from typing import Tuple, List, Optional
import torchvision.transforms as T
from tqdm import tqdm
from contextlib import nullcontext

from .load_model import load_model

script_directory = os.path.dirname(os.path.abspath(__file__))

# Global model cache for keeping models in memory
MODEL_CACHE = {}

# Unified model registry with prefixed names for easy identification
MODEL_REGISTRY = {
    # GroundingDINO models
    "GroundingDINO: SwinT OGC (694MB)": {
        "type": "grounding_dino",
        "hf_id": "IDEA-Research/grounding-dino-tiny",
        "framework": "transformers",
    },
    "GroundingDINO: SwinB (938MB)": {
        "type": "grounding_dino",
        "hf_id": "IDEA-Research/grounding-dino-base",
        "framework": "transformers",
    },
    # MM-GroundingDINO Tiny models
    "MM-GroundingDINO: Tiny O365+GoldG (50.4 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg",
        "framework": "transformers",
    },
    "MM-GroundingDINO: Tiny O365+GoldG+GRIT (50.5 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit",
        "framework": "transformers",
    },
    "MM-GroundingDINO: Tiny O365+GoldG+V3Det (50.6 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
        "framework": "transformers",
    },
    # MM-GroundingDINO Base models
    "MM-GroundingDINO: Base O365+GoldG+V3Det (52.5 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
        "framework": "transformers",
    },
    "MM-GroundingDINO: Base All Datasets (59.5 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_base_all",
        "framework": "transformers",
    },
    # MM-GroundingDINO Large models
    "MM-GroundingDINO: Large O365v2+OIv6+GoldG (53.0 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg",
        "framework": "transformers",
    },
    "MM-GroundingDINO: Large All Datasets (60.3 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_large_all",
        "framework": "transformers",
    },
    # OWLv2 models
    "OWLv2: Base Patch16": {
        "type": "owlv2",
        "hf_id": "google/owlv2-base-patch16",
        "framework": "transformers",
    },
    "OWLv2: Large Patch14": {
        "type": "owlv2",
        "hf_id": "google/owlv2-large-patch14",
        "framework": "transformers",
    },
    "OWLv2: Base Patch16 Ensemble": {
        "type": "owlv2",
        "hf_id": "google/owlv2-base-patch16-ensemble",
        "framework": "transformers",
    },
    "OWLv2: Large Patch14 Ensemble": {
        "type": "owlv2",
        "hf_id": "google/owlv2-large-patch14-ensemble",
        "framework": "transformers",
    },
    # Florence-2 models
    "Florence-2: Base (0.23B params)": {
        "type": "florence2",
        "hf_id": "microsoft/Florence-2-base",
        "framework": "transformers",
    },
    "Florence-2: Large (0.77B params)": {
        "type": "florence2",
        "hf_id": "microsoft/Florence-2-large",
        "framework": "transformers",
    },
    # YOLO-World models
    "YOLO-World: v8s (Small)": {
        "type": "yolo_world",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
        "framework": "ultralytics",
    },
    "YOLO-World: v8m (Medium)": {
        "type": "yolo_world",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-worldv2.pt",
        "framework": "ultralytics",
    },
    "YOLO-World: v8l (Large)": {
        "type": "yolo_world",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt",
        "framework": "ultralytics",
    },
    "YOLO-World: v8x (Extra Large)": {
        "type": "yolo_world",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt",
        "framework": "ultralytics",
    },
}


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
                "keep_in_memory": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "florence2_attn": (["eager", "sdpa", "flash_attention_2"], {
                    "default": "eager",
                    "tooltip": "⚠️ FLORENCE-2 ONLY! Ignored for all other models. eager=most compatible, sdpa=PyTorch 2.0+, flash_attention_2=A100/H100"
                }),
            }
        }

    RETURN_TYPES = ("GROUNDING_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "grounding"

    def load_model(self, model: str, keep_in_memory: bool = True, florence2_attn: str = "eager"):
        """Load any grounding model with unified interface"""

        # Get model config
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model}")

        config = MODEL_REGISTRY[model]
        model_type = config["type"]

        # Create cache key
        cache_key = f"{model_type}_{model}_{florence2_attn if model_type == 'florence2' else 'default'}"

        # Check cache
        if keep_in_memory and cache_key in MODEL_CACHE:
            print(f"✅ Loading {model} from cache")
            return (MODEL_CACHE[cache_key],)

        # Clear cache if requested
        if not keep_in_memory and cache_key in MODEL_CACHE:
            print(f"🗑️ Removing {model} from cache")
            del MODEL_CACHE[cache_key]

        # Load model based on framework
        if config["framework"] == "transformers":
            loaded_model = self._load_transformers_model(model, config, florence2_attn)
        elif config["framework"] == "ultralytics":
            loaded_model = self._load_yolo_model(model, config)
        else:
            raise ValueError(f"Unknown framework: {config['framework']}")

        # Cache if requested
        if keep_in_memory:
            MODEL_CACHE[cache_key] = loaded_model
            print(f"💾 Cached {model} in memory")

        return (loaded_model,)

    def _load_transformers_model(self, model_name: str, config: dict, florence2_attn: str):
        """Load transformers-based models (GroundingDINO, OWLv2, Florence-2)"""
        model_type = config["type"]
        hf_id = config["hf_id"]
        device = mm.get_torch_device()

        # Use ComfyUI standard model directories
        if model_type == "grounding_dino":
            cache_dir = os.path.join(folder_paths.models_dir, "grounding-dino")
        else:  # owlv2, florence2
            cache_dir = os.path.join(folder_paths.models_dir, "LLM")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"📥 Loading {model_name} from HuggingFace ({hf_id})...")
        print(f"📂 Cache directory: {cache_dir}")

        if model_type in ["grounding_dino"]:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            processor = AutoProcessor.from_pretrained(hf_id, cache_dir=cache_dir)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_id, cache_dir=cache_dir)

        elif model_type == "owlv2":
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            processor = Owlv2Processor.from_pretrained(hf_id, cache_dir=cache_dir)
            model = Owlv2ForObjectDetection.from_pretrained(hf_id, cache_dir=cache_dir)

        elif model_type == "florence2":
            from transformers import AutoProcessor, AutoModelForCausalLM
            print(f"Using Florence-2 attention implementation: {florence2_attn}")
            processor = AutoProcessor.from_pretrained(hf_id, cache_dir=cache_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
                attn_implementation=florence2_attn
            )
        else:
            raise ValueError(f"Unknown transformers model type: {model_type}")

        model.to(device)
        model.eval()

        print(f"✅ Successfully loaded {model_name}")

        return {
            "model": model,
            "processor": processor,
            "type": model_type,
            "framework": "transformers"
        }

    def _load_yolo_model(self, model_name: str, config: dict):
        """Load YOLO-World models"""
        try:
            from ultralytics import YOLOWorld
        except ImportError:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

        # Use ComfyUI standard models directory
        models_dir = os.path.join(folder_paths.models_dir, "yolo_world")
        os.makedirs(models_dir, exist_ok=True)

        # Extract filename from URL
        url = config["url"]
        filename = url.split("/")[-1]
        model_path = os.path.join(models_dir, filename)

        # Download if needed
        if not os.path.exists(model_path):
            print(f"📥 Downloading {model_name}...")
            from torch.hub import download_url_to_file
            download_url_to_file(url, model_path, progress=True)
            print(f"✅ Downloaded {model_name}")

        # Load model
        print(f"📂 Loading {model_name} from {model_path}")
        model = YOLOWorld(model_path)

        # Move to device
        device = mm.get_torch_device()
        model.to(device)

        print(f"✅ Successfully loaded {model_name}")

        return {
            "model": model,
            "type": "yolo_world",
            "framework": "ultralytics"
        }


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
                    "tooltip": "Use periods (.) for GroundingDINO/OWLv2/Florence-2, commas (,) for YOLO-World"
                }),
                "confidence_threshold": ("FLOAT", {
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
                    "tooltip": "⚠️ Only used for GroundingDINO models"
                }),
                "single_box_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return only highest-scoring detection (for referring expressions)"
                }),
                "bbox_output_format": (["list_only", "dict_with_data"], {
                    "default": "list_only",
                    "tooltip": "list_only: SAM2-compatible | dict_with_data: includes labels/scores"
                }),
                "output_masks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Convert bounding boxes to binary masks"
                }),
            }
        }

    RETURN_TYPES = ("BBOX", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("bboxes", "annotated_image", "labels", "masks")
    FUNCTION = "detect"
    CATEGORY = "grounding"

    def detect(self, model, image, prompt: str, confidence_threshold: float,
               text_threshold: float = 0.25, single_box_mode: bool = False,
               bbox_output_format: str = "list_only", output_masks: bool = False):
        """Universal detection that works with any model"""

        model_type = model["type"]

        # Route to appropriate detection method
        if model_type == "grounding_dino":
            return self._detect_grounding_dino(model, image, prompt, confidence_threshold,
                                              text_threshold, single_box_mode, bbox_output_format, output_masks)
        elif model_type == "owlv2":
            return self._detect_owlv2(model, image, prompt, confidence_threshold,
                                     single_box_mode, bbox_output_format, output_masks)
        elif model_type == "florence2":
            return self._detect_florence2(model, image, prompt, confidence_threshold,
                                         single_box_mode, bbox_output_format, output_masks)
        elif model_type == "yolo_world":
            return self._detect_yolo_world(model, image, prompt, confidence_threshold,
                                          single_box_mode, bbox_output_format, output_masks)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _detect_grounding_dino(self, model_dict, image, prompt, box_threshold, text_threshold,
                              single_box_mode, bbox_output_format, output_masks):
        """Detection using GroundingDINO/MM-GroundingDINO"""
        model = model_dict["model"]
        processor = model_dict["processor"]
        device = mm.get_torch_device()

        batch_size = image.shape[0]
        all_boxes = []
        all_labels = []
        all_scores = []
        annotated_images = []

        for i in range(batch_size):
            # Convert to PIL
            image_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            # Prepare inputs
            inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[pil_image.size[::-1]]
            )[0]

            # Extract results
            boxes = results["boxes"].cpu().numpy()
            labels = results["labels"]
            scores = results["scores"].cpu().numpy()

            # Override labels if no separators (REC mode)
            if '.' not in prompt and ',' not in prompt:
                labels = [prompt.strip()] * len(labels)

            # Single box mode
            if single_box_mode and len(boxes) > 0:
                top_idx = scores.argmax()
                boxes = boxes[top_idx:top_idx+1]
                labels = [labels[top_idx]]
                scores = scores[top_idx:top_idx+1]

            # Draw boxes
            annotated = self._draw_boxes(pil_image, boxes, labels, scores)
            annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)
            annotated_images.append(annotated_tensor)

        return self._format_output(all_boxes, all_labels, all_scores, annotated_images,
                                   image.shape, bbox_output_format, output_masks)

    def _detect_owlv2(self, model_dict, image, prompt, box_threshold,
                     single_box_mode, bbox_output_format, output_masks):
        """Detection using OWLv2"""
        model = model_dict["model"]
        processor = model_dict["processor"]
        device = mm.get_torch_device()

        batch_size = image.shape[0]
        all_boxes = []
        all_labels = []
        all_scores = []
        annotated_images = []

        for i in range(batch_size):
            # Convert to PIL
            image_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            # Parse text queries
            has_separator = '.' in prompt
            if has_separator:
                text_queries = [[c.strip() for c in prompt.split(".") if c.strip()]]
            else:
                text_queries = [[prompt.strip()]]

            # Prepare inputs
            inputs = processor(images=pil_image, text=text_queries, return_tensors="pt").to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs=outputs,
                threshold=box_threshold,
                target_sizes=target_sizes
            )[0]

            # Extract results
            boxes = results["boxes"].cpu().numpy()
            labels_indices = results["labels"].cpu().numpy()
            scores = results["scores"].cpu().numpy()

            # Map labels
            labels = [text_queries[0][idx] for idx in labels_indices]

            # Override labels if no separator
            if not has_separator and len(labels) > 0:
                labels = [prompt.strip()] * len(labels)

            # Single box mode
            if single_box_mode and len(boxes) > 0:
                top_idx = scores.argmax()
                boxes = boxes[top_idx:top_idx+1]
                labels = [labels[top_idx]]
                scores = scores[top_idx:top_idx+1]

            # Draw boxes
            annotated = self._draw_boxes(pil_image, boxes, labels, scores)
            annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)
            annotated_images.append(annotated_tensor)

        return self._format_output(all_boxes, all_labels, all_scores, annotated_images,
                                   image.shape, bbox_output_format, output_masks)

    def _detect_florence2(self, model_dict, image, prompt, box_threshold,
                         single_box_mode, bbox_output_format, output_masks):
        """Detection using Florence-2"""
        model = model_dict["model"]
        processor = model_dict["processor"]
        device = mm.get_torch_device()

        batch_size = image.shape[0]
        all_boxes = []
        all_labels = []
        all_scores = []
        annotated_images = []

        for i in range(batch_size):
            # Convert to PIL
            image_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            # Parse text queries
            has_separator = '.' in prompt
            if has_separator:
                text_queries = [c.strip() for c in prompt.split(".") if c.strip()]
                caption = " ".join(text_queries)
            else:
                caption = prompt.strip()
                text_queries = [caption]

            # Prepare task and inputs
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            inputs = processor(text=task_prompt + caption, images=pil_image, return_tensors="pt").to(device)

            # Run inference
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    use_cache=False,  # Prevent past_key_values issues
                )

            # Decode and parse
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=pil_image.size)

            # Extract bboxes (already in pixel coordinates!)
            result = parsed_answer.get(task_prompt, {})
            boxes = np.array(result.get("bboxes", []))
            labels_from_model = result.get("labels", [])

            # Assign labels
            if len(labels_from_model) == len(boxes) and all(labels_from_model):
                labels = labels_from_model
            else:
                labels = text_queries * (len(boxes) // max(len(text_queries), 1) + 1)
                labels = labels[:len(boxes)]

            # Create dummy scores
            scores = np.ones(len(boxes)) * 0.9

            # Override labels if no separator
            if not has_separator and len(labels) > 0:
                labels = [prompt.strip()] * len(labels)

            # Single box mode
            if single_box_mode and len(boxes) > 0:
                boxes = boxes[0:1]
                labels = [labels[0]]
                scores = scores[0:1]

            # Draw boxes
            annotated = self._draw_boxes(pil_image, boxes, labels, scores)
            annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)
            annotated_images.append(annotated_tensor)

        return self._format_output(all_boxes, all_labels, all_scores, annotated_images,
                                   image.shape, bbox_output_format, output_masks)

    def _detect_yolo_world(self, model_dict, image, classes, confidence_threshold,
                          single_box_mode, bbox_output_format, output_masks):
        """Detection using YOLO-World"""
        model = model_dict["model"]

        batch_size = image.shape[0]
        all_boxes = []
        all_labels = []
        all_scores = []
        annotated_images = []

        # Parse classes
        has_separator = ',' in classes
        if has_separator:
            class_list = [c.strip() for c in classes.split(",")]
        else:
            class_list = [classes.strip()]

        # Set classes for model
        model.set_classes(class_list)

        for i in range(batch_size):
            # Convert to numpy
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

            # Override labels if no separator
            if not has_separator and len(labels) > 0:
                labels = [classes.strip()] * len(labels)

            # Single box mode
            if single_box_mode and len(boxes) > 0:
                top_idx = scores.argmax()
                boxes = boxes[top_idx:top_idx+1]
                labels = [labels[top_idx]]
                scores = scores[top_idx:top_idx+1]

            # Draw boxes
            pil_image = Image.fromarray(image_np)
            annotated = self._draw_boxes(pil_image, boxes, labels, scores)
            annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)
            annotated_images.append(annotated_tensor)

        return self._format_output(all_boxes, all_labels, all_scores, annotated_images,
                                   image.shape, bbox_output_format, output_masks)

    def _draw_boxes(self, image, boxes, labels, scores):
        """Draw bounding boxes on image"""
        from PIL import ImageDraw

        image = image.copy()
        draw = ImageDraw.Draw(image)

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            text = f"{label}: {score:.2f}"
            draw.text((x1, max(y1 - 15, 0)), text, fill="red")

        return image

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
            masks = self._boxes_to_masks(all_boxes, image_shape)
        else:
            # Return empty mask
            batch_size, height, width, _ = image_shape
            masks = torch.zeros((1, height, width))

        return (batched_bboxes, annotated_batch, labels_str, masks)

    def _boxes_to_masks(self, all_boxes, image_shape):
        """Convert bounding boxes to binary masks

        Creates one mask per bounding box across all images in batch.
        Returns tensor of shape (N, H, W) where N is total number of boxes.
        """
        batch_size, height, width, _ = image_shape
        all_masks = []

        print(f"[Mask Creation] Image shape: {image_shape}")
        print(f"[Mask Creation] Number of images in batch: {len(all_boxes)}")

        for img_idx, boxes in enumerate(all_boxes):
            print(f"[Mask Creation] Image {img_idx}: {len(boxes)} boxes")

            # Create individual masks for each box in this image
            for box_idx, box in enumerate(boxes):
                mask = np.zeros((height, width), dtype=np.float32)
                x1, y1, x2, y2 = box.astype(int)

                print(f"  Box {box_idx}: [{x1}, {y1}, {x2}, {y2}]")

                # Clamp to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                print(f"  Clamped: [{x1}, {y1}, {x2}, {y2}]")

                # Check if box is valid
                if x2 > x1 and y2 > y1:
                    # Fill the box region with 1.0
                    mask[y1:y2, x1:x2] = 1.0
                    filled_pixels = (y2 - y1) * (x2 - x1)
                    print(f"  Filled {filled_pixels} pixels")
                else:
                    print(f"  WARNING: Invalid box dimensions!")

                all_masks.append(mask)

        # If no boxes detected, return single empty mask
        if len(all_masks) == 0:
            print("[Mask Creation] No boxes detected, returning empty mask")
            return torch.zeros((1, height, width))

        # Stack all masks
        masks_tensor = torch.from_numpy(np.stack(all_masks))
        print(f"[Mask Creation] Final masks shape: {masks_tensor.shape}")
        print(f"[Mask Creation] Masks min/max values: {masks_tensor.min()}/{masks_tensor.max()}")
        print(f"[Mask Creation] Non-zero pixels: {(masks_tensor > 0).sum().item()}")

        return masks_tensor


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
        """Draw bounding boxes on image (supports batches)"""
        from PIL import ImageDraw

        batch_size = image.shape[0]
        annotated_images = []

        for batch_idx in range(batch_size):
            # Convert to PIL
            image_np = (image[batch_idx].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            draw = ImageDraw.Draw(pil_image)

            # Handle both output formats
            if isinstance(bboxes, dict):
                # dict_with_data format
                boxes = bboxes["boxes"][batch_idx] if batch_idx < len(bboxes["boxes"]) else []
                labels = bboxes.get("labels", [[]])[batch_idx] if "labels" in bboxes and batch_idx < len(bboxes["labels"]) else []
                scores = bboxes.get("scores", [[]])[batch_idx] if "scores" in bboxes and batch_idx < len(bboxes["scores"]) else []
            elif isinstance(bboxes, list):
                # list_only format
                boxes = bboxes[batch_idx] if batch_idx < len(bboxes) else []
                labels = []
                scores = []
            else:
                # Fallback
                boxes = bboxes
                labels = []
                scores = []

            # Draw boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

                if i < len(labels) and i < len(scores):
                    label = labels[i]
                    score = scores[i]
                    text = f"{label}: {score:.2f}"
                    draw.text((x1, max(y1 - 10, 0)), text, fill="red")

            # Convert back to tensor
            annotated_np = np.array(pil_image).astype(np.float32) / 255.0
            annotated_tensor = torch.from_numpy(annotated_np)
            annotated_images.append(annotated_tensor)

        # Stack all annotated images into batch
        if len(annotated_images) > 1:
            result = torch.stack(annotated_images)
        else:
            result = annotated_images[0].unsqueeze(0)

        return (result,)


# ============================================================================
# SAM2 Nodes - Copied from comfyui-segment-anything-2
# ============================================================================

class DownloadAndLoadSAM2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ([
                    'sam2_hiera_base_plus.safetensors',
                    'sam2_hiera_large.safetensors',
                    'sam2_hiera_small.safetensors',
                    'sam2_hiera_tiny.safetensors',
                    'sam2.1_hiera_base_plus.safetensors',
                    'sam2.1_hiera_large.safetensors',
                    'sam2.1_hiera_small.safetensors',
                    'sam2.1_hiera_tiny.safetensors',
                    ],),
            "segmentor": (
                    ['single_image','video', 'automaskgenerator'],
                    ),
            "device": (['cuda', 'cpu', 'mps'], ),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),

            },
        }

    RETURN_TYPES = ("SAM2MODEL",)
    RETURN_NAMES = ("sam2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "SAM2"

    def loadmodel(self, model, segmentor, device, precision):
        if precision != 'fp32' and device == 'cpu':
            raise ValueError("fp16 and bf16 are not supported on cpu")

        if device == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        device = {"cuda": torch.device("cuda"), "cpu": torch.device("cpu"), "mps": torch.device("mps")}[device]

        # Create cache key for MODEL_CACHE
        cache_key = f"sam2_{model}_{segmentor}_{precision}_{device}"

        # Check cache
        if cache_key in MODEL_CACHE:
            print(f"✅ Loading SAM2 model from cache")
            return (MODEL_CACHE[cache_key],)

        download_path = os.path.join(folder_paths.models_dir, "sam2")
        if precision != 'fp32' and "2.1" in model:
            base_name, extension = model.rsplit('.', 1)
            model = f"{base_name}-fp16.{extension}"
        model_path = os.path.join(download_path, model)
        print("model_path: ", model_path)

        if not os.path.exists(model_path):
            print(f"Downloading SAM2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/sam2-safetensors",
                            allow_patterns=[f"*{model}*"],
                            local_dir=download_path,
                            local_dir_use_symlinks=False)

        model_mapping = {
            "2.0": {
                "base": "sam2_hiera_b+.yaml",
                "large": "sam2_hiera_l.yaml",
                "small": "sam2_hiera_s.yaml",
                "tiny": "sam2_hiera_t.yaml"
            },
            "2.1": {
                "base": "sam2.1_hiera_b+.yaml",
                "large": "sam2.1_hiera_l.yaml",
                "small": "sam2.1_hiera_s.yaml",
                "tiny": "sam2.1_hiera_t.yaml"
            }
        }
        version = "2.1" if "2.1" in model else "2.0"

        model_cfg_path = next(
            (os.path.join(script_directory, "sam2_configs", cfg)
            for key, cfg in model_mapping[version].items() if key in model),
            None
        )
        print(f"Using model config: {model_cfg_path}")

        model = load_model(model_path, model_cfg_path, segmentor, dtype, device)

        sam2_model = {
            'model': model,
            'dtype': dtype,
            'device': device,
            'segmentor' : segmentor,
            'version': version
            }

        # Cache the loaded model
        MODEL_CACHE[cache_key] = sam2_model
        print(f"💾 Cached SAM2 model in memory")

        return (sam2_model,)


class Sam2Segmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam2_model": ("SAM2MODEL", ),
                "image": ("IMAGE", ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "coordinates_negative": ("STRING", {"forceInput": True}),
                "bboxes": ("BBOX", ),
                "individual_objects": ("BOOLEAN", {"default": False}),
                "mask": ("MASK", ),
                "mask_threshold": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1, "tooltip": "Threshold for converting mask logits to binary. Positive=stricter, negative=looser"}),
                "max_hole_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Fill holes smaller than this area (0=disabled)"}),
                "max_sprinkle_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Remove isolated regions smaller than this area (0=disabled)"}),

            },
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES =("mask", )
    FUNCTION = "segment"
    CATEGORY = "SAM2"

    def segment(self, image, sam2_model, keep_model_loaded, coordinates_positive=None, coordinates_negative=None,
                individual_objects=False, bboxes=None, mask=None, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0):
        offload_device = mm.unet_offload_device()
        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]
        segmentor = sam2_model["segmentor"]
        B, H, W, C = image.shape

        if mask is not None:
            input_mask = mask.clone().unsqueeze(1)
            input_mask = F.interpolate(input_mask, size=(256, 256), mode="bilinear")
            input_mask = input_mask.squeeze(1)

        if segmentor == 'automaskgenerator':
            raise ValueError("For automaskgenerator use Sam2AutoMaskSegmentation -node")
        if segmentor == 'single_image' and B > 1:
            print("Segmenting batch of images with single_image segmentor")

        if segmentor == 'video' and bboxes is not None and "2.1" not in sam2_model["version"]:
            raise ValueError("2.0 model doesn't support bboxes with video segmentor")

        if segmentor == 'video': # video model needs images resized first thing
            model_input_image_size = model.image_size
            print("Resizing to model input image size: ", model_input_image_size)
            image = common_upscale(image.movedim(-1,1), model_input_image_size, model_input_image_size, "bilinear", "disabled").movedim(1,-1)

        #handle point coordinates
        if coordinates_positive is not None:
            try:
                coordinates_positive = json.loads(coordinates_positive.replace("'", '"'))
                coordinates_positive = [(coord['x'], coord['y']) for coord in coordinates_positive]
                if coordinates_negative is not None:
                    coordinates_negative = json.loads(coordinates_negative.replace("'", '"'))
                    coordinates_negative = [(coord['x'], coord['y']) for coord in coordinates_negative]
            except:
                pass

            if not individual_objects:
                positive_point_coords = np.atleast_2d(np.array(coordinates_positive))
            else:
                positive_point_coords = np.array([np.atleast_2d(coord) for coord in coordinates_positive])

            if coordinates_negative is not None:
                negative_point_coords = np.array(coordinates_negative)
                # Ensure both positive and negative coords are lists of 2D arrays if individual_objects is True
                if individual_objects:
                    assert negative_point_coords.shape[0] <= positive_point_coords.shape[0], "Can't have more negative than positive points in individual_objects mode"
                    if negative_point_coords.ndim == 2:
                        negative_point_coords = negative_point_coords[:, np.newaxis, :]
                    # Extend negative coordinates to match the number of positive coordinates
                    while negative_point_coords.shape[0] < positive_point_coords.shape[0]:
                        negative_point_coords = np.concatenate((negative_point_coords, negative_point_coords[:1, :, :]), axis=0)
                    final_coords = np.concatenate((positive_point_coords, negative_point_coords), axis=1)
                else:
                    final_coords = np.concatenate((positive_point_coords, negative_point_coords), axis=0)
            else:
                final_coords = positive_point_coords

        # Handle possible bboxes
        if bboxes is not None:
            boxes_np_batch = []
            for bbox_list in bboxes:
                boxes_np = []
                for bbox in bbox_list:
                    boxes_np.append(bbox)
                boxes_np = np.array(boxes_np)
                boxes_np_batch.append(boxes_np)
            if individual_objects:
                final_box = np.array(boxes_np_batch)
            else:
                final_box = boxes_np_batch
            final_labels = None

        #handle labels
        if coordinates_positive is not None:
            if not individual_objects:
                positive_point_labels = np.ones(len(positive_point_coords))
            else:
                positive_labels = []
                for point in positive_point_coords:
                    positive_labels.append(np.array([1])) # 1)
                positive_point_labels = np.stack(positive_labels, axis=0)

            if coordinates_negative is not None:
                if not individual_objects:
                    negative_point_labels = np.zeros(len(negative_point_coords))  # 0 = negative
                    final_labels = np.concatenate((positive_point_labels, negative_point_labels), axis=0)
                else:
                    negative_labels = []
                    for point in positive_point_coords:
                        negative_labels.append(np.array([0])) # 1)
                    negative_point_labels = np.stack(negative_labels, axis=0)
                    #combine labels
                    final_labels = np.concatenate((positive_point_labels, negative_point_labels), axis=1)
            else:
                final_labels = positive_point_labels
            print("combined labels: ", final_labels)
            print("combined labels shape: ", final_labels.shape)

        mask_list = []
        try:
            model.to(device)
        except:
            model.model.to(device)

        # Set mask post-processing parameters
        if hasattr(model, 'mask_threshold'):
            model.mask_threshold = mask_threshold
        if hasattr(model, 'max_hole_area'):
            model.max_hole_area = max_hole_area
        if hasattr(model, 'max_sprinkle_area'):
            model.max_sprinkle_area = max_sprinkle_area

        autocast_condition = not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            if segmentor == 'single_image':
                image_np = (image.contiguous() * 255).byte().numpy()
                comfy_pbar = ProgressBar(len(image_np))
                tqdm_pbar = tqdm(total=len(image_np), desc="Processing Images")
                for i in range(len(image_np)):
                    model.set_image(image_np[i])
                    if bboxes is None:
                        input_box = None
                    else:
                        input_box = final_box[i] if i < len(final_box) else final_box[0]

                    out_masks, scores, logits = model.predict(
                        point_coords=final_coords if coordinates_positive is not None else None,
                        point_labels=final_labels if coordinates_positive is not None else None,
                        box=input_box,
                        multimask_output=True if not individual_objects else False,
                        mask_input = input_mask[i].unsqueeze(0) if mask is not None else None,
                        )

                    if out_masks.ndim == 3:
                        sorted_ind = np.argsort(scores)[::-1]
                        out_masks = out_masks[sorted_ind][0] #choose only the best result for now
                        scores = scores[sorted_ind]
                        logits = logits[sorted_ind]
                        mask_list.append(np.expand_dims(out_masks, axis=0))
                    else:
                        _, _, H, W = out_masks.shape
                        # Combine masks for all object IDs in the frame
                        combined_mask = np.zeros((H, W), dtype=bool)
                        for out_mask in out_masks:
                            combined_mask = np.logical_or(combined_mask, out_mask)
                        combined_mask = combined_mask.astype(np.uint8)
                        mask_list.append(combined_mask)
                    comfy_pbar.update(1)
                    tqdm_pbar.update(1)

            elif segmentor == 'video':
                mask_list = []
                if hasattr(self, 'inference_state') and self.inference_state is not None:
                    model.reset_state(self.inference_state)
                self.inference_state = model.init_state(image.permute(0, 3, 1, 2).contiguous(), H, W, device=device)
                if bboxes is None:
                        input_box = None
                else:
                    input_box = bboxes[0]

                if individual_objects and bboxes is not None:
                    raise ValueError("bboxes not supported with individual_objects")


                if individual_objects:
                    for i, (coord, label) in enumerate(zip(final_coords, final_labels)):
                        _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=i,
                        points=final_coords[i],
                        labels=final_labels[i],
                        clear_old_points=True,
                        box=input_box
                        )
                else:
                    _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=1,
                        points=final_coords if coordinates_positive is not None else None,
                        labels=final_labels if coordinates_positive is not None else None,
                        clear_old_points=True,
                        box=input_box
                    )

                pbar = ProgressBar(B)
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(self.inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    pbar.update(1)
                    if individual_objects:
                        _, _, H, W = out_mask_logits.shape
                        # Combine masks for all object IDs in the frame
                        combined_mask = np.zeros((H, W), dtype=np.uint8)
                        for i, out_obj_id in enumerate(out_obj_ids):
                            out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                            combined_mask = np.logical_or(combined_mask, out_mask)
                        video_segments[out_frame_idx] = combined_mask

                if individual_objects:
                    for frame_idx, combined_mask in video_segments.items():
                        mask_list.append(combined_mask)
                else:
                    for frame_idx, obj_masks in video_segments.items():
                        for out_obj_id, out_mask in obj_masks.items():
                            mask_list.append(out_mask)

        if not keep_model_loaded:
            try:
                model.to(offload_device)
            except:
                model.model.to(offload_device)
            if hasattr(self, 'inference_state') and self.inference_state is not None and hasattr(model, "reset_state"):
                model.reset_state(self.inference_state)
                self.inference_state = None
            mm.soft_empty_cache()

        out_list = []
        for mask in mask_list:
            mask_tensor = torch.from_numpy(mask)
            mask_tensor = mask_tensor.permute(1, 2, 0)
            mask_tensor = mask_tensor[:, :, 0]
            out_list.append(mask_tensor)
        mask_tensor = torch.stack(out_list, dim=0).cpu().float()
        return (mask_tensor,)
