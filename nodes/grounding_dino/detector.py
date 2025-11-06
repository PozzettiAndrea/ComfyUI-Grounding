"""
GroundingDINO detection logic
"""
import torch
import numpy as np
from PIL import Image
import comfy.model_management as mm
from ..utils import draw_boxes


def detect_grounding_dino(model_dict, image, prompt, box_threshold, text_threshold,
                          single_box_mode, bbox_output_format, output_masks, format_output_fn):
    """Detection using GroundingDINO/MM-GroundingDINO

    Args:
        model_dict: Dict with model, processor, type, framework
        image: ComfyUI IMAGE tensor (B, H, W, C)
        prompt: Text prompt string
        box_threshold: Box confidence threshold
        text_threshold: Text confidence threshold
        single_box_mode: Return only highest confidence box
        bbox_output_format: Format for bbox output
        output_masks: Whether to output masks from boxes
        format_output_fn: Function to format outputs

    Returns:
        Tuple of (annotated_image, boxes, labels, scores, masks)
    """
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

        # Parse text queries for label mapping
        has_separator = '.' in prompt or ',' in prompt
        if has_separator:
            # Split by period or comma
            if '.' in prompt:
                text_queries = [c.strip() for c in prompt.split(".") if c.strip()]
            else:
                text_queries = [c.strip() for c in prompt.split(",") if c.strip()]
        else:
            text_queries = [prompt.strip()]

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
        labels_from_model = results["labels"]
        scores = results["scores"].cpu().numpy()

        # Use labels from model if they exist and are non-empty, otherwise use text_queries
        if labels_from_model and all(label.strip() for label in labels_from_model):
            labels = labels_from_model
        else:
            # Fall back to assigning text queries to boxes
            labels = []
            for idx in range(len(boxes)):
                # Cycle through text_queries if we have more boxes than queries
                labels.append(text_queries[idx % len(text_queries)])

        # Override labels if no separators (REC mode)
        if not has_separator:
            labels = [prompt.strip()] * len(labels)

        # Single box mode
        if single_box_mode and len(boxes) > 0:
            top_idx = scores.argmax()
            boxes = boxes[top_idx:top_idx+1]
            labels = [labels[top_idx]]
            scores = scores[top_idx:top_idx+1]

        # Draw boxes
        annotated = draw_boxes(pil_image, boxes, labels, scores)
        annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

        all_boxes.append(boxes)
        all_labels.append(labels)
        all_scores.append(scores)
        annotated_images.append(annotated_tensor)

    return format_output_fn(all_boxes, all_labels, all_scores, annotated_images,
                           image.shape, bbox_output_format, output_masks)
