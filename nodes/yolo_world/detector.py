"""
YOLO-World detection logic
"""
import torch
import numpy as np
from PIL import Image
from ..utils import draw_boxes


def detect_yolo_world(model_dict, image, classes, confidence_threshold,
                      single_box_mode, single_box_per_prompt_mode, bbox_output_format, output_masks,
                      yolo_iou, yolo_agnostic_nms, yolo_max_det, format_output_fn):
    """Detection using YOLO-World

    Args:
        model_dict: Dict with model, type, framework
        image: ComfyUI IMAGE tensor (B, H, W, C)
        classes: Text prompt string (period-separated for multiple objects)
        confidence_threshold: Confidence threshold
        single_box_mode: Return only highest confidence box
        single_box_per_prompt_mode: Return highest confidence box per label
        bbox_output_format: Format for bbox output
        output_masks: Whether to output masks from boxes
        yolo_iou: IoU threshold for NMS
        yolo_agnostic_nms: Class-agnostic NMS
        yolo_max_det: Max detections per image
        format_output_fn: Function to format outputs

    Returns:
        Tuple of (annotated_image, boxes, labels, scores, masks)
    """
    model = model_dict["model"]

    batch_size = image.shape[0]
    all_boxes = []
    all_labels = []
    all_scores = []
    annotated_images = []

    # Parse classes
    # Period (.) = multiple objects, comma or no separator = single object
    has_separator = '.' in classes
    if has_separator:
        class_list = [c.strip() for c in classes.split(".") if c.strip()]
    else:
        # Comma or no separator = treat as single label
        class_list = [classes.strip()]

    # Set classes for model
    model.set_classes(class_list)

    for i in range(batch_size):
        # Convert to numpy
        image_np = (image[i].cpu().numpy() * 255).astype(np.uint8)

        # Run inference with custom parameters
        results = model(
            image_np,
            conf=confidence_threshold,
            iou=yolo_iou,
            agnostic_nms=yolo_agnostic_nms,
            max_det=yolo_max_det
        )

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

        # Single box mode (takes precedence)
        if single_box_mode and len(boxes) > 0:
            top_idx = scores.argmax()
            boxes = boxes[top_idx:top_idx+1]
            labels = [labels[top_idx]]
            scores = scores[top_idx:top_idx+1]
        # Single box per prompt mode (only if single_box_mode is False)
        elif single_box_per_prompt_mode and len(boxes) > 0:
            # Group boxes by label and keep highest scoring box per label
            label_groups = {}
            for idx, label in enumerate(labels):
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(idx)

            # Keep highest scoring box for each label
            keep_indices = []
            for label, indices in label_groups.items():
                best_idx = max(indices, key=lambda i: scores[i])
                keep_indices.append(best_idx)

            keep_indices = sorted(keep_indices)
            boxes = boxes[keep_indices]
            labels = [labels[i] for i in keep_indices]
            scores = scores[keep_indices]

        # Draw boxes
        pil_image = Image.fromarray(image_np)
        annotated = draw_boxes(pil_image, boxes, labels, scores)
        annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

        all_boxes.append(boxes)
        all_labels.append(labels)
        all_scores.append(scores)
        annotated_images.append(annotated_tensor)

    return format_output_fn(all_boxes, all_labels, all_scores, annotated_images,
                           image.shape, bbox_output_format, output_masks)
