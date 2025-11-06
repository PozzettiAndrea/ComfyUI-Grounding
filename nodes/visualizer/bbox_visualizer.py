"""
BboxVisualizer logic
"""
import torch
import numpy as np
from PIL import Image, ImageDraw


def visualize_bboxes(image, bboxes, line_width=3):
    """Draw bounding boxes on image (supports batches)

    Args:
        image: ComfyUI IMAGE tensor (B, H, W, C)
        bboxes: BBOX data (dict_with_data or list_only format)
        line_width: Width of bbox lines

    Returns:
        Tuple of (annotated_image,)
    """
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
