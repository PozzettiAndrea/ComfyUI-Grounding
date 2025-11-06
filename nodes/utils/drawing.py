"""
Drawing utilities for visualizing detection and segmentation results
"""
import numpy as np
import random
from PIL import ImageDraw


def draw_boxes(image, boxes, labels, scores):
    """Draw bounding boxes on image

    Args:
        image: PIL Image
        boxes: Array of boxes in xyxy format
        labels: List of label strings
        scores: Array of confidence scores

    Returns:
        PIL Image with boxes drawn
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        text = f"{label}: {score:.2f}"
        draw.text((x1, max(y1 - 15, 0)), text, fill="red")

    return image


def draw_segmentation(image_np, masks, labels):
    """Draw segmentation masks on image

    Args:
        image_np: Numpy array (H, W, C) uint8
        masks: List of binary masks (H, W) float32
        labels: List of label strings

    Returns:
        Numpy array (H, W, C) uint8 with masks overlaid
    """
    annotated = image_np.copy()

    # Generate colors for each mask
    for i, (mask, label) in enumerate(zip(masks, labels)):
        # Generate random color
        color = np.array([
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ], dtype=np.uint8)

        # Apply colored mask with transparency
        mask_3d = np.stack([mask] * 3, axis=-1)
        colored_mask = mask_3d * color
        annotated = np.where(mask_3d > 0,
                            annotated * 0.6 + colored_mask * 0.4,
                            annotated).astype(np.uint8)

    return annotated
