"""
Format conversion utilities for bounding boxes and masks
"""
import torch
import numpy as np
from PIL import Image, ImageDraw


def boxes_to_masks(all_boxes, image_shape):
    """Convert bounding boxes to binary masks

    Creates one mask per bounding box across all images in batch.
    Returns tensor of shape (N, H, W) where N is total number of boxes.

    Args:
        all_boxes: List of box arrays, one per image
        image_shape: Tuple (B, H, W, C)

    Returns:
        Torch tensor of masks (N, H, W)
    """
    batch_size, height, width, _ = image_shape
    all_masks = []

    for img_idx, boxes in enumerate(all_boxes):
        # Create individual masks for each box in this image
        for box_idx, box in enumerate(boxes):
            mask = np.zeros((height, width), dtype=np.float32)
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Check if box is valid
            if x2 > x1 and y2 > y1:
                # Fill the box region with 1.0
                mask[y1:y2, x1:x2] = 1.0

            all_masks.append(mask)

    # If no boxes detected, return empty masks (one per image in batch)
    if len(all_masks) == 0:
        return torch.zeros((batch_size, height, width))

    # Stack all masks
    masks_tensor = torch.from_numpy(np.stack(all_masks))
    return masks_tensor


def polygon_to_mask(polygon, image_shape):
    """Convert polygon coordinates to binary mask

    Args:
        polygon: Flat list [x0, y0, x1, y1, ...]
        image_shape: Tuple (H, W)

    Returns:
        Numpy array (H, W) float32
    """
    height, width = image_shape
    mask = Image.new('L', (width, height), 0)

    # Polygon is a flat list [x0, y0, x1, y1, ...]
    # Convert to list of tuples [(x0, y0), (x1, y1), ...]
    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]

    ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)

    return np.array(mask, dtype=np.float32)


def mask_to_bbox(mask):
    """Extract bounding box from binary mask

    Args:
        mask: Binary mask (H, W)

    Returns:
        List [x1, y1, x2, y2] in xyxy format
    """
    # Find coordinates where mask is non-zero
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        # Empty mask
        return [0, 0, 0, 0]

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Return in xyxy format
    return [float(cmin), float(rmin), float(cmax), float(rmax)]
