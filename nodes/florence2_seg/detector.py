"""
Florence-2 Segmentation detection logic
"""
import torch
import numpy as np
from PIL import Image
import comfy.model_management as mm
from ..utils import polygon_to_mask, mask_to_bbox, draw_segmentation


def detect_florence2_seg(model_dict, image, prompt, confidence_threshold,
                         florence2_max_tokens, florence2_num_beams, format_output_fn):
    """Detection using Florence-2 segmentation

    Args:
        model_dict: Dict with model, processor, type, framework
        image: ComfyUI IMAGE tensor (B, H, W, C)
        prompt: Text prompt string (use . separator for multiple objects)
        confidence_threshold: Not used by Florence-2
        florence2_max_tokens: Max tokens for generation
        florence2_num_beams: Beam search width
        format_output_fn: Function to format outputs

    Returns:
        Tuple of (masks, boxes, labels, annotated_image, text)
    """
    model = model_dict["model"]
    processor = model_dict["processor"]
    device = mm.get_torch_device()

    batch_size = image.shape[0]
    all_masks = []
    all_boxes = []
    all_labels = []
    annotated_images = []

    for i in range(batch_size):
        # Convert to PIL
        image_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Parse text queries
        has_separator = '.' in prompt
        if has_separator:
            text_queries = [c.strip() for c in prompt.split(".") if c.strip()]
            # Format for Florence-2: "phrase1. phrase2. phrase3."
            caption = ". ".join(text_queries) + "."
        else:
            caption = prompt.strip()
            text_queries = [caption]

        # Prepare task and inputs for segmentation
        task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
        inputs = processor(text=task_prompt + caption, images=pil_image, return_tensors="pt").to(device)

        # Use generation parameters
        max_new_tokens = florence2_max_tokens
        num_beams = florence2_num_beams

        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                use_cache=False,
            )

        # Decode output
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_image.width, pil_image.height)
        )

        # Extract polygons and labels
        if task_prompt in parsed and "polygons" in parsed[task_prompt]:
            polygons_data = parsed[task_prompt]["polygons"]
            labels_data = parsed[task_prompt].get("labels", [])

            image_masks = []
            image_boxes = []
            image_labels = []

            for poly_list, label in zip(polygons_data, labels_data):
                # Each poly_list contains one or more polygons for this object
                for polygon in poly_list:
                    # Convert polygon to mask
                    mask = polygon_to_mask(polygon, image_np.shape[:2])

                    # Extract bbox from mask
                    bbox = mask_to_bbox(mask)

                    image_masks.append(mask)
                    image_boxes.append(bbox)
                    image_labels.append(label)

            all_masks.append(image_masks)
            all_boxes.append(image_boxes)
            all_labels.append(image_labels)
        else:
            # No detections
            all_masks.append([])
            all_boxes.append([])
            all_labels.append([])

        # Create annotated image
        annotated_image_np = draw_segmentation(image_np, all_masks[i], all_labels[i])
        annotated_tensor = torch.from_numpy(annotated_image_np).float() / 255.0
        # Add batch dimension for ComfyUI IMAGE format (B, H, W, C)
        annotated_tensor = annotated_tensor.unsqueeze(0)
        annotated_images.append(annotated_tensor)

    # Format outputs (no text output for Florence-2 seg)
    return format_output_fn(all_masks, all_boxes, all_labels, annotated_images, None)
