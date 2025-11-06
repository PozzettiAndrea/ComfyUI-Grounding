"""
Florence-2 bbox detection logic
"""
import torch
import numpy as np
from PIL import Image
import comfy.model_management as mm
from ..utils import draw_boxes


def detect_florence2(model_dict, image, prompt, box_threshold,
                     single_box_mode, bbox_output_format, output_masks,
                     florence2_max_tokens, florence2_num_beams, format_output_fn):
    """Detection using Florence-2

    Args:
        model_dict: Dict with model, processor, type, framework
        image: ComfyUI IMAGE tensor (B, H, W, C)
        prompt: Text prompt string
        box_threshold: Not used by Florence-2
        single_box_mode: Return only first box
        bbox_output_format: Format for bbox output
        output_masks: Whether to output masks from boxes
        florence2_max_tokens: Max tokens for generation
        florence2_num_beams: Beam search width
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

        # Parse text queries
        has_separator = '.' in prompt
        if has_separator:
            text_queries = [c.strip() for c in prompt.split(".") if c.strip()]
            caption = ". ".join(text_queries)  # Preserve period separator
        else:
            caption = prompt.strip()
            text_queries = [caption]

        # Prepare task and inputs
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = processor(text=task_prompt + caption, images=pil_image, return_tensors="pt").to(device)

        # Use generation parameters passed from detector
        max_new_tokens = florence2_max_tokens
        num_beams = florence2_num_beams

        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
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
        annotated = draw_boxes(pil_image, boxes, labels, scores)
        annotated_tensor = torch.from_numpy(np.array(annotated).astype(np.float32) / 255.0).unsqueeze(0)

        all_boxes.append(boxes)
        all_labels.append(labels)
        all_scores.append(scores)
        annotated_images.append(annotated_tensor)

    return format_output_fn(all_boxes, all_labels, all_scores, annotated_images,
                           image.shape, bbox_output_format, output_masks)
