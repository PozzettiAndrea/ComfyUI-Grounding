"""
SA2VA detection logic
"""
import torch
import numpy as np
from PIL import Image
import comfy.model_management as mm
from ..utils import mask_to_bbox, draw_segmentation


def detect_sa2va(model_dict, image, prompt, confidence_threshold,
                 sa2va_max_tokens, sa2va_num_beams, format_output_fn):
    """Detection using SA2VA vision-language segmentation

    Args:
        model_dict: Dict with model, tokenizer, type, framework
        image: ComfyUI IMAGE tensor (B, H, W, C)
        prompt: Text prompt string
        confidence_threshold: Not used by SA2VA
        sa2va_max_tokens: Max tokens for generation
        sa2va_num_beams: Beam search width
        format_output_fn: Function to format outputs

    Returns:
        Tuple of (masks, boxes, labels, annotated_image, text)
    """
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = mm.get_torch_device()

    # Configure generation parameters
    if hasattr(model, 'gen_config'):
        model.gen_config.max_new_tokens = sa2va_max_tokens
        model.gen_config.num_beams = sa2va_num_beams
        model.gen_config.do_sample = False  # Deterministic generation

    batch_size = image.shape[0]
    all_masks = []
    all_boxes = []
    all_labels = []
    all_texts = []
    annotated_images = []

    for i in range(batch_size):
        # Convert ComfyUI tensor to PIL
        image_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Format prompt - SA2VA requires <image> token
        if '<image>' not in prompt:
            sa2va_prompt = f"<image>{prompt}"
        else:
            sa2va_prompt = prompt

        print(f"🎨 SA2VA inference with prompt: {sa2va_prompt}")

        # Run inference using SA2VA's predict_forward method
        result = model.predict_forward(
            image=pil_image,
            text=sa2va_prompt,
            tokenizer=tokenizer
        )

        # Extract masks and prediction text
        image_masks = []
        image_boxes = []
        image_labels = []

        prediction_text = result.get('prediction', '')
        prediction_masks = result.get('prediction_masks', [])

        print(f"📝 SA2VA prediction: {prediction_text}")
        print(f"🎭 SA2VA generated {len(prediction_masks)} mask(s)")

        # Store the prediction text
        all_texts.append(prediction_text)

        # Check if [SEG] token was generated
        seg_count = prediction_text.count('[SEG]')
        if seg_count == 0:
            print("⚠️  WARNING: SA2VA did not generate any [SEG] tokens!")
            print("💡 TIP: Try prompts like 'Segment the person' or 'Please segment all objects in the image'")
            print("💡 SA2VA needs explicit segmentation instructions, not just object descriptions")

        if prediction_masks and len(prediction_masks) > 0:
            for mask_idx, mask in enumerate(prediction_masks):
                print(f"  Mask {mask_idx}: Raw shape {mask.shape}, dtype {mask.dtype}")

                # Handle video masks (take first frame if 3D)
                if mask.ndim == 3:
                    print(f"  Mask {mask_idx}: Video mask, taking first frame")
                    mask = mask[0]

                # Convert boolean to float32 if needed
                if mask.dtype == bool:
                    mask = mask.astype(np.float32)

                # Ensure mask is 2D and has correct shape
                if mask.ndim == 1:
                    # If 1D, try to reshape to image dimensions
                    h, w = image_np.shape[:2]
                    if mask.shape[0] == h * w:
                        mask = mask.reshape(h, w)
                        print(f"  Mask {mask_idx}: Reshaped from 1D to ({h}, {w})")
                    else:
                        print(f"  ⚠️ WARNING: 1D mask size {mask.shape[0]} doesn't match image size {h}x{w}, skipping")
                        continue
                elif mask.ndim > 2:
                    # Squeeze extra dimensions but ensure we keep 2D
                    while mask.ndim > 2 and (mask.shape[0] == 1 or mask.shape[-1] == 1):
                        mask = mask.squeeze()
                    if mask.ndim != 2:
                        print(f"  ⚠️ WARNING: Cannot reduce mask to 2D, shape is {mask.shape}, skipping")
                        continue

                # Verify mask shape matches image
                h, w = image_np.shape[:2]
                if mask.shape != (h, w):
                    print(f"  ⚠️ WARNING: Mask shape {mask.shape} doesn't match image shape ({h}, {w})")
                    # Try to resize mask to match image using torch
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor,
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    )
                    mask = mask_tensor.squeeze().cpu().numpy()
                    print(f"  Resized mask to {mask.shape}")

                print(f"  Mask {mask_idx}: Final shape {mask.shape}, dtype {mask.dtype}")

                # Extract bbox from mask
                bbox = mask_to_bbox(mask)

                image_masks.append(mask)
                image_boxes.append(bbox)

                # Parse labels from prediction text
                # Count [SEG] tokens to assign labels
                seg_count = prediction_text.count('[SEG]')
                if mask_idx < seg_count:
                    # Try to extract object name from context around [SEG] token
                    label = f"object_{mask_idx}"
                else:
                    label = f"object_{mask_idx}"

                image_labels.append(label)

        all_masks.append(image_masks)
        all_boxes.append(image_boxes)
        all_labels.append(image_labels)

        # Create annotated image
        annotated_image_np = draw_segmentation(image_np, image_masks, image_labels)
        annotated_tensor = torch.from_numpy(annotated_image_np).float() / 255.0
        # Add batch dimension for ComfyUI IMAGE format (B, H, W, C)
        annotated_tensor = annotated_tensor.unsqueeze(0)
        annotated_images.append(annotated_tensor)

    # Format outputs
    return format_output_fn(all_masks, all_boxes, all_labels, annotated_images, all_texts)
