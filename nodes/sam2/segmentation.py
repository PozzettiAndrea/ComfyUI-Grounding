"""
SAM2 segmentation logic
"""
import json
import torch
import numpy as np
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm import tqdm
import comfy.model_management as mm
from comfy.utils import ProgressBar, common_upscale


def segment_sam2(image, sam2_model, keep_model_loaded, coordinates_positive=None,
                 coordinates_negative=None, individual_objects=False, bboxes=None,
                 mask=None, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0,
                 inference_state_ref=None):
    """Perform SAM2 segmentation

    Args:
        image: ComfyUI IMAGE tensor (B, H, W, C)
        sam2_model: Dict containing model, dtype, device, segmentor, version
        keep_model_loaded: Whether to keep model in memory
        coordinates_positive: JSON string of positive point coordinates
        coordinates_negative: JSON string of negative point coordinates
        individual_objects: Whether to segment each object individually
        bboxes: Bounding boxes input
        mask: Input mask for refinement
        mask_threshold: Threshold for mask binarization
        max_hole_area: Fill holes smaller than this
        max_sprinkle_area: Remove isolated regions smaller than this
        inference_state_ref: Reference to inference state (for video segmentor)

    Returns:
        Tuple of (mask_tensor,)
    """
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

    if segmentor == 'video':  # video model needs images resized first thing
        model_input_image_size = model.image_size
        print("Resizing to model input image size: ", model_input_image_size)
        image = common_upscale(image.movedim(-1,1), model_input_image_size,
                              model_input_image_size, "bilinear", "disabled").movedim(1,-1)

    # Handle point coordinates
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
                assert negative_point_coords.shape[0] <= positive_point_coords.shape[0], \
                    "Can't have more negative than positive points in individual_objects mode"
                if negative_point_coords.ndim == 2:
                    negative_point_coords = negative_point_coords[:, np.newaxis, :]
                # Extend negative coordinates to match the number of positive coordinates
                while negative_point_coords.shape[0] < positive_point_coords.shape[0]:
                    negative_point_coords = np.concatenate((negative_point_coords,
                                                           negative_point_coords[:1, :, :]), axis=0)
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

    # Handle labels
    if coordinates_positive is not None:
        if not individual_objects:
            positive_point_labels = np.ones(len(positive_point_coords))
        else:
            positive_labels = []
            for point in positive_point_coords:
                positive_labels.append(np.array([1]))
            positive_point_labels = np.stack(positive_labels, axis=0)

        if coordinates_negative is not None:
            if not individual_objects:
                negative_point_labels = np.zeros(len(negative_point_coords))  # 0 = negative
                final_labels = np.concatenate((positive_point_labels, negative_point_labels), axis=0)
            else:
                negative_labels = []
                for point in positive_point_coords:
                    negative_labels.append(np.array([0]))
                negative_point_labels = np.stack(negative_labels, axis=0)
                # Combine labels
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
                    mask_input=input_mask[i].unsqueeze(0) if mask is not None else None,
                )

                if out_masks.ndim == 3:
                    sorted_ind = np.argsort(scores)[::-1]
                    out_masks = out_masks[sorted_ind][0]  # choose only the best result for now
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
            if inference_state_ref is not None and inference_state_ref.get('state') is not None:
                model.reset_state(inference_state_ref['state'])

            inference_state = model.init_state(image.permute(0, 3, 1, 2).contiguous(), H, W, device=device)
            if inference_state_ref is not None:
                inference_state_ref['state'] = inference_state

            if bboxes is None:
                input_box = None
            else:
                input_box = bboxes[0]

            if individual_objects and bboxes is not None:
                raise ValueError("bboxes not supported with individual_objects")

            if individual_objects:
                for i, (coord, label) in enumerate(zip(final_coords, final_labels)):
                    _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=i,
                        points=final_coords[i],
                        labels=final_labels[i],
                        clear_old_points=True,
                        box=input_box
                    )
            else:
                _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=final_coords if coordinates_positive is not None else None,
                    labels=final_labels if coordinates_positive is not None else None,
                    clear_old_points=True,
                    box=input_box
                )

            pbar = ProgressBar(B)
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(inference_state):
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
        if inference_state_ref is not None and inference_state_ref.get('state') is not None and hasattr(model, "reset_state"):
            model.reset_state(inference_state_ref['state'])
            inference_state_ref['state'] = None
        mm.soft_empty_cache()

    out_list = []
    for mask in mask_list:
        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.permute(1, 2, 0)
        mask_tensor = mask_tensor[:, :, 0]
        out_list.append(mask_tensor)
    mask_tensor = torch.stack(out_list, dim=0).cpu().float()
    return (mask_tensor,)
