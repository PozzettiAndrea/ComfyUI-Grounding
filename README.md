# ComfyUI-Grounding

ComfyUI custom nodes for open-vocabulary object detection using state-of-the-art grounding models.

## Features

- **GroundingDINO**: Open-vocabulary object detection with natural language prompts
- **YOLO-World**: Real-time open-vocabulary detection
- **Batch Processing**: Process multiple images at once for improved efficiency
- **Referring Expression Comprehension (REC)**: Use full text as label when no separators present
- **Flexible Architecture**: Support for both local and HuggingFace implementations
- **Utility Nodes**: Bounding box visualization and mask conversion

## Supported Models

### GroundingDINO (Original)
- GroundingDINO-T (SwinT backbone, 694MB)
- GroundingDINO-B (SwinB backbone, 938MB)

### MM-GroundingDINO (Multimodal)
**Tiny Models (Swin-T backbone):**
- MM-GroundingDINO-Tiny (O365+GoldG, 50.4 mAP)
- MM-GroundingDINO-Tiny (O365+GoldG+GRIT, 50.5 mAP)
- MM-GroundingDINO-Tiny (O365+GoldG+V3Det, 50.6 mAP)
- MM-GroundingDINO-Tiny (O365+GoldG+GRIT+V3Det, 50.4 mAP)

**Base Models (Swin-B backbone):**
- MM-GroundingDINO-Base (O365+GoldG+V3Det, 52.5 mAP)
- MM-GroundingDINO-Base (All datasets, 59.5 mAP) ⭐ Recommended

**Large Models (Swin-L backbone):**
- MM-GroundingDINO-Large (O365v2+OIv6+GoldG, 53.0 mAP)
- MM-GroundingDINO-Large (All datasets, 60.3 mAP) ⭐ Best accuracy

### YOLO-World
- YOLOv8s-World (small)
- YOLOv8m-World (medium)
- YOLOv8l-World (large)
- YOLOv8x-World (extra large)

## Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
```

2. Clone or copy this repository:
```bash
git clone <repository-url> ComfyUI-Grounding
```

3. Install dependencies:
```bash
cd ComfyUI-Grounding
pip install -r requirements.txt
```

## Nodes

### 1. GroundingDINOModelLoader
Loads a GroundingDINO or MM-GroundingDINO model for object detection.

**Inputs:**
- `model_name`: Select from available GroundingDINO/MM-GroundingDINO models
- `keep_in_memory`: Cache model in memory for faster reuse (default: True)

**Outputs:**
- `model`: Loaded GroundingDINO model

**Features:**
- ✅ Model caching - loads instantly on second use
- ✅ Local model storage in `ComfyUI-Grounding/Models/`
- ✅ Supports both original and MM-GroundingDINO variants

### 2. GroundingDINODetector
Detects objects in images using text prompts.

**Inputs:**
- `model`: GroundingDINO model (from loader)
- `image`: Input image (supports batches)
- `prompt`: Text description of objects to detect (e.g., "person . car . dog .")
- `box_threshold`: Confidence threshold for bounding boxes (0.0-1.0)
- `text_threshold`: Confidence threshold for text matching (0.0-1.0)
- `single_box_mode`: (Optional) Return only highest-scoring detection for REC
- `output_format`: (Optional) "list_only" (default, SAM2-compatible) or "dict_with_data"

**Outputs:**
- `bboxes`: Detected bounding boxes (format depends on output_format setting)
  - `list_only`: List of numpy arrays `[[boxes_img0], [boxes_img1], ...]` - SAM2-compatible
  - `dict_with_data`: Dict `{"boxes": [...], "labels": [...], "scores": [...]}`
- `annotated_image`: Image with drawn bounding boxes (batched)
- `labels`: String with detected object labels

**Features:**
- ✅ Batch processing - process multiple images at once
- ✅ Automatic REC mode - when prompt has no separators (. or ,), uses full text as label
- ✅ Single box mode - returns only highest-scoring detection for referring expressions
- ✅ Flexible output format - choose between SAM2-compatible list or full dict with metadata

### 3. YOLOWorldModelLoader
Loads a YOLO-World model for real-time detection.

**Inputs:**
- `model_name`: Select from available YOLO-World models
- `keep_in_memory`: Cache model in memory for faster reuse (default: True)

**Outputs:**
- `model`: Loaded YOLO-World model

**Features:**
- ✅ Model caching - loads instantly on second use
- ✅ Local model storage in `ComfyUI-Grounding/Models/`
- ✅ Auto-downloads models on first use

### 4. YOLOWorldDetector
Detects objects using YOLO-World with custom classes.

**Inputs:**
- `model`: YOLO-World model (from loader)
- `image`: Input image (supports batches)
- `classes`: Comma-separated list of classes (e.g., "person, car, dog")
- `confidence_threshold`: Minimum confidence for detections (0.0-1.0)
- `single_box_mode`: (Optional) Return only highest-scoring detection for REC
- `output_format`: (Optional) "list_only" (default, SAM2-compatible) or "dict_with_data"

**Outputs:**
- `bboxes`: Detected bounding boxes (format depends on output_format setting)
  - `list_only`: List of numpy arrays `[[boxes_img0], [boxes_img1], ...]` - SAM2-compatible
  - `dict_with_data`: Dict `{"boxes": [...], "labels": [...], "scores": [...]}`
- `annotated_image`: Image with drawn bounding boxes (batched)
- `labels`: String with detected object labels

**Features:**
- ✅ Batch processing - process multiple images at once
- ✅ Automatic REC mode - when classes has no comma, uses full text as label
- ✅ Single box mode - returns only highest-scoring detection for referring expressions
- ✅ Flexible output format - choose between SAM2-compatible list or full dict with metadata

### 5. BboxVisualizer
Draws bounding boxes on images.

**Inputs:**
- `image`: Input image
- `bboxes`: Bounding box data
- `line_width`: Thickness of bounding box lines (optional)

**Outputs:**
- `annotated_image`: Image with drawn bounding boxes

### 5. UnifiedDetector ⭐ NEW
Universal detector that works with BOTH GroundingDINO and YOLO-World models.

**Inputs:**
- `model`: Any model (GroundingDINO or YOLO-World)
- `image`: Input image
- `prompt`: Text prompt (supports both formats)
  - For GroundingDINO: "cat . dog . car ."
  - For YOLO-World: "cat, dog, car"
- `threshold`: Confidence threshold (0.0-1.0)
- `text_threshold`: Text matching threshold (optional, GroundingDINO only)

**Outputs:**
- `bboxes`: Detected bounding boxes with labels and scores
- `annotated_image`: Image with drawn bounding boxes
- `labels`: String with detected object labels

**Features:**
- ✅ Auto-detects model type
- ✅ Single node for all detection models
- ✅ Simplified workflow

### 6. BboxToMask
Converts bounding boxes to binary masks.

**Inputs:**
- `image`: Input image (for dimensions)
- `bboxes`: Bounding box data
- `combine_masks`: Whether to combine all boxes into one mask (optional)

**Outputs:**
- `masks`: Binary mask(s) for detected regions

## Key Features

### 🚀 Model Caching
Models are cached in memory by default for instant reuse:
- First load: Downloads and initializes model
- Second load: Instant loading from cache
- Toggle `keep_in_memory=False` to clear cache

### 📁 Local Model Storage
All models are downloaded to the plugin's local Models directory:
- Location: `ComfyUI/custom_nodes/ComfyUI-Grounding/Models/`
- Auto-created on first use with a `models_go_here.txt` placeholder
- GroundingDINO/MM-GroundingDINO: Stored in HuggingFace transformers cache format
- YOLO-World: Stored as `.pt` files (yolov8s-world.pt, etc.)

### 🎯 UnifiedDetector
One detector for all models - automatically detects whether you're using GroundingDINO or YOLO-World and routes to the appropriate inference method.

### 📦 Batch Processing
All detector nodes support batch processing:
- Process multiple images in a single forward pass
- Input images as a batch tensor (B, H, W, C)
- Returns batched bounding boxes with lists of detections for each image
- Annotated images are returned as a single batch tensor

### 🎯 Referring Expression Comprehension (REC)
Automatic REC mode for natural language queries:
- **GroundingDINO**: If prompt has no periods (`.`) or commas (`,`), uses full text as label
  - Example: `"main object in the center of the picture"` → all detections labeled with full text
  - With separators: `"person . car . dog ."` → uses standard noun phrase extraction
- **YOLO-World**: If classes has no comma (`,`), uses full text as label
  - Example: `"red car in the parking lot"` → all detections labeled with full text
  - With separators: `"person, car, dog"` → uses standard class mapping
- **Single Box Mode**: Enable `single_box_mode` to return only the highest-scoring detection (perfect for REC tasks)

### 📋 Output Formats
Detector nodes support two output formats for bounding boxes:

**1. `list_only` (Default - SAM2 Compatible)**
- Returns: List of numpy arrays, one per image in batch
- Format: `[[bbox1, bbox2], [bbox3], ...]` where each bbox is `[x1, y1, x2, y2]`
- Use when: Connecting to SAM2 segmentation nodes
- Example:
  ```python
  # For 2 images with 2 and 1 detections respectively
  [
      array([[10, 20, 100, 200], [50, 60, 150, 250]]),  # Image 0: 2 boxes
      array([[30, 40, 120, 180]])                       # Image 1: 1 box
  ]
  ```

**2. `dict_with_data` (Full Metadata)**
- Returns: Dictionary with boxes, labels, and scores
- Format: `{"boxes": [...], "labels": [...], "scores": [...]}`
- Use when: Need label and confidence information for downstream processing
- NOT compatible with SAM2 (dict iteration gives keys, not values)
- Example:
  ```python
  {
      "boxes": [array([...]), array([...])],      # Boxes per image
      "labels": [["cat", "dog"], ["bird"]],       # Labels per image
      "scores": [array([0.9, 0.8]), array([0.7])] # Scores per image
  }
  ```

**Recommendation:** Use `list_only` (default) unless you specifically need label/score metadata for your workflow.

## Usage Examples

### Example 1: Basic Object Detection with GroundingDINO

1. Add `GroundingDINOModelLoader` node
2. Select model (e.g., "GroundingDINO_SwinT_OGC (694MB)")
3. Add `GroundingDINODetector` node
4. Connect model output to detector
5. Load an image and connect to detector
6. Set prompt: "person . car . bicycle ."
7. Adjust thresholds as needed
8. View annotated image output

### Example 2: YOLO-World Detection

1. Add `YOLOWorldModelLoader` node
2. Select model (e.g., "yolov8s-world")
3. Add `YOLOWorldDetector` node
4. Connect model and image
5. Set classes: "cat, dog, bird, horse"
6. Adjust confidence threshold
7. View results

### Example 3: Using UnifiedDetector (Recommended)

1. Add any model loader (GroundingDINO or YOLO-World)
2. Load your model
3. Add `UnifiedDetector` node
4. Connect model and image
5. Set prompt (works with both formats!)
6. The node automatically detects model type and uses the right inference

### Example 4: MM-GroundingDINO for High Accuracy

1. Add `GroundingDINOModelLoader` node
2. Select "MM-GroundingDINO-Base (All datasets, 59.5 mAP)"
3. Enable `keep_in_memory=True` (default)
4. Add `GroundingDINODetector` or `UnifiedDetector`
5. Set prompt: "specific object . another object ."
6. Enjoy state-of-the-art detection accuracy!

### Example 5: Convert Detections to Masks

1. Use any detector node (GroundingDINO or YOLO-World)
2. Add `BboxToMask` node
3. Connect image and bboxes outputs
4. Choose whether to combine masks
5. Use masks for segmentation, inpainting, etc.

## Model Downloads

Models will be automatically downloaded on first use:
- **All models**: Stored in `ComfyUI/custom_nodes/ComfyUI-Grounding/Models/`
- **GroundingDINO/MM-GroundingDINO**: HuggingFace transformers cache format
- **YOLO-World**: Direct `.pt` files (e.g., yolov8s-world.pt)

## Tips

- **GroundingDINO prompts**: End each object with a period (e.g., "cat . dog . car .")
- **YOLO-World classes**: Use comma-separated values (e.g., "cat, dog, car")
- **Threshold tuning**: Start with 0.3 and adjust based on results
  - Lower threshold = more detections (including false positives)
  - Higher threshold = fewer, more confident detections
- **Performance**: YOLO-World is faster but GroundingDINO may be more accurate

## Integration with SAM2

These nodes work seamlessly with SAM2 for instance segmentation:

1. Use GroundingDINO or YOLO-World to detect objects
2. Convert bboxes to masks using `BboxToMask`
3. Use masks with SAM2 for precise segmentation

## Troubleshooting

### ImportError: transformers not found
```bash
pip install transformers timm
```

### ImportError: ultralytics not found
```bash
pip install ultralytics
```

### Models not downloading
Check your internet connection and ensure you have write permissions to the ComfyUI models directory.

### CUDA out of memory
Try using a smaller model variant (e.g., GroundingDINO-T instead of -B, or yolov8s instead of yolov8x).

## License

MIT License - See LICENSE file for details

## Credits

- GroundingDINO: [IDEA-Research](https://github.com/IDEA-Research/GroundingDINO)
- YOLO-World: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Transformers: [HuggingFace](https://huggingface.co/transformers)

## Support

For issues and feature requests, please visit the GitHub repository.
