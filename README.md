# ComfyUI-Grounding

**Grounding for dummies, simplest workflow**

![Simple Workflow](docs/simple.png)

🎯 **3 Nodes Total** - Loader → Detector → Visualizer

🚀 **5 Model Families** - GroundingDINO, MM-GroundingDINO, OWLv2, Florence-2, YOLO-World

💾 **Smart Caching** - Instant reload

📦 **Batch Processing** - Multiple images at once

🎭 **Built-in Masks** - No separate BboxToMask node needed

## Quick Start

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/PozzettiAndrea/ComfyUI-Grounding
cd ComfyUI-Grounding
pip install -r requirements.txt
```

## The Nodes


### 1. Grounding Model Loader
Load any of 15+ models from a single dropdown.

<div style="font-size: 0.75em; line-height: 1.4;">

  1. GroundingDINO: SwinT OGC (694MB) - IDEA-Research/grounding-dino-tiny
  2. GroundingDINO: SwinB (938MB) - IDEA-Research/grounding-dino-base
  3. MM-GroundingDINO: Tiny O365+GoldG (50.4 mAP) - openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg
  4. MM-GroundingDINO: Tiny O365+GoldG+GRIT (50.5 mAP) - openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit
  5. MM-GroundingDINO: Tiny O365+GoldG+V3Det (50.6 mAP) - openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det
  6. MM-GroundingDINO: Base O365+GoldG+V3Det (52.5 mAP) - openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det
  7. MM-GroundingDINO: Base All Datasets (59.5 mAP) - openmmlab-community/mm_grounding_dino_base_all
  8. MM-GroundingDINO: Large O365v2+OIv6+GoldG (53.0 mAP) - openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg
  9. MM-GroundingDINO: Large All Datasets (60.3 mAP) - openmmlab-community/mm_grounding_dino_large_all
  10. OWLv2: Base Patch16 - google/owlv2-base-patch16
  11. OWLv2: Large Patch14 - google/owlv2-large-patch14
  12. OWLv2: Base Patch16 Ensemble - google/owlv2-base-patch16-ensemble
  13. OWLv2: Large Patch14 Ensemble - google/owlv2-large-patch14-ensemble
  14. Florence-2: Base (0.23B params) - microsoft/Florence-2-base
  15. Florence-2: Large (0.77B params) - microsoft/Florence-2-large
  16. YOLO-World: v8s (Small) - yolov8s-worldv2.pt
  17. YOLO-World: v8m (Medium) - yolov8m-worldv2.pt
  18. YOLO-World: v8l (Large) - yolov8l-worldv2.pt
  19. YOLO-World: v8x (Extra Large) - yolov8x-worldv2.pt
</div>

### 2. Grounding Detector
Universal detector for all models.

#### Key Features

<div style="font-size: 0.9em; line-height: 1.4;">

- Overrides standard text label splitting. It splits only at ".", otherwise label is what you write
- Enable `single_box_mode` for single detection
- First load: Downloads model
- Second load: Instant from cache
- Models stored in ComfyUI standard folders:
  - GroundingDINO → `models/grounding-dino/`
  - Florence-2 & OWLv2 → `models/LLM/`
  - YOLO-World → `models/yolo_world/`
  - SAM2 → `models/sam2/`
- Process multiple images in one pass
- All nodes support batches
</div>

#### Florence-2 Attention Modes

<div style="font-size: 0.9em; line-height: 1.4;">

- `eager` - Most compatible (default)
- `sdpa` - Faster on PyTorch 2.0+
- `flash_attention_2` - Fastest on A100/H100
</div>

### 3. Bounding Box Visualizer
<div style="font-size: 0.9em; line-height: 1.4;">

Re-draw bboxes on images with custom line width. Optional since detector already returns annotated images.
</div>

### 4. SAM2 Integration

<div style="font-size: 0.9em; line-height: 1.4;">

**DownloadAndLoadSAM2Model** - Load SAM2 models (base/large/small/tiny)
**Sam2Segmentation** - Segment using bboxes from grounding models or point coordinates

- Supports batch processing
- Models cached in memory for instant reload
- Compatible with all grounding model outputs
</div>

### Tips

**Prompt Format:**
- Use periods for multiple labels → `"laser. crocodile."`
- Use commas to keep a single label → `"laser, crocodile"`

## Credits

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - IDEA-Research
- [OWLv2](https://huggingface.co/google/owlv2-base-patch16) - Google Research
- [Florence-2](https://huggingface.co/microsoft/Florence-2-base) - Microsoft Research
- [YOLO-World](https://github.com/ultralytics/ultralytics) - Ultralytics

## License

MIT License