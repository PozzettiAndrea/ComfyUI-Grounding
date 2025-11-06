"""
Bounding box detection model registry
"""

# Unified model registry with prefixed names for easy identification
MODEL_REGISTRY = {
    # GroundingDINO models
    "GroundingDINO: SwinT OGC (694MB)": {
        "type": "grounding_dino",
        "hf_id": "IDEA-Research/grounding-dino-tiny",
        "framework": "transformers",
    },
    "GroundingDINO: SwinB (938MB)": {
        "type": "grounding_dino",
        "hf_id": "IDEA-Research/grounding-dino-base",
        "framework": "transformers",
    },
    # MM-GroundingDINO Tiny models
    "MM-GroundingDINO: Tiny O365+GoldG (50.4 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg",
        "framework": "transformers",
    },
    "MM-GroundingDINO: Tiny O365+GoldG+GRIT (50.5 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit",
        "framework": "transformers",
    },
    "MM-GroundingDINO: Tiny O365+GoldG+V3Det (50.6 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
        "framework": "transformers",
    },
    # MM-GroundingDINO Base models
    "MM-GroundingDINO: Base O365+GoldG+V3Det (52.5 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
        "framework": "transformers",
    },
    "MM-GroundingDINO: Base All Datasets (59.5 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_base_all",
        "framework": "transformers",
    },
    # MM-GroundingDINO Large models
    "MM-GroundingDINO: Large O365v2+OIv6+GoldG (53.0 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg",
        "framework": "transformers",
    },
    "MM-GroundingDINO: Large All Datasets (60.3 mAP)": {
        "type": "grounding_dino",
        "hf_id": "openmmlab-community/mm_grounding_dino_large_all",
        "framework": "transformers",
    },
    # OWLv2 models
    "OWLv2: Base Patch16": {
        "type": "owlv2",
        "hf_id": "google/owlv2-base-patch16",
        "framework": "transformers",
    },
    "OWLv2: Large Patch14": {
        "type": "owlv2",
        "hf_id": "google/owlv2-large-patch14",
        "framework": "transformers",
    },
    "OWLv2: Base Patch16 Ensemble": {
        "type": "owlv2",
        "hf_id": "google/owlv2-base-patch16-ensemble",
        "framework": "transformers",
    },
    "OWLv2: Large Patch14 Ensemble": {
        "type": "owlv2",
        "hf_id": "google/owlv2-large-patch14-ensemble",
        "framework": "transformers",
    },
    # Florence-2 models
    "Florence-2: Base (0.23B params)": {
        "type": "florence2",
        "hf_id": "microsoft/Florence-2-base",
        "framework": "transformers",
    },
    "Florence-2: Large (0.77B params)": {
        "type": "florence2",
        "hf_id": "microsoft/Florence-2-large",
        "framework": "transformers",
    },
    # YOLO-World models
    "YOLO-World: v8s (Small)": {
        "type": "yolo_world",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
        "framework": "ultralytics",
    },
    "YOLO-World: v8m (Medium)": {
        "type": "yolo_world",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-worldv2.pt",
        "framework": "ultralytics",
    },
    "YOLO-World: v8l (Large)": {
        "type": "yolo_world",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt",
        "framework": "ultralytics",
    },
    "YOLO-World: v8x (Extra Large)": {
        "type": "yolo_world",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt",
        "framework": "ultralytics",
    },
}
