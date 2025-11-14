"""
YOLO-World model loading
"""
import os
import folder_paths
import comfy.model_management as mm


def load_yolo_world(model_name, config):
    """Load YOLO-World model

    Args:
        model_name: Model display name
        config: Model configuration dict from registry

    Returns:
        Dict containing model, type, and framework
    """
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    # Use ComfyUI standard models directory
    models_dir = os.path.join(folder_paths.models_dir, "yolo_world")
    os.makedirs(models_dir, exist_ok=True)

    # Extract filename from URL
    url = config["url"]
    filename = url.split("/")[-1]
    model_path = os.path.join(models_dir, filename)

    # Download if needed
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        from torch.hub import download_url_to_file
        download_url_to_file(url, model_path, progress=True)
        print(f"[OK] Downloaded {model_name}")

    # Load model
    print(f"Loading {model_name} from {model_path}")
    model = YOLOWorld(model_path)

    # Move to device
    device = mm.get_torch_device()
    model.to(device)

    print(f"[OK] Successfully loaded {model_name}")

    return {
        "model": model,
        "type": "yolo_world",
        "framework": "ultralytics"
    }
