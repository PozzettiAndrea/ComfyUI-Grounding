#!/usr/bin/env python3
"""
Create a comparison image showing detection results from all models
Run this after the tests have completed to generate a visual comparison
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys


def create_comparison_grid():
    """Create a grid comparison of all detection results"""
    test_outputs = Path(__file__).parent / "test_outputs"

    # Define the images we want to compare
    images_to_compare = [
        ("Florence-2", "florence2_eager_detection.png"),
        ("GroundingDINO", "grounding_dino_detection.png"),
        ("OWLv2", "owlv2_detection.png"),
        ("YOLO-World", "yolo_world_detection.png"),
    ]

    # Load all images
    loaded_images = []
    for title, filename in images_to_compare:
        img_path = test_outputs / filename
        if img_path.exists():
            img = Image.open(img_path)
            loaded_images.append((title, img))
            print(f"✓ Loaded {filename}")
        else:
            print(f"✗ Missing {filename}")

    if not loaded_images:
        print("No images found in test_outputs directory!")
        return False

    # Calculate grid dimensions
    # Use 2 columns
    cols = 2
    rows = (len(loaded_images) + cols - 1) // cols

    # Get image dimensions (assuming all are same size)
    img_width, img_height = loaded_images[0][1].size

    # Add space for titles
    title_height = 40
    padding = 20

    # Create output image
    grid_width = cols * img_width + (cols + 1) * padding
    grid_height = rows * (img_height + title_height) + (rows + 1) * padding

    comparison = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(comparison)

    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Place images in grid
    for idx, (title, img) in enumerate(loaded_images):
        row = idx // cols
        col = idx % cols

        x = padding + col * (img_width + padding)
        y = padding + row * (img_height + title_height + padding)

        # Draw title
        title_y = y
        # Center the title
        bbox = draw.textbbox((0, 0), title, font=font)
        title_width = bbox[2] - bbox[0]
        title_x = x + (img_width - title_width) // 2
        draw.text((title_x, title_y), title, fill='black', font=font)

        # Paste image
        comparison.paste(img, (x, y + title_height))

    # Save comparison
    output_path = test_outputs / "comparison_all_models.png"
    comparison.save(output_path)
    print(f"\n✅ Saved comparison to {output_path}")

    # Also save as high quality JPG
    jpg_path = test_outputs / "comparison_all_models.jpg"
    comparison.save(jpg_path, quality=95)
    print(f"✅ Saved comparison to {jpg_path}")

    return True


if __name__ == "__main__":
    success = create_comparison_grid()
    sys.exit(0 if success else 1)
