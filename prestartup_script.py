"""
PreStartup Script for ComfyUI-Grounding
Copies example assets and workflows to ComfyUI directories on first run
"""

import os
import shutil
import sys
from pathlib import Path

# Import folder_paths from ComfyUI
try:
    import folder_paths
except ImportError:
    print("[ComfyUI-Grounding] Warning: Could not import folder_paths")
    sys.exit(0)


def get_node_dir():
    """Get the ComfyUI-Grounding node directory"""
    return os.path.dirname(os.path.abspath(__file__))


def get_marker_file():
    """Get path to marker file that tracks if assets have been copied"""
    return os.path.join(get_node_dir(), ".assets_copied")


def has_assets_been_copied():
    """Check if assets have already been copied"""
    return os.path.exists(get_marker_file())


def mark_assets_as_copied():
    """Create marker file to indicate assets have been copied"""
    try:
        with open(get_marker_file(), 'w') as f:
            f.write("Assets copied successfully\n")
    except Exception as e:
        print(f"[ComfyUI-Grounding] Warning: Could not create marker file: {e}")


def copy_assets():
    """Copy example assets directly to ComfyUI input directory"""
    node_dir = get_node_dir()
    assets_src = os.path.join(node_dir, "assets")

    if not os.path.exists(assets_src):
        print("[ComfyUI-Grounding] Warning: assets folder not found")
        return False

    # Get ComfyUI input directory
    input_dir = folder_paths.get_input_directory()

    try:
        # Copy all files from assets folder directly to input
        copied_count = 0
        for item in os.listdir(assets_src):
            src_path = os.path.join(assets_src, item)

            # Skip directories (like tobatch/)
            if os.path.isdir(src_path):
                continue

            dst_path = os.path.join(input_dir, item)

            # Skip if file already exists (don't overwrite user files)
            if os.path.exists(dst_path):
                continue

            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"[ComfyUI-Grounding] Copied asset: {item} to input/")

        if copied_count > 0:
            print(f"[ComfyUI-Grounding] ✅ Copied {copied_count} assets to {input_dir}")
        else:
            print(f"[ComfyUI-Grounding] All assets already exist in {input_dir}")

        return True

    except Exception as e:
        print(f"[ComfyUI-Grounding] ❌ Error copying assets: {e}")
        return False


def copy_workflows():
    """Copy example workflows to ComfyUI user workflows directory"""
    node_dir = get_node_dir()
    workflows_src = os.path.join(node_dir, "workflows")

    if not os.path.exists(workflows_src):
        print("[ComfyUI-Grounding] Warning: workflows folder not found")
        return False

    # Get ComfyUI user workflows directory
    user_dir = folder_paths.get_user_directory()
    workflows_dst = os.path.join(user_dir, "default", "workflows")

    try:
        # Create destination directory if it doesn't exist
        os.makedirs(workflows_dst, exist_ok=True)

        # Copy all workflow JSON files
        copied_count = 0
        for filename in os.listdir(workflows_src):
            if not filename.endswith('.json'):
                continue

            src_path = os.path.join(workflows_src, filename)
            # Prefix workflow names to avoid conflicts
            dst_filename = f"ComfyUI-Grounding_{filename}"
            dst_path = os.path.join(workflows_dst, dst_filename)

            # Skip if file already exists (don't overwrite user workflows)
            if os.path.exists(dst_path):
                continue

            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"[ComfyUI-Grounding] Copied workflow: {dst_filename}")

        if copied_count > 0:
            print(f"[ComfyUI-Grounding] ✅ Copied {copied_count} workflows to {workflows_dst}")
        else:
            print(f"[ComfyUI-Grounding] All workflows already exist in {workflows_dst}")

        return True

    except Exception as e:
        print(f"[ComfyUI-Grounding] ❌ Error copying workflows: {e}")
        return False


def main():
    """Main initialization function"""
    print("[ComfyUI-Grounding] PreStartup Script Running...")

    # Check if this is the first run
    if has_assets_been_copied():
        print("[ComfyUI-Grounding] Assets already copied (skipping)")
        return

    # Copy assets and workflows
    assets_ok = copy_assets()
    workflows_ok = copy_workflows()

    # Mark as completed if both succeeded
    if assets_ok and workflows_ok:
        mark_assets_as_copied()
        print("[ComfyUI-Grounding] ✅ PreStartup initialization complete")
    else:
        print("[ComfyUI-Grounding] ⚠️  PreStartup completed with some warnings")


if __name__ == "__main__":
    main()
