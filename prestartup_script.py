"""
PreStartup Script for ComfyUI-Grounding
- Attempts to install flash_attn (optional dependency) on first run
- Copies example assets and workflows to ComfyUI directories on first run
"""

import os
import shutil
import subprocess
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


def get_flash_attn_marker():
    """Get path to marker file that tracks if flash_attn installation was attempted"""
    return os.path.join(get_node_dir(), ".flash_attn_attempted")


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


def mark_flash_attn_attempted(success):
    """Create marker file to indicate flash_attn installation was attempted"""
    try:
        with open(get_flash_attn_marker(), 'w') as f:
            if success:
                f.write("flash_attn installed successfully\n")
            else:
                f.write("flash_attn installation failed (optional)\n")
    except Exception as e:
        print(f"[ComfyUI-Grounding] Warning: Could not create flash_attn marker: {e}")


def try_install_flash_attn():
    """
    Attempt to install flash_attn (optional dependency for faster inference).
    Only runs once - uses marker file to track attempts.
    """
    # Check if already attempted
    if os.path.exists(get_flash_attn_marker()):
        return True  # Already attempted, skip

    print("[ComfyUI-Grounding] Checking for flash_attn...")

    # Check if already installed
    try:
        import flash_attn
        print("[ComfyUI-Grounding] ✅ flash_attn already installed")
        mark_flash_attn_attempted(True)
        return True
    except ImportError:
        pass

    print("[ComfyUI-Grounding] Attempting to install flash_attn (optional)...")
    print("[ComfyUI-Grounding] Note: This may take 5-10 minutes and requires CUDA GPU")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print("[ComfyUI-Grounding] ✅ flash_attn installed successfully")
            print("[ComfyUI-Grounding] Florence-2 and SA2VA will use faster inference")
            mark_flash_attn_attempted(True)
            return True
        else:
            print("[ComfyUI-Grounding] ⚠️  flash_attn installation failed")
            print("[ComfyUI-Grounding] This is optional - nodes will work without it")
            mark_flash_attn_attempted(False)
            return False

    except subprocess.TimeoutExpired:
        print("[ComfyUI-Grounding] ⚠️  flash_attn installation timed out")
        print("[ComfyUI-Grounding] Continuing without it (nodes will still work)")
        mark_flash_attn_attempted(False)
        return False
    except Exception as e:
        print(f"[ComfyUI-Grounding] ⚠️  flash_attn installation error: {e}")
        print("[ComfyUI-Grounding] Continuing without flash_attn")
        mark_flash_attn_attempted(False)
        return False


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

    # Try to install flash_attn (optional, only on first run)
    try_install_flash_attn()

    # Check if this is the first run for asset copying
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
