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
            [sys.executable, "-m", "pip", "install", "flash-attn", "--prefer-binary"],
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
    """Copy example assets recursively to ComfyUI input directory"""
    node_dir = get_node_dir()
    assets_src = os.path.join(node_dir, "assets")

    if not os.path.exists(assets_src):
        print("[ComfyUI-Grounding] Warning: assets folder not found")
        return False

    # Get ComfyUI input directory with error handling and fallback
    try:
        input_dir = folder_paths.get_input_directory()
        if not input_dir:
            raise ValueError("folder_paths.get_input_directory() returned None")
        print(f"[ComfyUI-Grounding] Target input directory: {input_dir}")
    except Exception as e:
        print(f"[ComfyUI-Grounding] ❌ Error getting input directory: {e}")
        print(f"[ComfyUI-Grounding] Attempting fallback to default ComfyUI/input")
        # Fallback: try to find ComfyUI/input relative to custom_nodes
        try:
            custom_nodes_dir = os.path.dirname(node_dir)
            comfyui_dir = os.path.dirname(custom_nodes_dir)
            input_dir = os.path.join(comfyui_dir, "input")
            if not os.path.exists(input_dir):
                print(f"[ComfyUI-Grounding] ❌ Fallback input directory does not exist: {input_dir}")
                return False
            print(f"[ComfyUI-Grounding] Using fallback input directory: {input_dir}")
        except Exception as fallback_e:
            print(f"[ComfyUI-Grounding] ❌ Fallback failed: {fallback_e}")
            return False

    try:
        # Recursively copy all files and directories from assets folder to input
        copied_count = 0
        skipped_count = 0

        def copy_recursive(src_dir, dst_dir, rel_path=""):
            """Recursively copy files and directories"""
            nonlocal copied_count, skipped_count

            for item in os.listdir(src_dir):
                src_path = os.path.join(src_dir, item)
                dst_path = os.path.join(dst_dir, item)
                rel_item_path = os.path.join(rel_path, item) if rel_path else item

                if os.path.isdir(src_path):
                    # Handle subdirectory
                    print(f"[ComfyUI-Grounding] Processing subdirectory: {rel_item_path}/")

                    # Create destination directory if it doesn't exist
                    if not os.path.exists(dst_path):
                        os.makedirs(dst_path, exist_ok=True)
                        print(f"[ComfyUI-Grounding]   Created directory: {rel_item_path}/")

                    # Recursively copy contents
                    copy_recursive(src_path, dst_path, rel_item_path)

                else:
                    # Handle file
                    if os.path.exists(dst_path):
                        print(f"[ComfyUI-Grounding]   Skipped (exists): {rel_item_path}")
                        skipped_count += 1
                    else:
                        shutil.copy2(src_path, dst_path)
                        file_size = os.path.getsize(src_path)
                        size_kb = file_size / 1024
                        print(f"[ComfyUI-Grounding]   ✅ Copied: {rel_item_path} ({size_kb:.1f} KB)")
                        copied_count += 1

        # Start recursive copy from assets directory
        copy_recursive(assets_src, input_dir)

        # Summary
        if copied_count > 0:
            print(f"[ComfyUI-Grounding] ✅ Successfully copied {copied_count} file(s) to {input_dir}")
        if skipped_count > 0:
            print(f"[ComfyUI-Grounding] ℹ️  Skipped {skipped_count} file(s) (already exist)")
        if copied_count == 0 and skipped_count == 0:
            print(f"[ComfyUI-Grounding] ⚠️  No files found in {assets_src}")

        return True

    except Exception as e:
        print(f"[ComfyUI-Grounding] ❌ Error copying assets: {e}")
        import traceback
        print(f"[ComfyUI-Grounding] Traceback: {traceback.format_exc()}")
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

    # Copy assets and workflows (skips files that already exist)
    print("[ComfyUI-Grounding] Checking assets and workflows...")
    assets_ok = copy_assets()
    workflows_ok = copy_workflows()

    # Report completion status
    if assets_ok and workflows_ok:
        print("[ComfyUI-Grounding] ✅ PreStartup initialization complete")
    else:
        print("[ComfyUI-Grounding] ⚠️  PreStartup completed with some warnings")


# Call main() at module level so it runs when ComfyUI imports this script
main()
