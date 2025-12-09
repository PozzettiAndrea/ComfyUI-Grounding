"""
PreStartup Script for ComfyUI-Grounding
- Copies example assets and workflows to ComfyUI directories on first run
"""

import os
import shutil
import sys

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
        print(f"[ComfyUI-Grounding] [ERROR] Error getting input directory: {e}")
        print(f"[ComfyUI-Grounding] Attempting fallback to default ComfyUI/input")
        # Fallback: try to find ComfyUI/input relative to custom_nodes
        try:
            custom_nodes_dir = os.path.dirname(node_dir)
            comfyui_dir = os.path.dirname(custom_nodes_dir)
            input_dir = os.path.join(comfyui_dir, "input")
            if not os.path.exists(input_dir):
                print(f"[ComfyUI-Grounding] [ERROR] Fallback input directory does not exist: {input_dir}")
                return False
            print(f"[ComfyUI-Grounding] Using fallback input directory: {input_dir}")
        except Exception as fallback_e:
            print(f"[ComfyUI-Grounding] [ERROR] Fallback failed: {fallback_e}")
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
                        print(f"[ComfyUI-Grounding]   [OK] Copied: {rel_item_path} ({size_kb:.1f} KB)")
                        copied_count += 1

        # Start recursive copy from assets directory
        copy_recursive(assets_src, input_dir)

        # Summary
        if copied_count > 0:
            print(f"[ComfyUI-Grounding] [OK] Successfully copied {copied_count} file(s) to {input_dir}")
        if skipped_count > 0:
            print(f"[ComfyUI-Grounding] [INFO] Skipped {skipped_count} file(s) (already exist)")
        if copied_count == 0 and skipped_count == 0:
            print(f"[ComfyUI-Grounding] [WARNING] No files found in {assets_src}")

        return True

    except Exception as e:
        print(f"[ComfyUI-Grounding] [ERROR] Error copying assets: {e}")
        import traceback
        print(f"[ComfyUI-Grounding] Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main initialization function"""
    print("[ComfyUI-Grounding] PreStartup Script Running...")

    # Copy assets (skips files that already exist)
    print("[ComfyUI-Grounding] Checking assets...")
    assets_ok = copy_assets()

    # Report completion status
    if assets_ok:
        print("[ComfyUI-Grounding] [OK] PreStartup initialization complete")
    else:
        print("[ComfyUI-Grounding] [WARNING] PreStartup completed with some warnings")


# Call main() at module level so it runs when ComfyUI imports this script
main()
