"""
Grounding Node Initialization
Ensures web files are accessible in both old and new ComfyUI versions
"""

import os
import shutil
import inspect


def get_ext_dir():
    """Get the extension directory path"""
    return os.path.dirname(os.path.abspath(__file__))


def get_comfy_dir():
    """Get the ComfyUI root directory"""
    try:
        from server import PromptServer
        return os.path.dirname(inspect.getfile(PromptServer))
    except:
        # Fallback: go up from custom_nodes/ComfyUI-Grounding
        ext_dir = get_ext_dir()
        return os.path.dirname(os.path.dirname(ext_dir))


def should_install_js():
    """Check if we need to manually install JS files for older ComfyUI versions"""
    try:
        from server import PromptServer
        # New ComfyUI versions have automatic web extension serving
        return not hasattr(PromptServer.instance, "supports") or \
               "custom_nodes_from_web" not in PromptServer.instance.supports
    except:
        # If PromptServer isn't available, assume we need manual installation
        return True


def link_js(src, dst):
    """Create a symlink (or junction on Windows) from src to dst"""
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)

    # Windows junction
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(src, dst)
            return True
        except:
            pass

    # Unix symlink
    try:
        os.symlink(src, dst)
        return True
    except Exception as e:
        print(f"[ComfyUI-Grounding] Failed to create symlink: {e}")
        return False


def is_junction(path):
    """Check if path is a Windows junction"""
    if os.name != "nt":
        return False
    try:
        return bool(os.readlink(path))
    except OSError:
        return False


def install_js():
    """Install/link JavaScript files for the extension"""
    src_dir = os.path.join(get_ext_dir(), "web", "js")

    if not os.path.exists(src_dir):
        print("[ComfyUI-Grounding] Warning: web/js directory not found")
        return

    # Check if we need to install
    if not should_install_js():
        print("[ComfyUI-Grounding] Modern ComfyUI detected - using automatic web extension serving")
        return

    print("[ComfyUI-Grounding] Older ComfyUI detected - installing web files manually")

    # Create destination directory in ComfyUI/web/extensions/
    comfy_web_ext = os.path.join(get_comfy_dir(), "web", "extensions")
    if not os.path.exists(comfy_web_ext):
        os.makedirs(comfy_web_ext, exist_ok=True)

    dst_dir = os.path.join(comfy_web_ext, "ComfyUI-Grounding")

    # Check if already linked or exists
    linked = os.path.islink(dst_dir) or is_junction(dst_dir)

    if linked or os.path.exists(dst_dir):
        if linked:
            print("[ComfyUI-Grounding] JS files already linked")
        else:
            print("[ComfyUI-Grounding] JS directory already exists")
        return

    # Try to create symlink first (faster and saves space)
    if link_js(src_dir, dst_dir):
        print("[ComfyUI-Grounding] ✅ JS files linked successfully")
        return

    # Fallback: copy files
    try:
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        print("[ComfyUI-Grounding] ✅ JS files copied successfully")
    except Exception as e:
        print(f"[ComfyUI-Grounding] ❌ Failed to install JS files: {e}")


def init():
    """Initialize the extension"""
    print("[ComfyUI-Grounding] Initializing...")
    install_js()
    return True
