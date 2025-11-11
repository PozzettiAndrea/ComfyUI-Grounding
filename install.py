"""
Installation script for ComfyUI-Grounding optional dependencies.
Called by ComfyUI Manager during installation/update.
"""
import subprocess
import sys


def try_install_flash_attn():
    """
    Attempt to install flash_attn (optional dependency for faster inference).
    Fails gracefully if installation is not possible (e.g., no CUDA, wrong GPU architecture).
    """
    print("[ComfyUI-Grounding] Checking for flash_attn...")

    # Check if already installed
    try:
        import flash_attn
        print("[ComfyUI-Grounding] ✅ flash_attn already installed")
        return True
    except ImportError:
        pass

    print("[ComfyUI-Grounding] Attempting to install flash_attn (optional)...")
    print("[ComfyUI-Grounding] Note: This may take 5-10 minutes to compile and requires CUDA GPU")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for compilation
        )

        if result.returncode == 0:
            print("[ComfyUI-Grounding] ✅ flash_attn installed successfully")
            print("[ComfyUI-Grounding] Florence-2 and SA2VA will use faster inference")
            return True
        else:
            print("[ComfyUI-Grounding] ⚠️  flash_attn installation failed")
            print("[ComfyUI-Grounding] This is optional - nodes will work without it (just slower)")
            if result.stderr:
                print(f"[ComfyUI-Grounding] Error details: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("[ComfyUI-Grounding] ⚠️  flash_attn installation timed out")
        print("[ComfyUI-Grounding] You can try installing manually: pip install flash-attn")
        return False
    except Exception as e:
        print(f"[ComfyUI-Grounding] ⚠️  flash_attn installation error: {e}")
        print("[ComfyUI-Grounding] Continuing without flash_attn (nodes will still work)")
        return False


if __name__ == "__main__":
    print("[ComfyUI-Grounding] Running installation script...")
    try_install_flash_attn()
    print("[ComfyUI-Grounding] Installation script completed")
