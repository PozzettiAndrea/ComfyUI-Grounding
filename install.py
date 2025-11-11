"""
Installation script for ComfyUI-Grounding optional dependencies.
Called by ComfyUI Manager during installation/update.
"""
import os
import subprocess
import sys


def install_requirements():
    """
    Install dependencies from requirements.txt.
    """
    print("[ComfyUI-Grounding] Installing requirements.txt dependencies...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, "requirements.txt")

    if not os.path.exists(requirements_path):
        print("[ComfyUI-Grounding] ⚠️  requirements.txt not found, skipping")
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_path],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("[ComfyUI-Grounding] ✅ Requirements installed successfully")
            return True
        else:
            print("[ComfyUI-Grounding] ⚠️  Requirements installation had issues")
            if result.stderr:
                print(f"[ComfyUI-Grounding] Error details: {result.stderr[:500]}")
            return False

    except Exception as e:
        print(f"[ComfyUI-Grounding] ⚠️  Requirements installation error: {e}")
        return False


def get_torch_cuda_version():
    """
    Detect installed torch and CUDA versions.
    Returns tuple: (torch_version, cuda_version) or (None, None) if not found.
    """
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.9.0"
        cuda_version = torch.version.cuda  # e.g., "12.8"

        # Extract major.minor for torch (e.g., "2.9.0" -> "2.9")
        torch_major_minor = '.'.join(torch_version.split('.')[:2])

        # Extract major minor for CUDA and remove dots (e.g., "12.8" -> "128")
        cuda_compact = cuda_version.replace('.', '') if cuda_version else None

        return torch_major_minor, cuda_compact
    except Exception as e:
        print(f"[ComfyUI-Grounding] Could not detect torch/CUDA: {e}")
        return None, None


def get_python_version():
    """
    Get Python version in cpXX format (e.g., "cp310" for Python 3.10).
    """
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def try_install_flash_attn():
    """
    Attempt to install flash_attn (optional dependency for faster inference).
    Tries prebuilt wheels first, then falls back to source compilation.
    """
    print("[ComfyUI-Grounding] Checking for flash_attn...")

    # Check if already installed
    try:
        import flash_attn
        print("[ComfyUI-Grounding] ✅ flash_attn already installed")
        return True
    except ImportError:
        pass

    # Detect versions
    torch_version, cuda_version = get_torch_cuda_version()
    py_version = get_python_version()

    print(f"[ComfyUI-Grounding] Detected: Python {py_version}, Torch {torch_version}, CUDA {cuda_version}")

    # Try prebuilt wheel if we have version info
    if torch_version and cuda_version:
        wheel_url = (
            f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/"
            f"flash_attn-2.8.1%2Bcu{cuda_version}torch{torch_version}-{py_version}-{py_version}-linux_x86_64.whl"
        )

        print(f"[ComfyUI-Grounding] Attempting to install prebuilt flash_attn wheel...")
        print(f"[ComfyUI-Grounding] Wheel URL: {wheel_url}")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", wheel_url],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                print("[ComfyUI-Grounding] ✅ flash_attn wheel installed successfully")
                print("[ComfyUI-Grounding] Florence-2 and SA2VA will use faster inference")
                return True
            else:
                print("[ComfyUI-Grounding] ⚠️  Prebuilt wheel not found or failed")
                if result.stderr:
                    print(f"[ComfyUI-Grounding] Error: {result.stderr[:300]}")
        except Exception as e:
            print(f"[ComfyUI-Grounding] ⚠️  Wheel installation error: {e}")

    # Fall back to source compilation
    print("[ComfyUI-Grounding] Falling back to source compilation (may take 5-10 minutes)...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("[ComfyUI-Grounding] ✅ flash_attn compiled and installed successfully")
            print("[ComfyUI-Grounding] Florence-2 and SA2VA will use faster inference")
            return True
        else:
            print("[ComfyUI-Grounding] ⚠️  flash_attn compilation failed")
            print("[ComfyUI-Grounding] This is optional - nodes will work without it (just slower)")
            if result.stderr:
                print(f"[ComfyUI-Grounding] Error details: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("[ComfyUI-Grounding] ⚠️  flash_attn compilation timed out")
        print("[ComfyUI-Grounding] Nodes will work without it (just slower)")
        return False
    except Exception as e:
        print(f"[ComfyUI-Grounding] ⚠️  flash_attn installation error: {e}")
        print("[ComfyUI-Grounding] Continuing without flash_attn (nodes will still work)")
        return False


if __name__ == "__main__":
    print("[ComfyUI-Grounding] Running installation script...")

    # Install requirements.txt first
    install_requirements()

    # Then try to install flash_attn with smart detection
    try_install_flash_attn()

    print("[ComfyUI-Grounding] Installation script completed")
