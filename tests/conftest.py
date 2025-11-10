"""
Pytest configuration and fixtures for ComfyUI-Grounding tests
"""
import sys
import os
from pathlib import Path
import pytest
import torch
from unittest.mock import MagicMock


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Run tests on GPU instead of CPU (much faster for real model tests)"
    )


# Add the custom node directory to Python path so we can import nodes package
custom_nodes_dir = Path(__file__).parent.parent
sys.path.insert(0, str(custom_nodes_dir))

# Mock ComfyUI modules at module level BEFORE pytest starts
# This prevents import errors when pytest tries to load __init__.py files
mock_folder_paths = type("folder_paths", (), {})()
mock_folder_paths.models_dir = "/tmp/test_models"
mock_folder_paths.get_folder_paths = lambda x: ["/tmp/test_models"]
sys.modules["folder_paths"] = mock_folder_paths

mock_comfy = type("comfy", (), {})()
mock_comfy_utils = type("utils", (), {})()
mock_comfy_utils.load_torch_file = lambda x: {}
mock_comfy_utils.ProgressBar = MagicMock()
mock_comfy_utils.common_upscale = MagicMock()
mock_comfy.utils = mock_comfy_utils

mock_comfy_mm = type("model_management", (), {})()

# Device selection: Check environment variable set by session fixture
def _get_test_device():
    """Get device for testing - GPU if --use-gpu flag is set, else CPU"""
    use_gpu = os.environ.get("PYTEST_USE_GPU", "0") == "1"
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

mock_comfy_mm.get_torch_device = _get_test_device
mock_comfy_mm.soft_empty_cache = lambda: None
mock_comfy_mm.load_models_gpu = lambda x: None
mock_comfy.model_management = mock_comfy_mm

sys.modules["comfy"] = mock_comfy
sys.modules["comfy.utils"] = mock_comfy_utils
sys.modules["comfy.model_management"] = mock_comfy_mm

# Mock grounding_init module that __init__.py tries to import
mock_grounding_init = type("grounding_init", (), {})()
mock_grounding_init.init = lambda: None
sys.modules["grounding_init"] = mock_grounding_init

# Mock flash_attn module for SA2VA tests (SA2VA requires flash_attn imports)
# This mock allows transformers to validate imports without actually installing flash_attn
import importlib.util
import importlib.machinery

# Create a proper module spec so transformers recognizes it as a real package
mock_flash_attn_spec = importlib.machinery.ModuleSpec("flash_attn", None, is_package=True)
mock_flash_attn = type("flash_attn", (), {})()
mock_flash_attn.__spec__ = mock_flash_attn_spec
mock_flash_attn.__name__ = "flash_attn"
mock_flash_attn.__package__ = "flash_attn"

mock_flash_attn_interface_spec = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", None)
mock_flash_attn_interface = type("flash_attn_interface", (), {})()
mock_flash_attn_interface.__spec__ = mock_flash_attn_interface_spec
mock_flash_attn_interface.__name__ = "flash_attn.flash_attn_interface"
mock_flash_attn_interface.__package__ = "flash_attn"

# Mock the functions that SA2VA's flash_attention.py tries to import
# These won't actually be called since we use attn_implementation="eager"
def _mock_flash_attn_func(*args, **kwargs):
    """Dummy flash attention function - should not be called"""
    raise NotImplementedError("Flash attention mock called - this should not happen with eager attention")

mock_flash_attn_interface.flash_attn_unpadded_qkvpacked_func = _mock_flash_attn_func  # v1
mock_flash_attn_interface.flash_attn_varlen_qkvpacked_func = _mock_flash_attn_func  # v2
mock_flash_attn_interface.flash_attn_varlen_func = _mock_flash_attn_func  # Additional function
mock_flash_attn_interface.flash_attn_func = _mock_flash_attn_func  # Base function

mock_flash_attn.flash_attn_interface = mock_flash_attn_interface

sys.modules["flash_attn"] = mock_flash_attn
sys.modules["flash_attn.flash_attn_interface"] = mock_flash_attn_interface

# Mock torch.cuda() and .to() calls for SA2VA (SA2VA has hardcoded CUDA calls in model code)
# This allows tests to run on CPU by making CUDA calls redirect to CPU
_original_tensor_cuda = torch.Tensor.cuda
_original_tensor_to = torch.Tensor.to

def _mock_tensor_cuda(self, device=None, non_blocking=False):
    """Mock .cuda() to return CPU tensor when CUDA is not available"""
    if torch.cuda.is_available():
        return _original_tensor_cuda(self, device=device, non_blocking=non_blocking)
    else:
        # Return CPU tensor instead of failing
        return self

def _mock_tensor_to(self, *args, **kwargs):
    """Mock .to() to redirect CUDA device requests to CPU when CUDA is not available"""
    if torch.cuda.is_available():
        return _original_tensor_to(self, *args, **kwargs)
    else:
        # Check if trying to move to CUDA device
        if args and (isinstance(args[0], str) and 'cuda' in args[0].lower() or
                     isinstance(args[0], torch.device) and args[0].type == 'cuda'):
            # Redirect to CPU
            new_args = (torch.device('cpu'),) + args[1:]
            return _original_tensor_to(self, *new_args, **kwargs)
        return _original_tensor_to(self, *args, **kwargs)

# Patch torch.Tensor methods
torch.Tensor.cuda = _mock_tensor_cuda
torch.Tensor.to = _mock_tensor_to


def pytest_ignore_collect(collection_path, path, config):
    """Ignore __init__.py files during collection"""
    if collection_path.name == "__init__.py":
        return True
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_test_device(request):
    """Configure test device based on --use-gpu flag"""
    use_gpu = request.config.getoption("--use-gpu")
    if use_gpu:
        os.environ["PYTEST_USE_GPU"] = "1"
        if torch.cuda.is_available():
            print(f"\n🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("\n⚠️  --use-gpu specified but CUDA not available, using CPU")
    else:
        os.environ["PYTEST_USE_GPU"] = "0"
        print("\n💻 Using CPU (use --use-gpu for GPU acceleration)")

    yield

    # Cleanup
    os.environ.pop("PYTEST_USE_GPU", None)


@pytest.fixture(scope="session", autouse=True)
def setup_mock_comfy():
    """Set up mock ComfyUI modules for testing - runs once per session"""
    # Ensure mocks persist throughout test session
    return True


@pytest.fixture
def mock_comfy_environment():
    """Provide access to mocked ComfyUI environment (already set up at module level)"""
    return sys.modules["folder_paths"]


@pytest.fixture
def small_image():
    """Load plantpot.png test image"""
    from PIL import Image
    import numpy as np

    # Load the test image
    img_path = Path(__file__).parent.parent / "assets" / "plantpot.png"
    img = Image.open(img_path).convert("RGB")

    # Resize to reasonable size for testing (keeping aspect ratio)
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)

    # Convert to torch tensor in format (1, H, W, C) with values in [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)

    return img_tensor


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Clear model cache between tests"""
    from nodes.utils.cache import MODEL_CACHE
    MODEL_CACHE.clear()
    yield
    MODEL_CACHE.clear()


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no model loading)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests with mocked models"
    )
    config.addinivalue_line(
        "markers", "real_model: Tests that download and use real models (slow)"
    )
    config.addinivalue_line(
        "markers", "workflow: Workflow validation tests"
    )
