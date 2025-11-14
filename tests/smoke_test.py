"""
Smoke test for ComfyUI-Grounding
Tests basic module import functionality without requiring ComfyUI to be installed
"""

import sys
import traceback

def test_basic_imports():
    """Test that basic Python dependencies can be imported"""
    print("Testing basic dependencies...")
    try:
        import torch
        print(f"  [PASS] torch {torch.__version__}")
    except ImportError as e:
        print(f"  [FAIL] torch: {e}")
        return False

    try:
        import numpy as np
        print(f"  [PASS] numpy {np.__version__}")
    except ImportError as e:
        print(f"  [FAIL] numpy: {e}")
        return False

    try:
        from PIL import Image
        print(f"  [PASS] PIL")
    except ImportError as e:
        print(f"  [FAIL] PIL: {e}")
        return False

    return True

def test_config_imports():
    """Test that config modules can be imported"""
    print("\nTesting config imports...")
    try:
        from nodes.config import MODEL_REGISTRY, MASK_MODEL_REGISTRY
        print(f"  [PASS] Config imported ({len(MODEL_REGISTRY)} bbox models, {len(MASK_MODEL_REGISTRY)} mask models)")
        return True
    except Exception as e:
        print(f"  [FAIL] Config import failed: {e}")
        traceback.print_exc()
        return False

def test_utils_imports():
    """Test that utility modules can be imported"""
    print("\nTesting utils imports...")
    try:
        from nodes.utils.cache import MODEL_CACHE
        print(f"  [PASS] Cache imported")
    except Exception as e:
        print(f"  [FAIL] Cache import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from nodes.utils import draw_boxes, boxes_to_masks
        print(f"  [PASS] Utils imported")
    except Exception as e:
        print(f"  [FAIL] Utils import failed: {e}")
        traceback.print_exc()
        return False

    return True

def test_model_module_imports():
    """Test that model modules can be imported"""
    print("\nTesting model module imports...")
    modules = ['grounding_dino', 'owlv2', 'florence2', 'yolo_world',
               'florence2_seg', 'sa2va', 'sam2', 'visualizer']

    failed = []
    for module_name in modules:
        try:
            module = __import__(f'nodes.{module_name}', fromlist=[''])
            print(f"  [PASS] {module_name}")
        except Exception as e:
            print(f"  [FAIL] {module_name}: {e}")
            failed.append(module_name)

    return len(failed) == 0

def test_node_class_definitions():
    """Test that node classes are defined"""
    print("\nTesting node class definitions...")
    try:
        from nodes import (
            GroundingModelLoader,
            GroundingDetector,
            BboxVisualizer,
            DownLoadSAM2Model,
            Sam2Segment,
            GroundingMaskModelLoader,
            GroundingMaskDetector,
            BatchCropAndPadFromMask,
        )

        # Check they're not None
        classes = {
            'GroundingModelLoader': GroundingModelLoader,
            'GroundingDetector': GroundingDetector,
            'BboxVisualizer': BboxVisualizer,
            'DownLoadSAM2Model': DownLoadSAM2Model,
            'Sam2Segment': Sam2Segment,
            'GroundingMaskModelLoader': GroundingMaskModelLoader,
            'GroundingMaskDetector': GroundingMaskDetector,
            'BatchCropAndPadFromMask': BatchCropAndPadFromMask,
        }

        failed = []
        for name, cls in classes.items():
            if cls is None:
                print(f"  [FAIL] {name} is None!")
                failed.append(name)
            else:
                print(f"  [PASS] {name}")

        return len(failed) == 0

    except Exception as e:
        print(f"  [FAIL] Node import failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all smoke tests"""
    print("="*60)
    print("ComfyUI-Grounding Smoke Test")
    print("="*60)

    results = []

    # Run tests
    results.append(("Basic imports", test_basic_imports()))
    results.append(("Config imports", test_config_imports()))
    results.append(("Utils imports", test_utils_imports()))
    results.append(("Model modules", test_model_module_imports()))
    results.append(("Node classes", test_node_class_definitions()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("[PASS] All smoke tests passed!")
        sys.exit(0)
    else:
        print("[FAIL] Some smoke tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
