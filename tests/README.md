# ComfyUI-Grounding Tests

## Running Tests

**Unit tests (fast):**
```bash
pytest tests/unit/ -v
```

**Real model tests (slow, downloads models):**
```bash
pytest tests/integration_real/ -v -m real_model
```

**Real model tests with GPU:**
```bash
pytest tests/integration_real/ -v -m real_model --use-gpu
```

## Test List

### Unit Tests (6 tests)
- `test_node_registration.py` - Node class registration
- `test_model_registry.py` - Model registry validation
- `test_input_validation.py` - Input validation
- `test_cache.py` - Model caching
- `test_conversions.py` - Format conversions
- `test_drawing.py` - Bounding box drawing

### Real Model Tests (16 tests)

**Florence-2** (7 tests)
- Load with eager attention
- Load with SDPA attention
- Detection
- Model caching
- Attention implementations comparison
- Mask generation
- Different prompts

**GroundingDINO** (3 tests)
- Load SwinT model
- Detection
- Model caching

**OWLv2** (3 tests)
- Load Base Patch16 model
- Detection
- Model caching

**YOLO-World** (3 tests)
- Load v8s model
- Detection
- Model caching
