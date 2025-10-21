@@# ComfyUI Custom Node Development Guide

## Getting Started
1. Create project structure:
```bash
mkdir comfyui-custom-node && cd comfyui-custom-node
mkdir -p nodes utils docs test-images
touch setup.py MANIFEST.in requirements.txt
```

2. Initialize node module:
```python
# nodes/__init__.py
NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']
```

## Project Structure Convention

## Repository Structure

```
custom_nodes/Comfyui-LightDirection-estimation/
├── nodes/                  # Node implementations
│   ├── enhanced_light_estimator.py  # Main analysis node
│   └── ...                 # Supplementary nodes
├── utils/                  # Processing utilities
│   ├── luma_mask_processor.py       # Tensor conversion helpers
│   └── light_estimator.py  # Core estimation logic
├── docs/                   # Technical documentation
├── test-images/            # Validation assets
├── test_modular_nodes.py   # Compatibility test suite
├── setup.py               # Packaging configuration
├── MANIFEST.in            # Distribution includes
└── requirements.txt       # Dependency spec
```

## Node Implementation Guidelines

### Basic Node Template
```python
class CustomNode:
    CATEGORY = "MyNodes/Category"  # Group in node menu
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0})
        },
        "optional": {
            "mask": ("MASK",)
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    def process(self, image, strength):
        # Always maintain batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return (image * strength,)
```

### Node Registration Patterns

### Entry Point Registration (setup.py)
```python
# setup.py configuration
entry_points={
    "comfyui.custom_nodes": [
        "LightDirection = nodes:NODE_CLASS_MAPPINGS"
    ]
}
```

### Direct Class Mapping
```python
# __init__.py or nodes/__init__.py 
NODE_CLASS_MAPPINGS = {
    "EnhancedLightEstimator": EnhancedLightEstimator,
    "IREAnalysisNode": IREAnalysisNode
}
```

## Core Node Structure Requirements

```python
class CustomLightNode:
    CATEGORY = "Custom/Lighting"  # Grouping in node menu
    INPUT_TYPES = {  # Define input parameters
        "required": {
            "image": ("IMAGE",),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0})
        }
    }
    RETURN_TYPES = ("IMAGE", "MASK")  # Output types
    FUNCTION = "process"  # Entry point method

    def process(self, image, threshold):
        # Tensor shape handling
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # Processing logic
        return (processed_image, mask)
```

## Essential Tensor Handling Rules

1. **Input Validation**
```python
def validate_image_input(image):
    if image.dtype != torch.float32:
        raise ValueError("Image tensor must be float32")
    if image.shape[-1] != 3:
        raise ValueError("Last dimension must be 3 channels (RGB)")
```

2. **Batch Dimension Management**
```python
# Accept either single or batch images
def normalize_batch(image):
    return image if image.ndim == 4 else image.unsqueeze(0)
```

3. **Device Awareness**
```python
def process_on_cpu(image):
    was_cuda = image.is_cuda
    image = image.cpu()
    # ... processing ...
    return image.cuda() if was_cuda else image
```

### Conversion Protocol

### Image Tensors (NHWC)
```python
# Conversion from PIL (RGB)
def pil_to_tensor(pil_img):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]

# Back to PIL
def tensor_to_pil(tensor):
    arr = tensor.squeeze().clamp(0,1).cpu().numpy() * 255
    return Image.fromarray(arr.astype(np.uint8))
```

### Mask Tensors (BCHW Remapping)
```python
def remap_mask(mask_tensor):
    # Ensure proper batch/channel dimensions
    if mask_tensor.ndim == 2:
        return mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return mask_tensor.permute(0, 3, 1, 2)  # NHWC → NCHW
```

## Compatibility Safeguards

1. **Shape Validation**
```python
def validate_shapes(tensor):
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Invalid tensor rank {tensor.ndim}")
```

2. **Batch Consistency**
```python
def match_batch_size(*tensors):
    batch_sizes = [t.shape[0] for t in tensors]
    if len(set(batch_sizes)) > 1:
        raise ValueError(f"Inconsistent batch sizes {batch_sizes}")
```

3. **Normalization Guards**
```python
def normalize_tensor(t):
    if t.max() > 1.0 or t.min() < 0.0:
        return t.sub(t.min()).div(t.max() - t.min())
    return t
```

## Packaging Requirements (setup.py)

```python
setup(
    name="Comfyui-LightDirection-estimation",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.1",
        "numpy>=1.24.3",
        "scipy>=1.10.1"
    ],
    entry_points={
        "comfyui.custom_nodes": 
            "LightDirection = nodes:CLAZZ"
    }
)
```

## Test Patterns

```python
class TestTensorCompatibility(unittest.TestCase):
    def test_image_shape_validation(self):
        # Valid NHWC tensor
        valid_tensor = torch.rand(1, 512, 512, 3)
        
        # Invalid tensor
        invalid_tensor = torch.rand(512, 512)
        with self.assertRaises(ValueError):
            validate_shapes(invalid_tensor)
```

## Path Handling Guidelines

1. **Resource Path Resolution**
```python
import os
from pathlib import Path

# Get current node directory
NODE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
ICON_PATH = NODE_DIR / "assets" / "node_icon.png"
```

2. **Portable Installation Support**
```python
# Check for portable installation
config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
if not os.path.exists(config_path):
    config_path = os.path.join(os.path.dirspath(__file__), "config.json")
```

3. **Cross-Platform Paths**
```python
# Always use forward slashes and Pathlib
model_path = str(NODE_DIR / "models" / "light_estimator.pt")
```

## Dependency Management

### Installation Best Practices
```python
# setup.py configuration
install_requires=[
    "torch>=2.0.1",
    "numpy>=1.24.3",
    "scipy>=1.10.1; sys_platform != 'win32'",  # Conditional dependencies
    "comfyui-utils @ git+https://github.com/ComfyUI/utils.git@main"
]
```

1. **Version Pinning**
```text
# requirements.txt
torch==2.1.0
numpy>=1.24.3,<2.0.0
```

2. **Optional Dependencies**
```python
try:
    import optional_package
except ImportError:
    print("Optional feature disabled: package not found")
```

3. **ComfyUI Version Checks**
```python
import comfy.utils
MIN_COMFY_VERSION = (1, 2, 0)
if not comfy.utils.version_check(comfy.__version__, MIN_COMFY_VERSION):
    raise ImportError(f"Requires ComfyUI version {MIN_COMFY_VERSION} or higher")
```

## Compliance Verification

### Tensor Validation
- [ ] NHWC format for images (Batch, Height, Width, Channels)
- [ ] Single-channel masks as [B,H,W] tensors
- [ ] Normalized values (0-1 for images, 0-1 for masks)

### Node Interface
- [ ] INPUT_TYPES defined with proper type annotations
- [ ] RETURN_TYPES matches output tuple
- [ ] FUNCTION points to entry method
- [ ] CATEGORY defined for organization

### Packaging Requirements
- [ ] setup.py includes entry_points for ComfyUI
- [ ] MANIFEST.in includes all node files
- [ ] requirements.txt pins critical versions

## Distribution & Installation

### Packaging Best Practices
```python
# setup.py minimum configuration
from setuptools import setup, find_packages

setup(
    name="comfyui-custom-node",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["torch>=2.0.0"],
    entry_points={"comfyui.custom_nodes": ["custom = nodes:NODE_CLASS_MAPPINGS"]}
)
```

### Dependency Management
1. **Core Requirements**
```python
# requirements.txt
comfyui>=1.2.0
torch>=2.0.1
```

2. **Platform-Specific Dependencies**
```python
# setup.py conditional install
install_requires=[
    "numpy",
    "windows-curses; sys_platform == 'win32'"
]
```

## Maintenance Tips

1. **Version Compatibility**
```python
# Check ComfyUI version during node registration
import comfy
if comfy.__version__ < "1.2.0":
    print("WARNING: Some features may require newer ComfyUI version")
```

2. **Deprecation Strategy**
```python
# Maintain backwards compatibility
class LegacyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}
        
    # ... rest of implementation ...
```

3. **Documentation Standards**
```markdown
<!-- docs/usage.md -->
## Node Parameters
- `image`: Input image tensor (RGB, 0-1 normalized)
- `strength`: Effect intensity multiplier
```

- [ ] Tensor shapes match ComfyUI expectations (NHWC for images)
- [ ] Mask tensors use single-channel format
- [ ] Batch dimensions preserved through processing
- [ ] All node outputs wrapped in tuples
- [ ] Package manifest includes node files
- [ ] Tests verify input/output tensor formats
