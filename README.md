# ComfyUI DreamLight Node

## Installation
1. Clone this repository into your ComfyUI's `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/APZmedia/ComfyUI-Dreamlight.git
```

2. Install dependencies:
```bash
cd ComfyUI-Dreamlight
pip install -r requirements.txt
```

3. Download DreamLight models:
```bash
# Create model directories
mkdir -p models/dreamlight/FLUX/transformer
mkdir -p models/dreamlight/CLIP

# Download FLUX model
curl -L "https://huggingface.co/LYAWWH/DreamLight/resolve/main/FLUX/transformer/model.pth?download=true" -o models/dreamlight/FLUX/transformer/model.pth

# Download CLIP model
curl -L "https://huggingface.co/LYAWWH/DreamLight/resolve/main/CLIP/models/config.json?download=true" -o models/dreamlight/CLIP/config.json
curl -L "https://huggingface.co/LYAWWH/DreamLight/resolve/main/CLIP/models/pytorch_model.bin?download=true" -o models/dreamlight/CLIP/pytorch_model.bin
```

## Node Usage
The node appears under "image/postprocessing" as "DreamLightNode"

**Inputs:**
- `foreground_image`: Subject to be relit (IMAGE)
- `background_image`: New background (IMAGE)
- `mask`: Foreground subject mask (MASK)
- `prompt`: Lighting guidance text
- `seed`: Random seed for reproducibility
- `resolution`: Output resolution (256-2048)
- `environment_map`: Optional environment map (IMAGE)

**Output:**
- `relit_image`: Final image with consistent lighting

## Example Workflow
1. Load `test_dreamlight_workflow.json` in ComfyUI
2. Provide these files in the project root:
   - `example_foreground.png`
   - `example_background.png`
   - `example_mask.png`

## Notes
- First run will take longer as it downloads additional dependencies
- Requires GPU with at least 8GB VRAM
- For optimal results, use 1024x1024 resolution
- Environment maps should be 360° equirectangular images

## Troubleshooting
If you encounter issues:
1. Verify all model files are in the correct directories
2. Ensure you have the required dependencies
3. Check console for specific error messages
4. Try reducing resolution if out of memory
