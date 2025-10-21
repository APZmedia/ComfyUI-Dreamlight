# RESTART REQUIRED

## The code has been updated, but you need to restart ComfyUI for the changes to take effect.

### Why?
ComfyUI loads Python modules at startup and caches them in memory. The error you're seeing is from the OLD code that was loaded when ComfyUI started.

### Steps to Fix:
1. **Stop ComfyUI** (close the application or press Ctrl+C in the terminal)
2. **Restart ComfyUI** 
3. **Try running the workflow again**

### What Changed:
- ✅ Fixed FLUX model download to use correct path: `FLUX/DreamLight-FLUX/transformer/model.pth`
- ✅ Fixed CLIP model download to use correct path: `CLIP/models/` directory
- ✅ Added support for `.safetensors` files (both FLUX and CLIP)
- ✅ Added recursive file scanning
- ✅ Better error messages and logging
- ✅ **CRITICAL FIX**: Code now fails fast if models can't be downloaded instead of trying to load non-existent files
- ✅ Added file existence checks before attempting to load models

### Expected Behavior After Restart:
You should see logs like:
- "Checking FLUX model files..."
- "No FLUX model files found. Downloading FLUX folder..."
- "FLUX download attempt 1/3..."
- List of files found with sizes

### If Still Getting Errors After Restart:
The model download may take a long time (23.8GB) and might timeout. If it fails:
1. Check your internet connection
2. Try manually downloading from: https://huggingface.co/LYAWWH/DreamLight
3. Place files in: `ComfyUI/models/dreamlight/FLUX/`

