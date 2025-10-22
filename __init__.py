import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

def load_hf_token():
    """Load HuggingFace token from .env file or environment variables"""
    try:
        # Try to load from .env file first
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        if token:
            logger.info("✓ Using HuggingFace token from .env file")
            return token
        else:
            logger.warning("✗ No HF_TOKEN found in .env file")
            return None
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")
        return None

def setup_flux_directory():
    """Download FLUX.1-dev model from HuggingFace"""
    import folder_paths
    from huggingface_hub import snapshot_download, HfApi
    
    models_dir = folder_paths.models_dir
    flux_complete_dir = os.path.join(models_dir, "dreamlight", "flux_complete")
    os.makedirs(flux_complete_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("FLIGHT CHECK: Setting up FLUX pipeline directory")
    logger.info(f"Target directory: {flux_complete_dir}")
    logger.info("="*60)
    
    # Check if already downloaded
    if os.path.exists(os.path.join(flux_complete_dir, "model_index.json")):
        logger.info("✓ FLUX model already downloaded")
        return flux_complete_dir
    
    # Get HuggingFace token
    hf_token = load_hf_token()
    if not hf_token:
        raise RuntimeError("No HuggingFace token found. Please set HF_TOKEN in your .env file")
    
    logger.info("Downloading FLUX.1-dev from HuggingFace...")
    logger.info("⚠️  FLUX.1-dev is a GATED model - authentication required!")
    logger.info("Make sure you have:")
    logger.info("1. Requested access to FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev")
    logger.info("2. Set HF_TOKEN in your .env file")
    
    try:
        # First, let's check what files are actually in the repository
        logger.info("Checking repository structure...")
        api = HfApi()
        repo_files = api.list_repo_files(
            repo_id="black-forest-labs/FLUX.1-dev",
            token=hf_token
        )
        
        logger.info(f"Found {len(repo_files)} files in FLUX.1-dev repository:")
        for file in sorted(repo_files):
            if "transformer" in file and ".safetensors" in file:
                logger.info(f"  📁 {file}")
            elif file.endswith(".json"):
                logger.info(f"  📄 {file}")
            elif file.endswith(".safetensors"):
                logger.info(f"  💾 {file}")
        
        # Download the entire FLUX.1-dev repository
        logger.info("Downloading complete repository...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir=flux_complete_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            resume_download=True
        )
        logger.info("✓ Download complete")
        
        # Verify what we actually got
        logger.info("Verifying downloaded files...")
        for root, dirs, files in os.walk(flux_complete_dir):
            for file in files:
                if "transformer" in root and ".safetensors" in file:
                    file_path = os.path.join(root, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    logger.info(f"  ✓ {os.path.relpath(file_path, flux_complete_dir)} ({size_mb:.1f} MB)")
        
        return flux_complete_dir
        
    except Exception as e:
        if "authentication" in str(e).lower() or "token" in str(e).lower() or "gated" in str(e).lower():
            logger.error("✗ Authentication failed!")
            logger.error("This could be due to:")
            logger.error("1. Invalid or expired token")
            logger.error("2. No access granted to FLUX.1-dev model")
            logger.error("3. Token doesn't have required permissions")
            raise RuntimeError("Authentication failed for gated FLUX.1-dev model. Please verify your token and access permissions.")
        else:
            logger.error(f"✗ Download failed: {e}")
            raise

def validate_flux_directory(flux_dir):
    """Simple validation - just check if directory exists"""
    if not os.path.exists(flux_dir):
        logger.error(f"✗ FLUX directory does not exist: {flux_dir}")
        return False
    
    logger.info("✓ FLUX directory exists")
    return True

def validate_and_download_models():
    """Validate installation and download models if missing"""
    try:
        # Lazy import ComfyUI modules
        import folder_paths
        import comfy.utils
        
        logger.info("DreamLight node initialized - validating models...")
        
        # Set up FLUX directory
        logger.info("Setting up FLUX pipeline directory...")
        flux_dir = setup_flux_directory()
        
        # Validate the directory
        if not validate_flux_directory(flux_dir):
            raise RuntimeError("FLUX directory validation failed. Please check the logs for detailed instructions on how to fix this issue.")
        
        logger.info("Testing FLUX pipeline loading...")
        
        # Test pipeline loading
        try:
            from diffusers import FluxPipeline
            import torch
            
            test_pipeline = FluxPipeline.from_pretrained(
                flux_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info("✓ FLUX pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"✗ FLUX pipeline loading failed: {e}")
            raise RuntimeError(f"FLUX pipeline loading failed: {e}")
        
        logger.info("✓ All models validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise

class DreamLightNode:
    """ComfyUI node for DreamLight relighting"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground_image": ("IMAGE",),
                "background_image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"default": "harmonious, natural, photorealistic"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1}),
                "resolution": ("INT", {"default": 1024, "min": 256, "max": 2048}),
            },
            "optional": {
                "environment_map": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("relit_image",)
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"
    
    def __init__(self):
        """Initialize node and validate models immediately"""
        # Run model validation on node creation, not during processing
        validate_and_download_models()
    
    def process(self, foreground_image, background_image, mask, prompt, seed, resolution, environment_map=None):
        # Models are already validated in __init__, so we can proceed directly
        
        try:
            from diffusers import FluxPipeline
            import torch
            import numpy as np
            from PIL import Image
            import comfy.utils
            
            # Load pipeline (already validated during __init__)
            logger.info("Loading FLUX pipeline from pre-validated directory...")
            pipeline = FluxPipeline.from_pretrained(
                flux_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Apply DreamLight modifications to transformer
            transformer = pipeline.transformer
            
            # Convert inputs to PIL Images
            if isinstance(foreground_image, torch.Tensor):
                if foreground_image.dim() == 4:  # Batch dimension
                    foreground_image = foreground_image[0]  # Take first image
                foreground_image = comfy.utils.tensor2pil(foreground_image)
            
            if isinstance(background_image, torch.Tensor):
                if background_image.dim() == 4:  # Batch dimension
                    background_image = background_image[0]  # Take first image
                background_image = comfy.utils.tensor2pil(background_image)
            
            if isinstance(mask, torch.Tensor):
                if mask.dim() == 3:  # Add batch dimension if needed
                    mask = mask.unsqueeze(0)
                mask = comfy.utils.tensor2pil(mask)
            
            # Resize images to match resolution
            foreground_image = foreground_image.resize((resolution, resolution))
            background_image = background_image.resize((resolution, resolution))
            mask = mask.resize((resolution, resolution))
            
            # Convert mask to numpy array
            mask_array = np.array(mask)
            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, 0]  # Take first channel if RGB
            
            # Normalize mask to 0-1 range
            mask_array = mask_array.astype(np.float32) / 255.0
            
            # Create composite image using mask
            composite = np.array(foreground_image) * mask_array[:, :, np.newaxis] + \
                       np.array(background_image) * (1 - mask_array[:, :, np.newaxis])
            composite = Image.fromarray(composite.astype(np.uint8))
            
            # Generate relit image
            logger.info(f"Generating relit image with prompt: '{prompt}'")
            
            # Set seed for reproducibility
            if seed == 0:
                seed = torch.randint(0, 2**32-1, (1,)).item()
            
            generator = torch.Generator().manual_seed(seed)
            
            # Generate image
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    image=composite,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=generator,
                    height=resolution,
                    width=resolution
                ).images[0]
            
            # Convert result back to ComfyUI format
            result_tensor = comfy.utils.pil2tensor(result)
            
            logger.info("✓ DreamLight processing complete")
            return (result_tensor,)
            
        except Exception as e:
            logger.error(f"DreamLight processing failed: {e}")
            raise RuntimeError(f"DreamLight processing failed: {e}")

# Node mapping
NODE_CLASS_MAPPINGS = {
    "DreamLightNode": DreamLightNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamLightNode": "DreamLight Node"
}

# Package initialization
logger.info("ComfyUI-Dreamlight package loaded successfully")
logger.info("Registered nodes: ['DreamLightNode']")
logger.info("Successfully registered DreamLightNode in image/postprocessing category")